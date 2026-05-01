"""
Example SageMaker submit script (single-node multi-GPU).

Run this from a machine with AWS credentials configured:
  python sagemaker/submit_pretrain.py --role_arn <role> --s3_train_uri s3://.../pretrain/ --instance_type ml.g4dn.xlarge
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role_arn", required=True, type=str)
    p.add_argument("--s3_train_uri", required=True, type=str, help="S3 prefix or object for the 'train' channel")
    p.add_argument("--instance_type", default="ml.g4dn.xlarge", type=str)
    p.add_argument("--instance_count", default=1, type=int)
    p.add_argument("--job_name", default=None, type=str)
    args = p.parse_args()

    import sagemaker
    from sagemaker.pytorch import PyTorch

    sess = sagemaker.Session()

    estimator = PyTorch(
        entry_point="entrypoint_pretrain.py",
        source_dir=".",  # uploads the whole repo so trainer/ and model/ are available
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version="2.3",
        py_version="py311",
        # SageMaker will pass these as CLI args to entrypoint_pretrain.py
        hyperparameters={
            # Data file inside the train channel (recommended default)
            "data_file": "pretrain.jsonl",
            # Trainer hyperparameters (forwarded into trainer/train_pretrain.py)
            "epochs": 2,
            # g4dn.xlarge (T4 16GB) tends to need smaller micro-batches.
            "batch_size": 8,
            "learning_rate": 5e-4,
            # T4 does fp16 well; bf16 is usually not supported/beneficial here.
            "dtype": "float16",
            "max_seq_len": 340,
            "hidden_size": 768,
            "num_hidden_layers": 8,
            "use_moe": 0,
            # Effective batch ~= batch_size * accumulation_steps
            "accumulation_steps": 8,
            "save_weight": "pretrain",
            "from_weight": "none",
            "from_resume": 0,
            "use_compile": 0,
            "num_workers": 2,
        },
        # Single-GPU instance: no need for DDP.
        # Keep this simple: use the repo's requirements via a custom image, or add dependencies here.
        # dependencies=[],  # optional: add a requirements installer script
    )

    inputs = {"train": args.s3_train_uri}
    estimator.fit(inputs=inputs, job_name=args.job_name)


if __name__ == "__main__":
    main()

