"""
Example SageMaker submit script (single-node multi-GPU).

Run this from a machine with AWS credentials configured:
  python sagemaker/submit_pretrain.py --role_arn <role> --s3_train_uri s3://.../pretrain/ --instance_type ml.g4dn.xlarge
"""

import argparse
from pathlib import Path

# Repo root (parent of sagemaker/). Entry point path is relative to source_dir.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _validate_training_role_arn(role_arn: str) -> None:
    """SageMaker CreateTrainingJob requires an IAM role ARN, not a user/group."""
    if ":user/" in role_arn or ":group/" in role_arn:
        raise SystemExit(
            "Invalid --role_arn: pass an IAM role ARN for SageMaker to assume, e.g.\n"
            "  arn:aws:iam::<account-id>:role/<SageMakerExecutionRole>\n"
            f"You passed a principal that is not a role: {role_arn!r}\n"
            "Create an execution role in IAM (or use SageMaker console → Domains / Notebook "
            "execution role) and grant it access to your S3 data and ECR."
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role_arn", required=True, type=str)
    p.add_argument("--s3_train_uri", required=True, type=str, help="S3 prefix or object for the 'train' channel")
    p.add_argument("--instance_type", default="ml.g4dn.xlarge", type=str)
    p.add_argument("--instance_count", default=1, type=int)
    p.add_argument("--job_name", default=None, type=str)
    p.add_argument(
        "--no_wait",
        action="store_true",
        help="Start the job and exit immediately (no live log stream here). "
        "Follow progress in SageMaker console or CloudWatch Logs.",
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Trainer progress log frequency in steps (passed through to train_pretrain.py). Lower = more frequent lines.",
    )
    p.add_argument("--region", type=str, default=None, help="AWS region for SageMaker and S3 (default: env/profile).")
    p.add_argument("--profile", type=str, default=None, help="AWS named profile for credentials.")
    args = p.parse_args()

    _validate_training_role_arn(args.role_arn)

    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch

    boto_sess = boto3.Session(region_name=args.region, profile_name=args.profile)
    sess = sagemaker.Session(boto_session=boto_sess)

    sts = boto_sess.client("sts")
    ident = sts.get_caller_identity()
    print(
        f"Caller: {ident.get('Arn')}  Account: {ident.get('Account')}  "
        f"SageMaker region: {sess.boto_region_name}",
        flush=True,
    )
    venv_dir = _REPO_ROOT / ".venv"
    if venv_dir.is_dir():
        print(
            "Note: .venv/ is under the repo root. The SDK uploads source_dir to S3 before "
            "creating a training job; a large .venv can make this step very slow. "
            "Consider moving the venv outside the repo or using a minimal copy.",
            flush=True,
        )
    print(
        "Packaging repo and uploading to S3 (no training job exists in AWS until this finishes)...",
        flush=True,
    )

    estimator = PyTorch(
        entry_point="sagemaker/entrypoint_pretrain.py",
        source_dir=str(_REPO_ROOT),  # uploads the whole repo so trainer/ and model/ are available
        role=args.role_arn,
        sagemaker_session=sess,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version="2.3",
        py_version="py311",
        # Flush stdout/stderr promptly so CloudWatch shows lines without long buffering delays.
        environment={"PYTHONUNBUFFERED": "1"},
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
            "log_interval": args.log_interval,
        },
        # Single-GPU instance: no need for DDP.
        # Keep this simple: use the repo's requirements via a custom image, or add dependencies here.
        # dependencies=[],  # optional: add a requirements installer script
    )

    inputs = {"train": args.s3_train_uri}
    estimator.fit(inputs=inputs, job_name=args.job_name, wait=not args.no_wait)
    if estimator.latest_training_job:
        name = estimator.latest_training_job.name
        print(f"Training job name: {name}", flush=True)
        if args.no_wait:
            print(
                "Watch logs: SageMaker console → Training → Training jobs → Logs; "
                "or CloudWatch log group /aws/sagemaker/TrainingJobs",
                flush=True,
            )


if __name__ == "__main__":
    main()

