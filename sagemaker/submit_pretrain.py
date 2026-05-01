"""
Launch SageMaker pretrain: builds a small code bundle (trainer + dataset + model + entrypoint), uploads, runs training.
Local deps: pip install -r sagemaker/requirements.txt
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAGEMAKER_DIR = Path(__file__).resolve().parent


def _validate_training_role_arn(role_arn: str) -> None:
    if ":user/" in role_arn or ":group/" in role_arn:
        raise SystemExit(
            "Use an IAM role ARN, e.g. arn:aws:iam::<id>:role/<SageMakerExecutionRole>\n"
            f"Got: {role_arn!r}"
        )


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def _build_pretrain_bundle(repo_root: Path, dest: Path) -> None:
    """Only files needed for train_pretrain: no full repo, no .venv."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    t_train = repo_root / "trainer" / "train_pretrain.py"
    t_utils = repo_root / "trainer" / "trainer_utils.py"
    lm = repo_root / "dataset" / "lm_dataset.py"
    model_dir = repo_root / "model"
    entry = _SAGEMAKER_DIR / "entrypoint_pretrain.py"
    req = _SAGEMAKER_DIR / "requirements-training.txt"
    for p in (t_train, t_utils, lm, model_dir, entry, req):
        if not p.exists():
            raise FileNotFoundError(p)

    td = dest / "trainer"
    td.mkdir()
    shutil.copy2(t_train, td / "train_pretrain.py")
    shutil.copy2(t_utils, td / "trainer_utils.py")
    _touch(td / "__init__.py")

    (dest / "dataset").mkdir()
    shutil.copy2(lm, dest / "dataset" / "lm_dataset.py")
    _touch(dest / "dataset" / "__init__.py")

    shutil.copytree(
        model_dir,
        dest / "model",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=False,
    )
    _touch(dest / "model" / "__init__.py")

    shutil.copy2(entry, dest / "entrypoint_pretrain.py")
    shutil.copy2(req, dest / "requirements-training.txt")


def main() -> None:
    p = argparse.ArgumentParser(description="Submit MiniMind pretrain to SageMaker (minimal bundle only).")
    p.add_argument("--role_arn", required=True)
    p.add_argument("--s3_train_uri", required=True, help="S3 URI for the train channel (prefix or object)")
    p.add_argument("--instance_type", default="ml.g4dn.xlarge")
    p.add_argument("--job_name", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--no_wait", action="store_true", help="Return after job is started; watch logs in console")
    args = p.parse_args()

    _validate_training_role_arn(args.role_arn)

    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch

    boto_sess = boto3.Session(region_name=args.region, profile_name=args.profile)
    sess = sagemaker.Session(boto_session=boto_sess)

    tmp = Path(tempfile.mkdtemp(prefix="minimind-smbundle-"))
    try:
        _build_pretrain_bundle(_REPO_ROOT, tmp)
        print(f"Bundle ready → uploading ({tmp})", flush=True)

        estimator = PyTorch(
            entry_point="entrypoint_pretrain.py",
            source_dir=str(tmp),
            role=args.role_arn,
            sagemaker_session=sess,
            instance_type=args.instance_type,
            instance_count=1,
            framework_version="2.3",
            py_version="py311",
            environment={"PYTHONUNBUFFERED": "1"},
            hyperparameters={
                "data_file": "pretrain.jsonl",
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 5e-4,
                "dtype": "float16",
                "max_seq_len": 340,
                "hidden_size": 768,
                "num_hidden_layers": 8,
                "use_moe": 0,
                "accumulation_steps": 8,
                "save_weight": "pretrain",
                "from_weight": "none",
                "from_resume": 0,
                "use_compile": 0,
                "num_workers": 2,
                "log_interval": 100,
            },
        )
        estimator.fit(
            inputs={"train": args.s3_train_uri},
            job_name=args.job_name,
            wait=not args.no_wait,
        )
        if estimator.latest_training_job:
            print(f"Job: {estimator.latest_training_job.name}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
