# SageMaker pretraining (MiniMind)

This folder contains a minimal SageMaker integration for running `trainer/train_pretrain.py` on AWS.

## Files

- `sagemaker/entrypoint_pretrain.py`
  - Runs inside the SageMaker training container.
  - Maps SageMaker directories to `train_pretrain.py` arguments.
  - Automatically uses `torchrun` on **single-node multi-GPU** instances (when `SM_NUM_GPUS >= 2`).
- `sagemaker/submit_pretrain.py`
  - Example client-side script that creates a SageMaker PyTorch Estimator and launches a training job.
- `sagemaker/training_job_status.py`
  - CLI to describe one training job or list recent jobs (status, times, failure reason, instance type).
- `sagemaker/requirements-client.txt`
  - Minimal deps for **local** submit/status/upload only (`sagemaker` SDK—no PyTorch in your laptop venv).
- `sagemaker/setup_client_venv.sh`
  - Creates `~/.venvs/minimind-sagemaker` **outside** the repo so `submit_pretrain` packaging stays small.

## Lightweight venv for submit (avoid huge `.venv` in the repo)

`submit_pretrain.py` uploads the **entire repo root** as `source_dir`. A full project `.venv` under the repo is included in that tarball and slows uploads.

Use a **separate, small** venv with only the SageMaker client stack:

```bash
bash sagemaker/setup_client_venv.sh
source ~/.venvs/minimind-sagemaker/bin/activate
```

Override install location: `MINIMIND_SM_VENV=/path/to/venv bash sagemaker/setup_client_venv.sh`

Then use that shell when you run `submit_pretrain.py`, `training_job_status.py`, and `upload_train_data.py`. Keep your large training `.venv` for local `trainer/` work, or rename/move it so it is not under the repo root when you submit.

## Data format and filename

`trainer/train_pretrain.py` uses `dataset/PretrainDataset`, which expects a JSONL file where each row has a `text` field:

```json
{"text": "some training text"}
{"text": "more training text"}
```

Recommended filename:

- Put your data in S3 as **`pretrain.jsonl`**.
- In SageMaker it will be available under the `train` channel as:
  - `/opt/ml/input/data/train/pretrain.jsonl`

You can override the filename via `--data_file` (passed as an Estimator hyperparameter).

## What gets written where

SageMaker provides standard directories inside the container:

- **Training data**: `SM_CHANNEL_TRAIN` (defaults to `/opt/ml/input/data/train`)
- **Model artifacts** (uploaded by SageMaker at job end): `SM_MODEL_DIR` (defaults to `/opt/ml/model`)
- **Job output data**: `SM_OUTPUT_DATA_DIR` (defaults to `/opt/ml/output/data`)

The entrypoint maps these to your trainer:

- `--data_path`: resolved to `SM_CHANNEL_TRAIN/pretrain.jsonl` (or `--data_file`)
- `--save_dir`: `SM_MODEL_DIR`
  - weights saved as: `<save_dir>/<save_weight>_<hidden_size>[_moe].pth`
- `--ckpt_dir`: `SM_OUTPUT_DATA_DIR/checkpoints`
  - resume bundle saved as: `<ckpt_dir>/<save_weight>_<hidden_size>[_moe]_resume.pth`

## Running a job (example)

### 1) Upload your dataset to S3

Upload `pretrain.jsonl` to an S3 prefix (example):

- `s3://YOUR_BUCKET/minimind/pretrain/`
  - contains `pretrain.jsonl`

You can do this either with the AWS CLI (fastest) or with the included Python helper.

#### Option A: AWS CLI

Upload a single file:

```bash
aws s3 cp ./data/pretrain.jsonl s3://YOUR_BUCKET/minimind/pretrain/pretrain.jsonl
```

Upload/sync a whole folder (useful if you have shards like `pretrain-00001.jsonl`, etc.):

```bash
aws s3 sync ./data/ s3://YOUR_BUCKET/minimind/pretrain/
```

#### Option B: Python (`boto3`) helper

Single file:

```bash
python sagemaker/upload_train_data.py \
  --local_path ./data/pretrain.jsonl \
  --s3_uri s3://YOUR_BUCKET/minimind/pretrain/
```

Directory sync:

```bash
python sagemaker/upload_train_data.py \
  --local_path ./data/ \
  --s3_uri s3://YOUR_BUCKET/minimind/pretrain/
```

### 2) Launch SageMaker training

Install the **SageMaker Python SDK** on the machine where you run the submit script (not inside the training container). Prefer the minimal client requirements:

```bash
pip install -r sagemaker/requirements-client.txt
```

Or use `pip install -r sagemaker/requirements-submit.txt` (includes the same deps).

Run from a machine with AWS credentials configured (local, EC2, or a notebook):

```bash
python sagemaker/submit_pretrain.py \
  --role_arn arn:aws:iam::<acct-id>:role/<SageMakerExecutionRole> \
  --s3_train_uri s3://YOUR_BUCKET/minimind/pretrain/ \
  --instance_type ml.g4dn.xlarge
```

The included submit script is tuned for `ml.g4dn.xlarge` (1×T4 16GB):

- `dtype=float16`
- `batch_size=8`
- `accumulation_steps=8`

If you use a multi-GPU instance (e.g. `ml.p4d.24xlarge`, `ml.p3.16xlarge`, etc.), `entrypoint_pretrain.py` will automatically launch with `torchrun --nproc_per_node=$SM_NUM_GPUS`.

### Logs and progress (why the terminal may look idle)

`estimator.fit(...)` **blocks until the job finishes** and streams **CloudWatch** training logs to your terminal by default. There is often a **multi‑minute gap** while SageMaker provisions the instance and downloads your image/data—nothing is wrong if nothing prints yet.

- **More frequent training metrics**: lower `--log_interval` (steps between loss lines), for example:

```bash
python sagemaker/submit_pretrain.py ... --log_interval 50
```

- **Don’t block this shell**: start the job and exit immediately; open logs in the AWS console instead:

```bash
python sagemaker/submit_pretrain.py ... --no_wait
```

Then use **SageMaker console → Training → Training jobs → your job → Logs**, or **CloudWatch Logs** log group `/aws/sagemaker/TrainingJobs`.

The submit script sets **`PYTHONUNBUFFERED=1`** so stdout reaches CloudWatch with less batching delay.

Check status from the CLI (same credentials as `aws` / `submit_pretrain.py`):

```bash
python sagemaker/training_job_status.py --job-name YOUR_TRAINING_JOB_NAME

python sagemaker/training_job_status.py --list --max-results 15

python sagemaker/training_job_status.py --job-name YOUR_TRAINING_JOB_NAME --json
```

## Resume / checkpointing

If you want to resume after an interruption, set:

- `from_resume=1` (hyperparameter)

This loads the resume bundle from `--ckpt_dir` and continues from the saved epoch/step.

## Notes / common pitfalls

### Submit script seems hung / `training_job_status.py --list` shows nothing

1. **No training job until upload finishes**: the SageMaker Python SDK **packages your whole `source_dir` (repo root)**, uploads `sourcedir.tar.gz` to the default SageMaker S3 bucket, **then** calls `CreateTrainingJob`. Until that finishes, **there is no training job** to list—only slow upload progress locally (often dominated by a large `.venv` under the repo).
2. **`training_job_status.py` uses your default AWS region** unless you pass `--region`. If submit used `us-west-2` (via config) but you list `us-east-1`, the list looks empty. Align regions: `python sagemaker/training_job_status.py --list --region us-west-2` (same account as `submit_pretrain.py`; both scripts print account + region).
3. **`--role_arn` must be a role**: user ARNs (`...:user/...`) are rejected by `submit_pretrain.py`. Use `arn:aws:iam::<acct>:role/<ExecutionRole>`.

- **`ModuleNotFoundError: No module named 'sagemaker'`**: install client deps with `pip install -r sagemaker/requirements-submit.txt` (or `pip install sagemaker`).
- **`--role_arn`**: must be an IAM **role** ARN that SageMaker can assume for training (typically `arn:aws:iam::<acct>:role/<SageMakerExecutionRole>`), not an IAM user ARN.
- **Dependencies**: `submit_pretrain.py` uses the official SageMaker PyTorch container. If you need extra Python packages beyond the container defaults, you can either:
  - build a custom image, or
  - extend the Estimator with an install step (project-specific).
- **Tokenizer path**: `trainer_utils.init_model()` loads tokenizer from the repo’s `model/` directory. `source_dir="."` in the estimator uploads the repo so that path exists in the container.

