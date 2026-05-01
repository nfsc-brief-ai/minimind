# SageMaker pretraining (MiniMind)

This folder contains a minimal SageMaker integration for running `trainer/train_pretrain.py` on AWS.

## Files

- `sagemaker/entrypoint_pretrain.py`
  - Runs inside the SageMaker training container.
  - Maps SageMaker directories to `train_pretrain.py` arguments.
  - Automatically uses `torchrun` on **single-node multi-GPU** instances (when `SM_NUM_GPUS >= 2`).
- `sagemaker/submit_pretrain.py`
  - Example client-side script that creates a SageMaker PyTorch Estimator and launches a training job.

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

### 2) Launch SageMaker training

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

## Resume / checkpointing

If you want to resume after an interruption, set:

- `from_resume=1` (hyperparameter)

This loads the resume bundle from `--ckpt_dir` and continues from the saved epoch/step.

## Notes / common pitfalls

- **Dependencies**: `submit_pretrain.py` uses the official SageMaker PyTorch container. If you need extra Python packages beyond the container defaults, you can either:
  - build a custom image, or
  - extend the Estimator with an install step (project-specific).
- **Tokenizer path**: `trainer_utils.init_model()` loads tokenizer from the repo’s `model/` directory. `source_dir="."` in the estimator uploads the repo so that path exists in the container.

