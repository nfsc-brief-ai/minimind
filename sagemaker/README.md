# SageMaker pretrain (MiniMind)

Scripts:

- `submit_pretrain.py` — package minimal code and start training
- `upload_train_data.py` — push JSONL (or a folder) to S3
- `training_job_status.py` — list or describe training jobs

The submit script **only** packages:

- `trainer/train_pretrain.py`, `trainer/trainer_utils.py`
- `dataset/lm_dataset.py`
- `model/` (code + tokenizer files)
- `entrypoint_pretrain.py`, `requirements-training.txt`

Python deps for training are installed **on the GPU instance** from `requirements-training.txt` when the job starts (PyTorch comes from the SageMaker image).

## Setup (local)

```bash
pip install -r sagemaker/requirements.txt
```

Use an IAM **role** ARN (not a user): `arn:aws:iam::<acct>:role/<SageMakerExecutionRole>`.

## Data

JSONL with a `text` field per line. Upload with the helper or AWS CLI:

```bash
python sagemaker/upload_train_data.py \
  --local_path ./pretrain.jsonl \
  --s3_uri s3://YOUR_BUCKET/pretrain/

aws s3 cp ./pretrain.jsonl s3://YOUR_BUCKET/pretrain/pretrain.jsonl
```

Use the same bucket/prefix (or object) as `--s3_train_uri` when you submit.

## Run

```bash
python sagemaker/submit_pretrain.py \
  --role_arn arn:aws:iam::<acct>:role/<Role> \
  --s3_train_uri s3://YOUR_BUCKET/pretrain/ \
  --instance_type ml.g4dn.xlarge
```

Optional: `--region`, `--profile`, `--job_name`, `--no_wait`.

### Job status

```bash
python sagemaker/training_job_status.py --list
python sagemaker/training_job_status.py --job-name YOUR_JOB_NAME
```

Use `--region` / `--profile` if needed (same as submit).

Artifacts: model → `SM_MODEL_DIR`; checkpoints → `SM_OUTPUT_DATA_DIR/checkpoints`. Resume: set hyperparameter `from_resume=1` in `submit_pretrain.py` if you fork and edit.
