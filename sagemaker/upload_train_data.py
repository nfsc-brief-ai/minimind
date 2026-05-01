#!/usr/bin/env python3
"""
Upload MiniMind pretraining data to S3 for SageMaker.

Supports:
  - Uploading a single file (e.g. pretrain.jsonl) to an S3 prefix/object
  - Syncing a directory to an S3 prefix (best for datasets with multiple shards)

Examples:
  python sagemaker/upload_train_data.py \
    --local_path ./data/pretrain.jsonl \
    --s3_uri s3://YOUR_BUCKET/minimind/pretrain/

  python sagemaker/upload_train_data.py \
    --local_path ./data/ \
    --s3_uri s3://YOUR_BUCKET/minimind/pretrain/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


def _ensure_trailing_slash(s3_uri: str) -> str:
    return s3_uri if s3_uri.endswith("/") else (s3_uri + "/")


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"--s3_uri must start with s3://, got: {s3_uri!r}")
    rest = s3_uri[len("s3://") :]
    if "/" not in rest:
        return rest, ""
    bucket, key = rest.split("/", 1)
    return bucket, key


def _iter_files(root: Path) -> Iterable[Path]:
    # Recursively yield files, skipping common junk.
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name in {".DS_Store"}:
            continue
        if any(part in {".git", "__pycache__"} for part in p.parts):
            continue
        yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload training data to S3 for SageMaker.")
    parser.add_argument(
        "--local_path",
        required=True,
        type=str,
        help="Local file or directory to upload (e.g. ./data/pretrain.jsonl or ./data/).",
    )
    parser.add_argument(
        "--s3_uri",
        required=True,
        type=str,
        help="Destination S3 URI (prefix or object), e.g. s3://bucket/minimind/pretrain/ .",
    )
    parser.add_argument(
        "--region",
        default=None,
        type=str,
        help="Optional AWS region override (otherwise uses normal AWS config resolution).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    args = parser.parse_args()

    local = Path(args.local_path).expanduser().resolve()
    if not local.exists():
        raise FileNotFoundError(f"--local_path does not exist: {str(local)!r}")

    s3_uri = args.s3_uri
    bucket, key_prefix = _parse_s3_uri(s3_uri)

    # If local is a directory, destination must be a prefix.
    if local.is_dir():
        s3_uri = _ensure_trailing_slash(s3_uri)
        bucket, key_prefix = _parse_s3_uri(s3_uri)
        key_prefix = _ensure_trailing_slash(key_prefix) if key_prefix else ""
    else:
        # For a file, allow either a prefix (".../") or an object key (".../file.jsonl").
        if s3_uri.endswith("/"):
            key_prefix = _ensure_trailing_slash(key_prefix)
            key_prefix = key_prefix + local.name

    if args.dry_run:
        if local.is_file():
            print(f"DRY RUN: would upload {local} -> s3://{bucket}/{key_prefix}")
            return 0
        base_prefix = _ensure_trailing_slash(key_prefix) if key_prefix else ""
        for f in _iter_files(local):
            rel = f.relative_to(local).as_posix()
            print(f"DRY RUN: would upload {f} -> s3://{bucket}/{base_prefix}{rel}")
        return 0

    try:
        import boto3
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "boto3 is required for upload_train_data.py. "
            "Install it with: pip install boto3"
        ) from e

    session = boto3.session.Session(region_name=args.region) if args.region else boto3.session.Session()
    s3 = session.client("s3")

    if local.is_file():
        print(f"Uploading {local} -> s3://{bucket}/{key_prefix}")
        s3.upload_file(str(local), bucket, key_prefix)
        print("Done.")
        return 0

    base_prefix = _ensure_trailing_slash(key_prefix) if key_prefix else ""
    files = list(_iter_files(local))
    if not files:
        raise RuntimeError(f"No files found under directory: {str(local)!r}")

    print(f"Uploading directory {local} -> s3://{bucket}/{base_prefix}")
    for f in files:
        rel = f.relative_to(local).as_posix()
        dst_key = f"{base_prefix}{rel}"
        print(f"- {rel} -> s3://{bucket}/{dst_key}")
        s3.upload_file(str(f), bucket, dst_key)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

