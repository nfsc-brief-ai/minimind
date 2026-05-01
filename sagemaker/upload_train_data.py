#!/usr/bin/env python3
"""Upload JSONL pretrain data to S3. boto3 is installed with sagemaker (requirements.txt)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _parse_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("URI must start with s3://")
    rest = uri[5:]
    if "/" not in rest:
        return rest, ""
    return rest.split("/", 1)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == ".DS_Store" or "__pycache__" in p.parts or ".git" in p.parts:
            continue
        yield p


def main() -> int:
    p = argparse.ArgumentParser(description="Upload pretrain JSONL (or a folder) to S3.")
    p.add_argument("--local_path", required=True)
    p.add_argument(
        "--s3_uri",
        required=True,
        help="Bucket prefix s3://bucket/prefix/ or full object s3://bucket/prefix/pretrain.jsonl",
    )
    p.add_argument("--region", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    local = Path(args.local_path).expanduser().resolve()
    if not local.exists():
        raise SystemExit(f"Not found: {local}")

    bucket, key_prefix = _parse_s3(args.s3_uri.strip())
    key_prefix = key_prefix.strip("/")

    if args.dry_run:
        import boto3  # noqa: F401 — dry-run may skip

    if local.is_file():
        if key_prefix.endswith(".jsonl") or "/" in key_prefix and not args.s3_uri.endswith("/"):
            object_key = key_prefix
        else:
            object_key = f"{key_prefix}/{local.name}" if key_prefix else local.name
        if args.dry_run:
            print(f"would put s3://{bucket}/{object_key} <- {local}")
            return 0
        import boto3

        b = boto3.Session(region_name=args.region, profile_name=args.profile).client("s3")
        print(f"s3://{bucket}/{object_key} <- {local}")
        b.upload_file(str(local), bucket, object_key)
        return 0

    # directory → prefix must be a “folder”
    prefix = key_prefix + "/" if key_prefix else ""
    if args.dry_run:
        for f in _iter_files(local):
            rel = f.relative_to(local).as_posix()
            print(f"would put s3://{bucket}/{prefix}{rel} <- {f}")
        return 0

    import boto3

    b = boto3.Session(region_name=args.region, profile_name=args.profile).client("s3")
    for f in _iter_files(local):
        rel = f.relative_to(local).as_posix()
        k = f"{prefix}{rel}"
        print(f"s3://{bucket}/{k} <- {f}")
        b.upload_file(str(f), bucket, k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
