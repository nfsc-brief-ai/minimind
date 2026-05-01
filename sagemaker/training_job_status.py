#!/usr/bin/env python3
"""
Fetch SageMaker training job status via DescribeTrainingJob / ListTrainingJobs.

Requires boto3 (installed with the SageMaker SDK: pip install -r sagemaker/requirements-submit.txt).

Examples:
  python sagemaker/training_job_status.py --job-name pytorch-training-2026-05-01-12-34-56-789

  python sagemaker/training_job_status.py --list --max-results 10

  python sagemaker/training_job_status.py --list --name-contains minimind --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any


def _session(region: str | None, profile: str | None):
    import boto3.session

    return boto3.session.Session(profile_name=profile, region_name=region)


def _fmt_dt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()
    return str(v)


def _describe_job(client, job_name: str) -> dict[str, Any]:
    return client.describe_training_job(TrainingJobName=job_name)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch SageMaker training job status.")
    p.add_argument("--job-name", type=str, default=None, help="Training job name (from console or submit output).")
    p.add_argument(
        "--list",
        action="store_true",
        help="List recent training jobs instead of describing one.",
    )
    p.add_argument("--name-contains", type=str, default=None, help="With --list: filter names containing this substring.")
    p.add_argument("--max-results", type=int, default=20, help="With --list: max jobs to return (default 20).")
    p.add_argument("--region", type=str, default=None, help="AWS region (default: from profile/env).")
    p.add_argument("--profile", type=str, default=None, help="AWS named profile.")
    p.add_argument("--json", action="store_true", help="Print raw API JSON.")
    args = p.parse_args()

    if not args.list and not args.job_name:
        p.error("Provide --job-name or use --list")

    try:
        import boto3
    except ImportError as e:
        print("boto3 is required. Install with: pip install boto3", file=sys.stderr)
        raise SystemExit(1) from e

    sess = _session(args.region, args.profile)
    sts = sess.client("sts")
    ident = sts.get_caller_identity()
    region = sess.region_name or "(default from profile/env)"
    print(f"Using AWS account {ident.get('Account')} in region {region}", flush=True)

    client = sess.client("sagemaker")

    if args.list:
        kwargs: dict[str, Any] = {"MaxResults": min(max(args.max_results, 1), 100)}
        if args.name_contains:
            kwargs["NameContains"] = args.name_contains
        resp = client.list_training_jobs(**kwargs)
        jobs = resp.get("TrainingJobSummaries", [])
        if args.json:
            print(json.dumps(jobs, default=str, indent=2))
            return 0
        if not jobs:
            print(
                "No training jobs found. If you expected jobs, check you are using the same "
                "region and account as submit_pretrain.py (use --region / --profile).",
                flush=True,
            )
            return 0
        for j in jobs:
            name = j.get("TrainingJobName", "")
            status = j.get("TrainingJobStatus", "")
            secondary = j.get("SecondaryStatus", "")
            created = j.get("CreationTime")
            line = f"{status:12} {secondary:24} {_fmt_dt(created)}  {name}"
            print(line)
        return 0

    job = _describe_job(client, args.job_name)
    if args.json:
        print(json.dumps(job, default=str, indent=2))
        return 0

    print(f"TrainingJobName: {job.get('TrainingJobName')}")
    print(f"TrainingJobStatus: {job.get('TrainingJobStatus')}")
    if job.get("SecondaryStatus"):
        print(f"SecondaryStatus: {job.get('SecondaryStatus')}")
    print(f"CreationTime: {_fmt_dt(job.get('CreationTime'))}")
    print(f"TrainingStartTime: {_fmt_dt(job.get('TrainingStartTime'))}")
    print(f"TrainingEndTime: {_fmt_dt(job.get('TrainingEndTime'))}")
    if job.get("BillableTimeInSeconds") is not None:
        print(f"BillableTimeInSeconds: {job.get('BillableTimeInSeconds')}")
    if job.get("FailureReason"):
        print(f"FailureReason: {job.get('FailureReason')}")
    role = job.get("RoleArn")
    if role:
        print(f"RoleArn: {role}")
    rm = job.get("ResourceConfig") or {}
    if rm:
        print(
            f"InstanceType: {rm.get('InstanceType')}  "
            f"InstanceCount: {rm.get('InstanceCount')}  "
            f"VolumeSizeInGB: {rm.get('VolumeSizeInGB')}"
        )
    om = job.get("OutputDataConfig") or {}
    if om.get("S3OutputPath"):
        print(f"ModelArtifacts S3OutputPath: {om['S3OutputPath']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
