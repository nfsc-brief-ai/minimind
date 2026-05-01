#!/usr/bin/env python3
"""Describe or list SageMaker training jobs. Uses same credentials as AWS CLI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()
    return str(v) if v is not None else ""


def main() -> int:
    p = argparse.ArgumentParser(description="SageMaker training job status")
    p.add_argument("--job-name", default=None, help="Describe this job")
    p.add_argument("--list", action="store_true", help="List recent jobs")
    p.add_argument("--max-results", type=int, default=20)
    p.add_argument("--name-contains", default=None, help="Filter list by name substring")
    p.add_argument("--region", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not args.list and not args.job_name:
        p.error("Pass --job-name or --list")

    try:
        import boto3
    except ImportError:
        print("Install boto3 (pip install -r sagemaker/requirements.txt)", file=sys.stderr)
        return 1

    sess = boto3.Session(region_name=args.region, profile_name=args.profile)
    sts = sess.client("sts")
    ident = sts.get_caller_identity()
    region = sess.region_name or "(default)"
    print(f"account={ident.get('Account')} region={region}", flush=True)

    sm = sess.client("sagemaker")

    if args.list:
        kw: dict = {"MaxResults": min(max(args.max_results, 1), 100)}
        if args.name_contains:
            kw["NameContains"] = args.name_contains
        resp = sm.list_training_jobs(**kw)
        jobs = resp.get("TrainingJobSummaries", [])
        if args.json:
            print(json.dumps(jobs, default=str, indent=2))
            return 0
        if not jobs:
            print("No jobs (wrong region/account?)")
            return 0
        for j in jobs:
            print(
                f"{j.get('TrainingJobStatus',''):12} {_fmt(j.get('CreationTime'))}  {j.get('TrainingJobName','')}"
            )
        return 0

    job = sm.describe_training_job(TrainingJobName=args.job_name)
    if args.json:
        print(json.dumps(job, default=str, indent=2))
        return 0

    print(f"name: {job.get('TrainingJobName')}")
    print(f"status: {job.get('TrainingJobStatus')}")
    print(f"secondary: {job.get('SecondaryStatus')}")
    print(f"created: {_fmt(job.get('CreationTime'))}")
    print(f"started: {_fmt(job.get('TrainingStartTime'))}")
    print(f"ended: {_fmt(job.get('TrainingEndTime'))}")
    if job.get("FailureReason"):
        print(f"failure: {job['FailureReason']}")
    rc = job.get("ResourceConfig") or {}
    if rc.get("InstanceType"):
        print(f"instance: {rc.get('InstanceType')} x {rc.get('InstanceCount', 1)}")
    out = job.get("OutputDataConfig") or {}
    if out.get("S3OutputPath"):
        print(f"output: {out['S3OutputPath']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
