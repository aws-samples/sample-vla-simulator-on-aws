#!/usr/bin/env python3
"""
vla-simulator Stack Destroy
Usage: python destroy.py --vla gr00t     [--region REGION]
       python destroy.py --vla gr00t-gr1 [--region REGION]
       python destroy.py --vla pi        [--region REGION]

Notes:
  - The S3 bucket is NOT deleted (RemovalPolicy.RETAIN).
  - To delete the S3 bucket manually:
      aws s3 rb s3://BUCKET_NAME --force
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent

STACK_NAMES = {"gr00t": "GR00T-Demo", "gr00t-gr1": "GR00T-GR1-Demo", "pi": "Pi-Demo"}


def _validate_region(region: str) -> str:
    if not re.match(r"^[a-z]{2,3}(-[a-z]+)+-\d+$", region):
        print(f"[error] Invalid AWS region format: {region}", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(region)


def main():
    parser = argparse.ArgumentParser(description="vla-simulator Stack Destroy")
    parser.add_argument("--vla", required=True, choices=["gr00t", "gr00t-gr1", "pi"],
                        help="VLA model stack to destroy")
    parser.add_argument("--region", "-r", metavar="REGION",
                        help="AWS region (overrides simulator-config.yaml)")
    args = parser.parse_args()

    sim_path = BASE_DIR / "simulator-config.yaml"
    config = yaml.safe_load(sim_path.read_text())
    region = args.region or config["deployment"]["region"]
    safe_region = _validate_region(region)

    stack_name = STACK_NAMES[args.vla]

    print(f"[destroy] VLA:    {args.vla}")
    print(f"[destroy] Stack:  {stack_name}")
    print(f"[destroy] Region: {region}")
    print("[destroy] Deleting CDK stack... (S3 bucket will be retained)")
    print()

    subprocess.run(  # nosec B603,B607 - hardcoded command list, no user input in executable path
        [
            "npx", "cdk", "destroy", stack_name,
            "-c", f"region={safe_region}",
            "-c", f"vla={args.vla}",
            "--force",
        ],
        cwd=str(BASE_DIR / "cdk"),
        check=True,
    )

    print()
    print("[destroy] Stack deleted.")
    print("  The S3 bucket still exists. To remove it manually:")
    print("  aws s3 rb s3://BUCKET_NAME --force")


if __name__ == "__main__":
    main()
