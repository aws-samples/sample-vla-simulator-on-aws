#!/usr/bin/env python3
"""
vla-simulator Stack Destroy
Usage: python destroy.py --vla gr00t        [--region REGION]
       python destroy.py --vla gr00t-gr1    [--region REGION]
       python destroy.py --vla pi           [--region REGION]
       python destroy.py --vla openvla-oft  [--region REGION]
       python destroy.py --vla lap          [--region REGION]
       python destroy.py --vla openarm-isaac [--region REGION]
       python destroy.py --vla openarm-lift-act [--region REGION]

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

from generate import (
    OFT_DEFAULT_SUITE,
    OFT_LIBERO_SUITES,
    normalise_libero_suite,
    oft_stack_name,
)

BASE_DIR = Path(__file__).parent

STACK_NAMES = {
    "gr00t": "GR00T-Demo",
    "gr00t-gr1": "GR00T-GR1-Demo",
    "gr00t-g1": "GR00T-G1-Demo",
    "pi": "Pi-Demo",
    "openvla-oft": "OpenVLA-OFT-Demo",   # default suite (10); non-default via oft_stack_name()
    "lap": "LAP-Demo",
    "rldx": "RLDX-Demo",
    "rldx-simpler": "RLDX-Simpler-Demo",
    "rldx-gr1": "RLDX-GR1-Demo",
    "rldx-kitchen": "RLDX-Kitchen-Demo",
    "openarm-isaac": "OpenArm-Isaac-Demo",
    "openarm-lift-act": "OpenArm-Lift-ACT-Demo",
    "molmoact2": "MolmoAct2-Demo",
}


def _validate_region(region: str) -> str:
    if not re.match(r"^[a-z]{2,3}(-[a-z]+)+-\d+$", region):
        print(f"[error] Invalid AWS region format: {region}", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(region)


def main():
    parser = argparse.ArgumentParser(description="vla-simulator Stack Destroy")
    parser.add_argument("--vla", required=True,
                        choices=["gr00t", "gr00t-gr1", "gr00t-g1", "pi", "openvla-oft", "lap", "rldx", "rldx-simpler", "rldx-gr1", "rldx-kitchen", "openarm-isaac", "openarm-lift-act", "molmoact2"],
                        help="VLA model stack to destroy")
    parser.add_argument("--region", "-r", metavar="REGION",
                        help="AWS region (overrides simulator-config.yaml)")
    parser.add_argument(
        "--libero-suite", default=OFT_DEFAULT_SUITE, choices=list(OFT_LIBERO_SUITES),
        help="openvla-oft only: LIBERO suite of the stack to destroy (default: 10)",
    )
    args = parser.parse_args()

    sim_path = BASE_DIR / "simulator-config.yaml"
    config = yaml.safe_load(sim_path.read_text())
    region = args.region or config["deployment"]["region"]
    safe_region = _validate_region(region)

    libero_suite = normalise_libero_suite(args.libero_suite)
    if args.vla == "openvla-oft":
        stack_name = oft_stack_name(libero_suite)
    else:
        stack_name = STACK_NAMES[args.vla]

    print(f"[destroy] VLA:    {args.vla}")
    if args.vla == "openvla-oft":
        print(f"[destroy] Suite:  libero_{libero_suite}")
    print(f"[destroy] Stack:  {stack_name}")
    print(f"[destroy] Region: {region}")
    print("[destroy] Deleting CDK stack... (S3 bucket will be retained)")
    print()

    oft_suite_ctx: list[str] = []
    if args.vla == "openvla-oft":
        oft_suite_ctx = ["-c", f"libero_suite={shlex.quote(libero_suite)}"]
    subprocess.run(  # nosec B603,B607 - hardcoded command list, no user input in executable path
        [
            "npx", "cdk", "destroy", stack_name,
            "-c", f"region={safe_region}",
            "-c", f"vla={args.vla}",
            *oft_suite_ctx,
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
