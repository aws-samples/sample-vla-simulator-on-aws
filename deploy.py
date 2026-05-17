#!/usr/bin/env python3
"""
vla-simulator 1-Click Deploy
Usage:
  python deploy.py --vla gr00t              # GR00T N1.7 + LIBERO local mode
  python deploy.py --vla gr00t-gr1          # GR00T N1.6 + GR1 humanoid + RoboCasa local mode
  python deploy.py --vla pi                 # π0.5  local mode
  python deploy.py --vla openvla-oft        # OpenVLA-OFT + LIBERO-10 local mode
  python deploy.py --vla lap                # LAP-3B + LIBERO-Spatial local mode
  python deploy.py --vla gr00t --bridge     # GR00T bridge mode (vla-hub ECS)
  python deploy.py --vla gr00t-gr1 --bridge # GR00T-GR1 bridge mode (vla-hub ECS, if N1.6 supported)
  python deploy.py --vla pi    --bridge     # π0.5  bridge mode (vla-hub ECS)

Steps:
  1. Load simulator-config.yaml + models/{vla}.yaml
  2. If bridge mode: resolve SSM-backed endpoints
  3. python generate.py → assets/userdata/{vla}.sh
  4. cdk deploy {VLA}-Demo

Notes:
  - On first deploy, a SNS subscription confirmation email will arrive.
    Click "Confirm subscription" to receive completion notifications.
  - After simulation completes, run: python destroy.py --vla {gr00t|pi}
"""

import argparse
import copy
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

BASE_DIR = Path(__file__).parent


def _resolve_ssm(value: str, region: str) -> str:
    """value가 'ssm:/path' 형식이면 SSM Parameter Store에서 조회."""
    if not value.startswith("ssm:"):
        return value
    param_name = value[4:]
    ssm_client = boto3.client("ssm", region_name=region)
    try:
        resp = ssm_client.get_parameter(Name=param_name)
        resolved = resp["Parameter"]["Value"]
        print(f"[ssm] {param_name} → {resolved}")
        return resolved
    except ssm_client.exceptions.ParameterNotFound:
        print(f"[error] SSM parameter not found: {param_name}", file=sys.stderr)
        print("  Deploy VlaHubStack first: cd projects/vla-hub && npx cdk deploy VlaHubStack",
              file=sys.stderr)
        sys.exit(1)


def _validate_email(email: str) -> str:
    if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email):
        print(f"[error] Invalid email format: {email}", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(email)


def _validate_region(region: str) -> str:
    if not re.match(r"^[a-z]{2,3}(-[a-z]+)+-\d+$", region):
        print(f"[error] Invalid AWS region format: {region}", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(region)


def _validate_vpc_id(vpc_id: str) -> str:
    if not re.match(r"^vpc-[0-9a-f]{8,17}$", vpc_id):
        print(f"[error] Invalid VPC ID format: {vpc_id} (expected: vpc-XXXXXXXX)", file=sys.stderr)
        sys.exit(1)
    return shlex.quote(vpc_id)


def _load_config(vla: str) -> dict:
    """simulator-config.yaml + models/{vla}.yaml 병합. 모델 설정 우선."""
    sim_path = BASE_DIR / "simulator-config.yaml"
    model_path = BASE_DIR / "models" / f"{vla}.yaml"
    if not sim_path.exists():
        print(f"[error] Config not found: {sim_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"[error] Model config not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    sim_cfg = yaml.safe_load(sim_path.read_text())
    model_cfg = yaml.safe_load(model_path.read_text())
    merged_deployment = {**sim_cfg.get("deployment", {}), **model_cfg.get("deployment", {})}
    return {**sim_cfg, **model_cfg, "deployment": merged_deployment}


def _maybe_import_orphan_bucket(
    cdk_dir: str,
    stack_name: str,
    region: str,
    safe_email: str,
    safe_vpc_id: str,
    s3_results_prefix: str,
    extra_ctx: list[str],
    cdk_out_dir: str = "cdk.out",
    vla: str = "",
) -> None:
    """S3 bucket이 orphan (RETAIN 후 stack 삭제됨)이면 cdk import로 재채택."""
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]
    bucket_name = f"{s3_results_prefix}-{stack_name.lower()}-{region}-{account_id}"

    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            return
        # 403 = bucket exists but restricted; fall through

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        cfn.describe_stacks(StackName=stack_name)
        return  # Stack still exists
    except ClientError:
        pass

    print(f"[pre-check] Orphan S3 bucket detected: {bucket_name}")
    print("[pre-check] Re-adopting via `cdk import` before deploy ...")

    mapping = {"ResultsBucketA95A2103": {"BucketName": bucket_name}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump(mapping, fh)
        mapping_file = fh.name

    vla_ctx = ["-c", f"vla={vla}"] if vla else []
    cdk_import_cmd = [
        "npx", "cdk", "import", stack_name,
        "-c", f"region={shlex.quote(region)}",
        "-c", f"notify_email={safe_email}",
        *vla_ctx,
        "--resource-mapping", mapping_file,
        "--force",
        "--output", cdk_out_dir,
        *extra_ctx,
    ]
    subprocess.run(cdk_import_cmd, cwd=cdk_dir, check=True)  # nosec B603,B607 - hardcoded command list
    print()


def main():
    parser = argparse.ArgumentParser(description="vla-simulator 1-Click Deploy")
    parser.add_argument("--vla", required=True, choices=["gr00t", "gr00t-gr1", "pi", "openvla-oft", "lap"],
                        help="VLA model to deploy")
    parser.add_argument("--bridge", action="store_true",
                        help="Bridge mode: use vla-hub ECS endpoint instead of local model")
    parser.add_argument("--email", "-e", metavar="ADDRESS",
                        help="Notification email (overrides simulator-config.yaml)")
    parser.add_argument("--region", "-r", metavar="REGION",
                        help="AWS region (overrides simulator-config.yaml)")
    args = parser.parse_args()

    config = _load_config(args.vla)
    deployment = config.get("deployment", {})
    model = config.get("model", {})
    bridge_cfg = config.get("bridge", {})
    instance_cfg = config.get("instance", {})

    region = args.region or deployment["region"]
    email = args.email or deployment["notify_email"]

    if email == "YOUR_EMAIL@example.com":
        print("[error] Email not set. Use --email YOUR@EMAIL.COM or edit notify_email in "
              "simulator-config.yaml", file=sys.stderr)
        sys.exit(1)

    safe_email = _validate_email(email)
    safe_region = _validate_region(region)

    # ── Bridge mode resolution ────────────────────────────────────
    vpc_id = ""
    extra_cdk_ctx: list[str] = []
    generate_extra: list[str] = []

    if args.bridge:
        if args.vla == "openvla-oft":
            print("[error] Bridge mode not supported for openvla-oft (local only).", file=sys.stderr)
            sys.exit(1)
        if args.vla == "lap":
            print("[error] Bridge mode not yet supported for lap (local only — vla-hub LAP server is a follow-up).",
                  file=sys.stderr)
            sys.exit(1)
        if args.vla in ("gr00t", "gr00t-gr1"):
            raw_grpc = str(bridge_cfg.get("remote_grpc_endpoint", "")).strip()
            raw_vpc = str(bridge_cfg.get("vpc_id", "")).strip()
            if not raw_grpc or not raw_vpc:
                print(f"[error] Bridge mode requires bridge.remote_grpc_endpoint and bridge.vpc_id "
                      f"in models/{args.vla}.yaml", file=sys.stderr)
                sys.exit(1)
            resolved_grpc = _resolve_ssm(raw_grpc, region)
            resolved_vpc = _resolve_ssm(raw_vpc, region)
            vpc_id = resolved_vpc
            safe_vpc = _validate_vpc_id(vpc_id)
            extra_cdk_ctx = ["-c", f"vpc_id={safe_vpc}"]
            generate_extra = ["--resolved-grpc", resolved_grpc, "--resolved-vpc", resolved_vpc]
        else:  # pi
            raw_vpc = str(bridge_cfg.get("vpc_id", "")).strip()
            raw_nlb = str(bridge_cfg.get("nlb_endpoint", "")).strip()
            if not raw_vpc or not raw_nlb:
                print("[error] Bridge mode requires bridge.vpc_id and bridge.nlb_endpoint "
                      "in models/pi.yaml", file=sys.stderr)
                sys.exit(1)
            resolved_vpc = _resolve_ssm(raw_vpc, region)
            resolved_nlb = _resolve_ssm(raw_nlb, region) if raw_nlb.startswith("ssm:") else raw_nlb
            vpc_id = resolved_vpc
            safe_vpc = _validate_vpc_id(vpc_id)
            extra_cdk_ctx = ["-c", f"vpc_id={safe_vpc}", "-c", f"nlb_endpoint={shlex.quote(resolved_nlb)}"]
            generate_extra = ["--resolved-vpc", resolved_vpc, "--resolved-nlb", resolved_nlb]

    if args.vla == "gr00t":
        stack_name = "GR00T-Demo"
    elif args.vla == "gr00t-gr1":
        stack_name = "GR00T-GR1-Demo"
    elif args.vla == "openvla-oft":
        stack_name = "OpenVLA-OFT-Demo"
    elif args.vla == "lap":
        stack_name = "LAP-Demo"
    else:
        stack_name = "Pi-Demo"
    mode = "bridge" if args.bridge else "local"
    s3_results_prefix = deployment.get("s3_results_prefix", "vla-sim-results")
    cdk_dir = str(BASE_DIR / "cdk")

    print(f"[deploy] VLA:    {args.vla}")
    print(f"[deploy] Region: {region}")
    print(f"[deploy] Email:  {email}")
    print(f"[deploy] Mode:   {mode}")
    print(f"[deploy] Stack:  {stack_name}")
    if vpc_id:
        print(f"[deploy] VPC:    {vpc_id}")
    print()

    # 1. Generate UserData
    print("[1/2] Generating UserData script...")
    subprocess.run(  # nosec B603 - sys.executable is the current Python interpreter, not user input
        [
            sys.executable, "generate.py",
            "--vla", args.vla,
            "--output-dir", "assets/userdata",
            *generate_extra,
        ],
        cwd=str(BASE_DIR),
        check=True,
    )
    print()

    # Use a VLA-specific output dir so parallel deploys don't clash on cdk.out
    cdk_out_dir = f"cdk.out-{args.vla}"

    # 1.5 Pre-deploy: re-adopt orphan S3 bucket
    _maybe_import_orphan_bucket(
        cdk_dir, stack_name, region, safe_email,
        shlex.quote(vpc_id) if vpc_id else "",
        s3_results_prefix, extra_cdk_ctx,
        cdk_out_dir=cdk_out_dir,
        vla=args.vla,
    )

    # 2. CDK deploy
    print("[2/2] Starting CDK deploy...")
    if mode == "bridge":
        if args.vla in ("gr00t", "gr00t-gr1"):
            print("  Estimated time: ~60 min (install ~20-30min + bridge eval ~30min)")
        else:
            print("  Estimated time: ~30-40 min (install ~10min + bridge eval ~20-30min)")
    else:
        if args.vla == "gr00t":
            print("  Estimated time: ~120 min (install ~30min + model download ~10min + sim ~80min)")
        elif args.vla == "gr00t-gr1":
            print("  Estimated time: ~90 min (install ~20min + model download ~5min + sim ~60min)")
        elif args.vla == "openvla-oft":
            print("  Estimated time: ~180 min (conda/pip ~30min + HF download ~10min + LIBERO-10 eval ~120min)")
        elif args.vla == "lap":
            print("  Estimated time: ~90-150 min (uv venvs ~25min + HF download ~10min + LIBERO-Spatial eval ~30-50min + buffer)")
        else:
            print("  Estimated time: ~90-120 min per suite (install ~30min + eval ~60-90min)")
    print("  If this is your first deploy, confirm the SNS subscription email to receive notifications.")
    print()

    cdk_cmd = [
        "npx", "cdk", "deploy", stack_name,
        "-c", f"region={safe_region}",
        "-c", f"notify_email={safe_email}",
        "-c", f"vla={args.vla}",
        "--require-approval", "never",
        "--output", cdk_out_dir,
        *extra_cdk_ctx,
    ]
    subprocess.run(cdk_cmd, cwd=cdk_dir, check=True)  # nosec B603,B607 - hardcoded command list

    print()
    print("[deploy] Done! Simulation is running in the background.")
    print(f"  A completion notification will be sent to: {email}")
    vla_safe = shlex.quote(args.vla)
    print(f"  To clean up: python destroy.py --vla {vla_safe}")


if __name__ == "__main__":
    main()
