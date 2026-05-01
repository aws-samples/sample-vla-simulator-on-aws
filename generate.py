"""
generate.py — model yaml + simulator-config.yaml 읽어 assets/userdata/{vla}.sh 생성

동작:
  1. simulator-config.yaml (공통) + models/{vla}.yaml (모델별) 로드
  2. bridge SSM 값 해석 (ssm:/path → 실제 값, deploy.py에서 해석 후 전달됨)
  3. templates/{vla}-userdata.sh.j2 Jinja2 템플릿 렌더링
  4. assets/userdata/{vla}.sh 저장

사용법:
    python generate.py --vla gr00t     [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla gr00t-gr1 [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla pi        [--resolved-vpc vpc-xxx --resolved-nlb host:port]
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"


def _load_merged_config(simulator_config_path: Path, vla: str) -> dict:
    """simulator-config.yaml + models/{vla}.yaml 병합. 모델 설정 우선."""
    sim_cfg = yaml.safe_load(simulator_config_path.read_text())
    model_path = BASE_DIR / "models" / f"{vla}.yaml"
    if not model_path.exists():
        print(f"[error] Model config not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    model_cfg = yaml.safe_load(model_path.read_text())
    # deployment: simulator-config가 base, model yaml에 deployment 항목 있으면 덮어씀
    merged_deployment = {**sim_cfg.get("deployment", {}), **model_cfg.get("deployment", {})}
    return {**sim_cfg, **model_cfg, "deployment": merged_deployment}


def _build_gr00t_ctx(config: dict, resolved_grpc: str, model_id: str) -> dict:
    """gr00t / gr00t-gr1 공통 context 빌드."""
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{model_id}.yaml에 tasks 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    tasks_json = json.dumps(tasks, ensure_ascii=False)
    deployment = config.get("deployment", {})
    model = config.get("model", {})
    default_hf_repo = "nvidia/GR00T-N1.7-3B" if model_id == "gr00t" else "nvidia/GR00T-N1.6-3B"

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "isaac_groot_commit": model.get("isaac_groot_commit", ""),
        "uv_version": model.get("uv_version", ""),
        "hf_repo": model.get("hf_repo", default_hf_repo),
        "hf_subfolder": model.get("hf_subfolder", ""),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "remote_grpc_endpoint": resolved_grpc,
    }

    if resolved_grpc:
        bridge_dir = BASE_DIR / "assets" / "bridge" / "gr00t"
        missing = [f for f in ("zmq_grpc_bridge.py", "gr00t_pb2.py", "gr00t_pb2_grpc.py")
                   if not (bridge_dir / f).exists()]
        if missing:
            print(f"[error] Bridge 파일 없음: {missing} (assets/bridge/gr00t/ 확인)", file=sys.stderr)
            sys.exit(1)
        ctx["zmq_grpc_bridge_b64"] = base64.encodebytes(
            (bridge_dir / "zmq_grpc_bridge.py").read_bytes()
        ).decode()
        ctx["gr00t_pb2_b64"] = base64.encodebytes(
            (bridge_dir / "gr00t_pb2.py").read_bytes()
        ).decode()
        ctx["gr00t_pb2_grpc_b64"] = base64.encodebytes(
            (bridge_dir / "gr00t_pb2_grpc.py").read_bytes()
        ).decode()

    return ctx


def generate_gr00t(config: dict, resolved_grpc: str, resolved_vpc: str, dry_run: bool) -> str:
    ctx = _build_gr00t_ctx(config, resolved_grpc, "gr00t")
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("gr00t-userdata.sh.j2").render(**ctx)


def generate_gr00t_gr1(config: dict, resolved_grpc: str, resolved_vpc: str, dry_run: bool) -> str:
    ctx = _build_gr00t_ctx(config, resolved_grpc, "gr00t-gr1")
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("gr00t-gr1-userdata.sh.j2").render(**ctx)


def generate_pi(config: dict, resolved_vpc: str, resolved_nlb: str, dry_run: bool) -> str:
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/pi.yaml에 tasks 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    tasks_json = json.dumps(tasks, ensure_ascii=False)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    bridge_dir = BASE_DIR / "assets" / "bridge" / "pi"
    pi_proto_path = bridge_dir / "pi.proto"
    pi_grpc_bridge_path = bridge_dir / "pi_grpc_bridge.py"

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "openpi_commit": model.get("openpi_commit", "").strip(),
        "pi_proto": pi_proto_path.read_text() if pi_proto_path.exists() else "",
        "pi_grpc_bridge_py": pi_grpc_bridge_path.read_text() if pi_grpc_bridge_path.exists() else "",
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("pi-userdata.sh.j2").render(**ctx)


def main():
    parser = argparse.ArgumentParser(description="vla-simulator UserData 스크립트 생성")
    parser.add_argument("--vla", required=True, choices=["gr00t", "gr00t-gr1", "pi"], help="VLA 모델")
    parser.add_argument(
        "--config", default=str(BASE_DIR / "simulator-config.yaml"),
        help="공통 설정 파일 경로 (기본: simulator-config.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="파일 저장 없이 출력만")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "assets" / "userdata"),
                        help="결과 저장 디렉토리")
    # Bridge 해석 값 (deploy.py가 SSM 조회 후 전달)
    parser.add_argument("--resolved-grpc", default="", help="GR00T: resolved gRPC endpoint")
    parser.add_argument("--resolved-vpc", default="", help="resolved VPC ID")
    parser.add_argument("--resolved-nlb", default="", help="Pi: resolved NLB endpoint")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[error] 설정 파일 없음: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = _load_merged_config(config_path, args.vla)
    print(f"VLA: {args.vla}")
    print(f"Config: {config_path}")
    print(f"Model:  {BASE_DIR / 'models' / (args.vla + '.yaml')}")

    if args.vla == "gr00t":
        rendered = generate_gr00t(config, args.resolved_grpc, args.resolved_vpc, args.dry_run)
    elif args.vla == "gr00t-gr1":
        rendered = generate_gr00t_gr1(config, args.resolved_grpc, args.resolved_vpc, args.dry_run)
    else:
        rendered = generate_pi(config, args.resolved_vpc, args.resolved_nlb, args.dry_run)

    if args.dry_run:
        print("=" * 60)
        print(f"[DRY-RUN] {args.vla}.sh 내용 미리보기:")
        print("=" * 60)
        print(rendered)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.vla}.sh"
    dest.write_text(rendered)
    print(f"생성 완료: {dest}")


if __name__ == "__main__":
    main()
