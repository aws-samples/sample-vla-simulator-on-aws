"""
generate.py — model yaml + simulator-config.yaml 읽어 assets/userdata/{vla}.sh 생성

동작:
  1. simulator-config.yaml (공통) + models/{vla}.yaml (모델별) 로드
  2. bridge SSM 값 해석 (ssm:/path → 실제 값, deploy.py에서 해석 후 전달됨)
  3. templates/{vla}-userdata.sh.j2 Jinja2 템플릿 렌더링
  4. assets/userdata/{vla}.sh 저장

사용법:
    python generate.py --vla gr00t       [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla gr00t-gr1   [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla pi          [--resolved-vpc vpc-xxx --resolved-nlb host:port]
    python generate.py --vla openvla-oft [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla lap         [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla rldx        [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla rldx-simpler [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla openarm-isaac [--config simulator-config.yaml] [--dry-run]
    python generate.py --vla openarm-lift-act [--config simulator-config.yaml] [--dry-run]
"""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

# OpenVLA-OFT supported LIBERO suites (`long` is an alias for `10`).
OFT_LIBERO_SUITES = ("spatial", "object", "goal", "10", "long")
OFT_DEFAULT_SUITE = "10"


def _tasks_json_for_bash(tasks: list) -> str:
    """Serialize tasks to JSON, then bash-escape for embedding in a single-quoted
    string (templates use `TASKS_JSON='{{ tasks_json }}'`).

    Inside bash single quotes the ONLY metacharacter is `'` itself, so replacing
    each `'` with the canonical `'\\''` (close-quote, literal quote, reopen-quote)
    makes ANY free-text task description safe. This is a no-op (byte-identical
    output) for descriptions without apostrophes. Without it, an apostrophe in a
    task description (e.g. "gr00t's") prematurely terminates the bash string →
    UserData syntax error → script aborts before the rollout loop → no cfn_signal
    → stack hangs on WaitCondition until timeout.
    """
    return json.dumps(tasks, ensure_ascii=False).replace("'", "'\\''")


def normalise_libero_suite(suite: str) -> str:
    """`long` → `10` (OFT paper: LIBERO-Long = LIBERO-10)."""
    return "10" if suite == "long" else suite


def oft_stack_name(suite: str) -> str:
    """Stack id for a given suite. Default (`10`) preserves legacy name for
    backwards compatibility with existing deployments."""
    s = normalise_libero_suite(suite)
    if s == OFT_DEFAULT_SUITE:
        return "OpenVLA-OFT-Demo"
    return f"OpenVLA-OFT-{s.capitalize()}-Demo"


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

    tasks_json = _tasks_json_for_bash(tasks)
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
        "robosuite_commit": model.get("robosuite_commit", ""),  # gr00t-g1 only (WBC robosuite fork pin)
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


def generate_gr00t_g1(config: dict, resolved_grpc: str, resolved_vpc: str, dry_run: bool,
                      collect: bool = False) -> str:
    """GR00T N1.6 G1 whole-body loco-manip. Two modes (same WBC/ZMQ/checkpoint stack):

      • rollout (default): run rollout_policy.py → MP4 + success rate to S3 (the verified 3/10 demo).
      • collect (--collect): run collect_trajectories.py → success-only (obs,action) HDF5 to S3, for
        imitation fine-tuning of an N1.7 G1 adapter. Forces n_action_steps=1 (per-timestep transitions)
        and embeds the 2 collect assets as base64. Everything else (setup, ZMQ server, ckpt) is shared.
    """
    ctx = _build_gr00t_ctx(config, resolved_grpc, "gr00t-g1")
    ctx["collect"] = collect

    # N1.7 rollout server (Step 2): if models/gr00t-g1.yaml has a `server:` block with a checkpoint
    # source (either `hf_repo` for the public, reproducible path OR `s3_ckpt_uri` for a private bucket),
    # the inference server loads the fine-tuned N1.7 G1 adapter ckpt in a SEPARATE venv (Isaac-GR00T
    # 65cc4a), while the WBC sim keeps the N1.6 stack (77866395). Mutually exclusive with --collect
    # (collect runs the N1.6 teacher) and bridge mode (remote inference). Absent → the N1.6 demo path.
    server = config.get("server", {}) or {}
    server_hf_repo = server.get("hf_repo", "").strip()
    server_s3 = server.get("s3_ckpt_uri", "").strip()
    server_n17 = bool(server_hf_repo or server_s3)
    if server_hf_repo and server_s3:
        print("[error] server.hf_repo and server.s3_ckpt_uri are mutually exclusive — pick one "
              "checkpoint source for the N1.7 rollout server.", file=sys.stderr)
        sys.exit(1)
    if server_n17 and collect:
        print("[error] server.* (N1.7 rollout) and --collect are mutually exclusive: "
              "collect runs the N1.6 cloudwalk teacher, not the N1.7 adapter.", file=sys.stderr)
        sys.exit(1)
    if server_n17 and resolved_grpc:
        print("[error] server.* (N1.7 local server) and bridge mode (remote_grpc_endpoint) "
              "are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    ctx["server_n17"] = server_n17
    ctx["server_hf_repo"] = server_hf_repo
    ctx["server_hf_revision"] = server.get("hf_revision", "").strip()
    ctx["server_s3_ckpt_uri"] = server_s3
    ctx["server_isaac_groot_commit"] = server.get("isaac_groot_commit", "")
    ctx["server_hf_token_ssm"] = server.get("hf_token_ssm", "/vla-simulator/hf-token")
    ctx["server_embodiment_tag"] = server.get("embodiment_tag", "UNITREE_G1")

    if collect:
        if resolved_grpc:
            print("[error] --collect is local mode only (bridge delegates inference, no obs/action "
                  "dict locally to dump).", file=sys.stderr)
            sys.exit(1)
        asset_dir = BASE_DIR / "assets" / "gr00t-g1"
        asset_files = {
            "collect_traj_b64": "collect_trajectories.py",
            "hdf5_to_lerobot_b64": "gr00t_hdf5_to_lerobot.py",
        }
        missing = [fn for fn in asset_files.values() if not (asset_dir / fn).exists()]
        if missing:
            print(f"[error] gr00t-g1 collect asset 파일 없음: {missing} (assets/gr00t-g1/ 확인)",
                  file=sys.stderr)
            sys.exit(1)
        for ctx_key, fname in asset_files.items():
            ctx[ctx_key] = base64.encodebytes((asset_dir / fname).read_bytes()).decode()
        collection = config.get("collection", {})
        ctx["collection"] = {
            "lerobot_task": collection.get("lerobot_task", "pick up the apple and place it on the plate"),
            "repo_id": collection.get("repo_id", "local/gr00t-g1-applepnp"),
            "max_total_episodes": collection.get("max_total_episodes", 0),
            # num_demos = SUCCESSFUL demos to export (collect target). Decoupled from tasks[].n_episodes
            # (which is the rollout/eval episode count) so a small validation collect run doesn't alter eval.
            "num_demos": collection.get("num_demos", 0),  # 0 → fall back to tasks[].n_episodes
        }
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template, autoescape not applicable
    return env.get_template("gr00t-g1-userdata.sh.j2").render(**ctx)


def generate_openvla_oft(config: dict, libero_suite: str, dry_run: bool) -> str:
    suite = normalise_libero_suite(libero_suite)
    model = config.get("model", {})
    suites = model.get("suites", {})
    suite_cfg = suites.get(suite)
    if not suite_cfg:
        print(f"[error] models/openvla-oft.yaml의 model.suites에 '{suite}' 항목이 없습니다. "
              f"사용 가능: {sorted(suites.keys())}", file=sys.stderr)
        sys.exit(1)

    tasks_by_suite = config.get("tasks", {})
    if isinstance(tasks_by_suite, list):
        # Backwards-compat: flat list treated as the default suite.
        tasks = tasks_by_suite
    else:
        tasks = tasks_by_suite.get(suite, [])
    if not tasks:
        print(f"[error] models/openvla-oft.yaml에 suite '{suite}'의 tasks가 없습니다.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "libero_suite": suite,
        "hf_repo": suite_cfg["hf_repo"],
        "hf_model_revision": suite_cfg.get("hf_model_revision", ""),
        "task_suite_name": suite_cfg["task_suite_name"],
        "oft_commit": model.get("oft_commit", ""),
        "transformers_fork_commit": model.get("transformers_fork_commit", ""),
        "libero_commit": model.get("libero_commit", "master"),
        "center_crop": model.get("center_crop", True),
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("openvla-oft-userdata.sh.j2").render(**ctx)


def generate_rldx(config: dict, dry_run: bool, vla: str = "rldx") -> str:
    """RLDX-1 (RLWRLD): MSAT/Qwen3-VL-8B × {LIBERO | SimplerEnv | ...}. Local mode only.

    ONE shared template (rldx-userdata.sh.j2) parameterised by `sim_id` (openvla-oft
    pattern), NOT a per-sim template fork. The server+client two-venv plumbing
    (cfn_signal / die / S3 upload / SNS / ZeroMQ warmup) is sim-agnostic single-source;
    only the [5/8] sim-venv setup+fix block, the server args, the rollout values, and
    the result labels diverge by sim. LIBERO (sim_id default) renders byte-identically
    to the pre-parameterisation template — guarded by the dry-run regression in .temp/.

    `vla` (= the model id / CDK context value) drives the CloudWatch log_prefix so the
    `/rldx/userdata` log group matches the stack's `/${vla}/*` IAM grant. A mismatch
    deploys fine but silently drops logs (no `aws logs tail`); see session risk #9.

    No bridge assets (vla-hub serving deferred — fused-kernel guardrail conflict).
    """
    tasks = config.get("tasks", [])
    if not tasks:
        print(f"[error] models/{vla}.yaml에 tasks 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    ctx = {
        "tasks_json": tasks_json,
        "tasks": tasks,
        "deployment": deployment,
        "rldx_commit": model.get("rldx_commit", "").strip(),
        "hf_repo": model.get("hf_repo", "RLWRLD/RLDX-1-FT-LIBERO"),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "backbone_hf_repo": model.get("backbone_hf_repo", ""),
        "embodiment_tag": model.get("embodiment_tag", "GENERAL_EMBODIMENT"),
        "compile": str(model.get("compile", "") or "").strip(),
        "max_episode_steps": model.get("max_episode_steps", 720),
        "n_action_steps": model.get("n_action_steps", 8),
        "n_envs": model.get("n_envs", 1),
        # ── sim parameterisation (empty/default → LIBERO, byte-identical) ──
        # log_prefix = vla so the CW log group matches the stack's /${vla}/* IAM grant.
        "log_prefix": vla,
        "sim_id": model.get("sim_id", "libero"),
        "sim_venv_subpath": model.get("sim_venv_subpath", ""),
        "sim_setup_script": model.get("sim_setup_script", ""),
        "sim_register_module": model.get("sim_register_module", ""),
        "sim_register_fn": model.get("sim_register_fn", ""),
        # FIX 5 — non-LIBERO sims whose external_dependencies/<sim> gitlink is absent at the pin
        # (SimplerEnv) must be pre-cloned --recursive so the setup script's already-populated
        # fallback runs. Empty for LIBERO (vendored dir present) → template skips the pre-clone.
        "sim_clone_url": model.get("sim_clone_url", ""),
        "sim_clone_subpath": model.get("sim_clone_subpath", ""),
        # FIX 1 (broadened) — pins the full rldx-dep install must not bump (sim renderer pins).
        "sim_protect_pins": model.get("sim_protect_pins", ""),
        # FIX 8 — SAPIEN/ManiSkill sims (SimplerEnv) render via Vulkan, which the DLAMI does
        # NOT ship (no libvulkan.so.1 loader, no ICD JSON) → `import sapien.core` dies with
        # `ImportError: libvulkan.so.1`. Install the Vulkan loader + point the NVIDIA ICD at
        # the pre-installed driver (libGLX_nvidia.so.0). Empty for LIBERO/MuJoCo sims (no
        # Vulkan) → template skips the block, LIBERO render stays byte-identical.
        # Source of truth: simpler-env/ManiSkill2_real2sim/docker/{Dockerfile,nvidia_icd.json}.
        "needs_vulkan": model.get("needs_vulkan", ""),
        # server CLI args appended after --use-sim-policy-wrapper. LIBERO ships --no-strict;
        # SIMPLER (eval_simpler.sh) runs strict (no flag). NOTE: eval_simpler.sh ALSO passes
        # --num-inference-timesteps 10, but that is NOT a ServerConfig field at pin ecbfaf80
        # (tyro rejects unknown args → server crash); inference steps come from the ckpt
        # config. Omitted by default; re-add via server_extra_args after a Gate-1 check.
        "server_extra_args": model.get("server_extra_args", ["--no-strict"]),
        "robot_label": model.get("robot_label", ""),
        "sim_label": model.get("sim_label", ""),
        # Server warmup watchdog (seconds). Default 900 (15min) = LIBERO byte-identical.
        # SIMPLER (Gate-3 run 2): MSAT/Qwen3-VL-8B shard load is interleaved with an fp32
        # up-cast of all trainable params → ~250-316s/shard × 3 on g6.2xlarge (8 vCPU/32GB),
        # ~17-18min cold. Raised for SIMPLER so the watchdog doesn't kill a healthy load
        # mid-shard. Stays well under creationpolicy_timeout (PT180M).
        "server_warmup_timeout": int(model.get("server_warmup_timeout", 900)),
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("rldx-userdata.sh.j2").render(**ctx)


def generate_lap(config: dict, resolved_nlb: str, dry_run: bool) -> str:
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/lap.yaml에 tasks 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    # Bridge assets (embedded into UserData as heredoc — same pattern as generate_pi).
    # Empty strings render harmless heredocs in local mode (BRIDGE_MODE=false skips them).
    bridge_dir = BASE_DIR / "assets" / "bridge" / "lap"
    lap_proto_path = bridge_dir / "lap.proto"
    lap_grpc_bridge_path = bridge_dir / "lap_grpc_bridge.py"

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "lap_commit": model.get("lap_commit", "").strip(),
        "hf_repo": model.get("hf_repo", "lihzha/LAP-3B-Libero"),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "policy_config": model.get("policy_config", "lap_libero"),
        "policy_type": model.get("policy_type", "flow"),
        "lap_proto": lap_proto_path.read_text() if lap_proto_path.exists() else "",
        "lap_grpc_bridge_py": lap_grpc_bridge_path.read_text() if lap_grpc_bridge_path.exists() else "",
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("lap-userdata.sh.j2").render(**ctx)


def generate_openarm_isaac(config: dict, dry_run: bool) -> str:
    """OpenArm-Isaac: π0.5 (LeRobot pi05 folding_latest) × Isaac Lab bimanual Reach.

    Embeds the 3 sibling python assets (env.py / run_eval.py / overlay env_cfg) as base64 so the
    userdata can materialise them on the instance and mount them into the Isaac Lab container.
    """
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/openarm-isaac.yaml에 tasks 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
    deployment = config.get("deployment", {})
    model = config.get("model", {})

    asset_dir = BASE_DIR / "assets" / "openarm-isaac"
    asset_files = {
        "env_py_b64": "env.py",
        "run_eval_py_b64": "run_eval.py",
        "overlay_cfg_b64": "openarm_bi_vla_env_cfg.py",
        # Rewrites LeRobot @ pin for Python 3.11 (NGC isaac-lab Kit runtime): relaxes the
        # requires-python gate + converts PEP-695 type syntax. See the asset's docstring.
        "patch_lerobot_b64": "patch_lerobot_py311.py",
    }
    missing = [fn for fn in asset_files.values() if not (asset_dir / fn).exists()]
    if missing:
        print(f"[error] openarm-isaac asset 파일 없음: {missing} (assets/openarm-isaac/ 확인)", file=sys.stderr)
        sys.exit(1)

    ctx = {
        "tasks_json": tasks_json,
        "deployment": deployment,
        "hf_repo": model.get("hf_repo", "lerobot/folding_latest"),
        "hf_model_revision": model.get("hf_model_revision", ""),
        "openarm_isaac_commit": model.get("openarm_isaac_commit", ""),
        "lerobot_commit": model.get("lerobot_commit", ""),
        "isaac_lab_image": model.get("isaac_lab_image", "nvcr.io/nvidia/isaac-lab:2.3.0"),
    }
    for ctx_key, fname in asset_files.items():
        ctx[ctx_key] = base64.encodebytes((asset_dir / fname).read_bytes()).decode()

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("openarm-isaac-userdata.sh.j2").render(**ctx)


def generate_openarm_lift_act(config: dict, dry_run: bool) -> str:
    """OpenArm-Lift-ACT: teleop-free scripted demo COLLECTION for the unimanual Lift-Cube × ACT env.

    Embeds 2 sibling python assets (overlay env_cfg + collection driver) as base64 so the userdata
    can materialise them on the instance and mount them into the Isaac Lab container. No LeRobot,
    no checkpoint, no rollout video — the deliverable is a success-only demo HDF5 uploaded to S3.
    """
    model = config.get("model", {})
    collection = config.get("collection", {})
    if not collection:
        print("[error] models/openarm-lift-act.yaml에 collection 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    asset_dir = BASE_DIR / "assets" / "openarm-lift-act"
    asset_files = {
        "overlay_cfg_b64": "openarm_uni_lift_act_env_cfg.py",
        "collect_sm_b64": "collect_demos_sm.py",
    }
    missing = [fn for fn in asset_files.values() if not (asset_dir / fn).exists()]
    if missing:
        print(f"[error] openarm-lift-act asset 파일 없음: {missing} (assets/openarm-lift-act/ 확인)", file=sys.stderr)
        sys.exit(1)

    ctx = {
        "openarm_isaac_commit": model.get("openarm_isaac_commit", ""),
        "isaac_lab_image": model.get("isaac_lab_image", "nvcr.io/nvidia/isaac-lab:2.3.0"),
        "task_id": model.get("task_id", "Isaac-Lift-Cube-OpenArm-ACT-v0"),
        "register_module": model.get(
            "register_module",
            "openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.openarm_uni_lift_act_env_cfg",
        ),
        "collection": {
            "num_demos": collection.get("num_demos", 5),
            "num_envs": collection.get("num_envs", 16),
            "position_threshold": collection.get("position_threshold", 0.01),
            "max_steps": collection.get("max_steps", 200000),
            "cam_res": collection.get("cam_res", 224),
            "task": collection.get("task", "lift the cube"),
        },
    }
    for ctx_key, fname in asset_files.items():
        ctx[ctx_key] = base64.encodebytes((asset_dir / fname).read_bytes()).decode()

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), keep_trailing_newline=True)  # nosec B701 - shell script template
    return env.get_template("openarm-lift-act-userdata.sh.j2").render(**ctx)


def generate_pi(config: dict, resolved_vpc: str, resolved_nlb: str, dry_run: bool) -> str:
    tasks = config.get("tasks", [])
    if not tasks:
        print("[error] models/pi.yaml에 tasks 항목이 없습니다.", file=sys.stderr)
        sys.exit(1)

    tasks_json = _tasks_json_for_bash(tasks)
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
    parser.add_argument("--vla", required=True, choices=["gr00t", "gr00t-gr1", "gr00t-g1", "pi", "openvla-oft", "lap", "rldx", "rldx-simpler", "openarm-isaac", "openarm-lift-act"], help="VLA 모델")
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
    parser.add_argument(
        "--libero-suite", default=OFT_DEFAULT_SUITE, choices=list(OFT_LIBERO_SUITES),
        help="openvla-oft: LIBERO suite (default: 10 = LIBERO-Long; `long` is alias for `10`)",
    )
    parser.add_argument("--collect", action="store_true",
                        help="gr00t-g1: trajectory COLLECTION mode (success-only obs/action HDF5 for "
                             "N1.7 G1 adapter FT) instead of the default rollout/video mode.")
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
    elif args.vla == "gr00t-g1":
        rendered = generate_gr00t_g1(config, args.resolved_grpc, args.resolved_vpc, args.dry_run,
                                     collect=args.collect)
    elif args.vla == "openvla-oft":
        rendered = generate_openvla_oft(config, args.libero_suite, args.dry_run)
    elif args.vla == "lap":
        rendered = generate_lap(config, args.resolved_nlb, args.dry_run)
    elif args.vla in ("rldx", "rldx-simpler"):
        rendered = generate_rldx(config, args.dry_run, vla=args.vla)
    elif args.vla == "openarm-isaac":
        rendered = generate_openarm_isaac(config, args.dry_run)
    elif args.vla == "openarm-lift-act":
        rendered = generate_openarm_lift_act(config, args.dry_run)
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

    # Syntax-validate the rendered UserData before it can reach a deploy. A bash
    # syntax error (e.g. an apostrophe in a task description breaking a single-quoted
    # string) otherwise only surfaces on the EC2 instance: the script aborts before
    # sending cfn_signal and the stack hangs on its WaitCondition until timeout
    # (idle GPU billing). `bash -n` catches it here, loudly, at generation time.
    bash_check = subprocess.run(  # nosec B603,B607 - hardcoded command, dest is our own output
        ["bash", "-n", str(dest)],
        capture_output=True,
        text=True,
    )
    if bash_check.returncode != 0:
        print(f"[error] Generated {dest} has a bash syntax error:", file=sys.stderr)
        print(bash_check.stderr.strip(), file=sys.stderr)
        print("  Most common cause: an apostrophe or unescaped quote in a task "
              "description in models/{vla}.yaml.", file=sys.stderr)
        sys.exit(1)

    print(f"생성 완료: {dest}")


if __name__ == "__main__":
    main()
