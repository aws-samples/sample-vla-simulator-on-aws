#!/usr/bin/env python3
# Copyright 2026 — vla-simulator openarm-isaac target (Phase 2c driver)
#
# OpenArm bimanual × π0.5 (LeRobot pi05 `folding_latest`) — thin LeRobot rollout driver.
#
# WHY THIS EXISTS (Phase 2c decision — see research artifact "배포 wiring 제약")
#   LeRobot's `lerobot-eval` CLI builds the env through `factory.make_env`, which for an
#   `isaaclab_arena` env ONLY loads `env.py` via the HF-Hub download path
#   (`IsaaclabArenaEnv.hub_path` → `_download_hub_file` → hf_hub_download/snapshot_download).
#   There is NO local-file / `file://` / `os.path.exists` branch (verified: envs/factory.py
#   make_env + envs/utils.py _download_hub_file, lerobot HEAD d1b1c5c). And `IsaaclabArenaEnv`
#   does NOT override `create_envs`, so the non-hub fallback (`cfg.create_envs`) also fails.
#   ⇒ To run our LOCAL-disk env.py without publishing to the Hub, this driver assembles the
#   SAME objects `eval_main` builds and calls stock `eval_policy` directly, replacing only the
#   `factory.make_env(...)` call with a direct call to our local `env.py:make_env(...)`.
#   Everything downstream (policy, 4 processors, rollout loop, video write) is stock LeRobot —
#   NO rollout loop is hand-written.
#
# CONTRACT (verified against lerobot HEAD d1b1c5c — scripts/lerobot_eval.py:521 eval_main):
#   eval_main builds, in order:
#     envs                              = make_env(cfg.env, n_envs, use_async, trust_remote_code)
#     policy                            = make_policy(cfg.policy, env_cfg=cfg.env, rename_map=...)
#     preprocessor, postprocessor       = make_pre_post_processors(cfg.policy, pretrained_path, overrides)
#     env_preprocessor, env_postproc    = make_env_pre_post_processors(cfg.env, cfg.policy)
#     eval_policy_all(envs, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, ...)
#   We run exactly ONE task / ONE env, so we call `eval_policy` (single vec_env;
#   scripts/lerobot_eval.py:264) instead of `eval_policy_all` (which only adds the
#   {suite:{task_id:vec_env}} flatten + ThreadPool fan-out we do not need).
#
#   `eval_policy(..., max_episodes_rendered>0, videos_dir=...)` writes
#   `<videos_dir>/eval_episode_<i>.mp4` via `write_video` and returns
#   info["video_paths"] + info["aggregated"]["pc_success"] (lerobot_eval.py:405-471).
#
# HOW IT IS RUN (Phase 2c — userdata)
#   python run_eval.py \
#     --policy-path  <local folding_latest dir | lerobot/folding_latest> \
#     --output-dir   /tmp/openarm-isaac-out \
#     --task         "fold the cloth" \
#     --state-keys   "right_arm_pos,right_gripper_pos,left_arm_pos,left_gripper_pos" \
#     --camera-keys  "base,left_wrist,right_wrist" \
#     --state-dim 16 --action-dim 16 --camera-height 224 --camera-width 224 \
#     --episode-length 150 --n-episodes 1 --device cuda:0
#
# The driver + env.py + the overlay env_cfg.py are all copied next to each other on the instance;
# `env.py` must be importable as a top-level module (this file inserts its own dir on sys.path).

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path


def _load_local_make_env():
    """Import `make_env` from the sibling env.py without colliding with lerobot.envs.make_env.

    We load it under a private module name via importlib so that `import env` ambiguity (there is
    a `lerobot.envs` package) cannot shadow it. env.py keeps its module top isaaclab-free (only
    `from isaaclab.app import AppLauncher`), so importing it here does NOT boot Isaac Sim — the
    sim is launched lazily inside make_env() AFTER AppLauncher, preserving import order (A1).
    """
    here = Path(__file__).resolve().parent
    env_path = here / "env.py"
    if not env_path.exists():
        raise FileNotFoundError(f"env.py not found next to driver: {env_path}")
    spec = importlib.util.spec_from_file_location("openarm_isaac_env", str(env_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["openarm_isaac_env"] = module
    spec.loader.exec_module(module)
    return module.make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenArm-Isaac × π0.5 LeRobot rollout driver")
    p.add_argument("--policy-path", required=True,
                   help="folding_latest checkpoint dir (local) or HF repo id (lerobot/folding_latest)")
    p.add_argument("--output-dir", required=True, help="dir for videos/ + eval_info.json")
    p.add_argument("--task", default="fold the cloth",
                   help="language instruction passed to pi05 (folding_latest is a cloth-folding policy)")
    # state_keys ORDER == 16-dim layout (A3). Default matches folding_latest [R_arm,R_grip,L_arm,L_grip].
    p.add_argument("--state-keys",
                   default="right_arm_pos,right_gripper_pos,left_arm_pos,left_gripper_pos")
    p.add_argument("--camera-keys", default="base,left_wrist,right_wrist",
                   help="MUST match folding_latest config.image_features (observation.images.<key>)")
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--action-dim", type=int, default=16)
    p.add_argument("--camera-height", type=int, default=224)
    p.add_argument("--camera-width", type=int, default=224)
    p.add_argument("--episode-length", type=int, default=150,
                   help="rollout step cap for a demo-sized clip (A4); stock bimanual Reach is ~720")
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-episodes-rendered", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--use-amp", action="store_true", help="autocast during inference (off by default)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[run_eval] %(asctime)s %(message)s")
    log = logging.getLogger("run_eval")
    args = parse_args()

    # ── Stock LeRobot imports (the userdata pip-installs lerobot before running this) ──────────
    import torch
    from lerobot.configs.eval import EvalConfig  # noqa: F401  (documents the contract; not constructed)
    from lerobot.envs.configs import IsaaclabArenaEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.random_utils import set_seed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    # ── 1. Build the IsaaclabArenaEnv cfg DIRECTLY (no hub_path round-trip) ─────────────────────
    # hub_path keeps its class default but is never consulted: we call our local make_env, not
    # factory.make_env. The fields below are read by (a) our env.py make_env (task/device/
    # enable_cameras/headless/camera_height/camera_width/episode_length) and (b)
    # get_env_processors → IsaaclabArenaProcessorStep (state_keys/camera_keys), and (c) make_policy
    # feature wiring (state_dim/action_dim/camera features set in __post_init__).
    env_cfg = IsaaclabArenaEnv(
        task=args.task,
        episode_length=args.episode_length,
        num_envs=1,
        device=args.device,
        enable_cameras=True,
        headless=True,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        state_keys=args.state_keys,
        camera_keys=args.camera_keys,
    )
    log.info("env_cfg: state_keys=%s camera_keys=%s state_dim=%d action_dim=%d episode_length=%d",
             env_cfg.state_keys, env_cfg.camera_keys, env_cfg.state_dim, env_cfg.action_dim,
             env_cfg.episode_length)

    # ── 2. Build the vec env via OUR local env.py (replaces factory.make_env's hub download) ────
    log.info("Booting Isaac Sim + building OpenArm-Bi-VLA env (this launches the sim app)...")
    make_local_env = _load_local_make_env()
    envs = make_local_env(n_envs=1, use_async_envs=False, cfg=env_cfg)
    # envs == {suite: {0: adapter}} — pull out the single vec env.
    suite = next(iter(envs))
    vec_env = envs[suite][0]
    log.info("vec env ready: suite=%s num_envs=%d render_fps=%s",
             suite, vec_env.num_envs, vec_env.unwrapped.metadata.get("render_fps"))

    # ── 3. Policy + 4 processors (stock; mirrors eval_main lerobot_eval.py:543-564) ─────────────
    log.info("Loading policy config + weights from %s ...", args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    # IsaaclabArena env processor: observation.policy{state_keys}→observation.state ;
    # observation.camera_obs{cam}→observation.images.<cam> (processor/env_processor.py:158).
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    # ── 4. Run the rollout through STOCK eval_policy (writes videos + metrics) ──────────────────
    log.info("Running rollout: n_episodes=%d max_episodes_rendered=%d videos_dir=%s",
             args.n_episodes, args.max_episodes_rendered, videos_dir)
    amp_ctx = torch.autocast(device_type=device.type) if args.use_amp else nullcontext()
    with torch.no_grad(), amp_ctx:
        info = eval_policy(
            env=vec_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            max_episodes_rendered=args.max_episodes_rendered,
            videos_dir=videos_dir,
            start_seed=args.seed,
        )

    # ── 5. Persist metrics + close ──────────────────────────────────────────────────────────────
    info_path = out_dir / "eval_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    agg = info.get("aggregated", {})
    log.info("DONE. pc_success=%s avg_sum_reward=%s videos=%s",
             agg.get("pc_success"), agg.get("avg_sum_reward"), info.get("video_paths"))
    # HONEST LABEL: folding_latest on an empty Reach scene is OOD — success is expected to be 0.
    # This run proves the PIPE (π0.5 → 16-joint OpenArm motion in Isaac Lab), not task success.

    try:
        vec_env.close()
    except Exception as e:  # noqa: BLE001 — best-effort sim teardown
        log.warning("vec_env.close() raised (ignored): %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
