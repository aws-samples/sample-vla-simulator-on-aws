# Copyright 2026 — vla-simulator openarm-isaac target (Phase 2b adapter)
#
# OpenArm bimanual × π0.5 (LeRobot pi05 `folding_latest`) — LeRobot EnvHub `env.py`.
#
# WHAT THIS IS
#   The ONE piece of glue Phase 2 needs. LeRobot's `lerobot-eval` rollout drives any environment
#   exposed through a hub-style `make_env(n_envs, use_async_envs, cfg)` (see lerobot
#   `envs/factory.py:make_env` → `_call_make_env`). This file is that `make_env` for our
#   `Isaac-Reach-OpenArm-Bi-VLA-v0` overlay env (registered by `openarm_bi_vla_env_cfg.py`, Phase 2a).
#
#   We reuse LeRobot's stock rollout end-to-end — NO hand-written rollout loop. The reuse chain:
#     lerobot_eval.rollout()
#       → preprocess_observation()         passes obs["policy"]/obs["camera_obs"] straight through
#       → IsaaclabArenaProcessorStep        observation.policy{state_keys}→observation.state ;
#                                           observation.camera_obs{cam}→observation.images.<cam>
#       → pi05 preprocessor → policy.select_action → postprocessor
#       → env.step(numpy (N,16))            ← THIS adapter converts numpy→torch for Isaac
#   Verified against: lerobot envs/utils.py:122 (isaac branch), processor/env_processor.py:156,
#   scripts/lerobot_eval.py:98 (rollout), envs/configs.py:641 (IsaaclabArenaEnv).
#
# WHY A CUSTOM VEC-ENV ADAPTER (and NOT gym.vector.SyncVectorEnv)
#   Isaac Lab's ManagerBasedRLEnv is ALREADY internally vectorized over `scene.num_envs`: it
#   takes a torch action (N, A) and returns torch obs/reward batched over N. Wrapping it in a
#   SyncVectorEnv (the LeIsaac n_envs=1 demo path) would double-batch. Instead `_LeRobotIsaacVecEnv`
#   presents the Isaac env *directly* as the `gym.vector.VectorEnv` surface that rollout() touches:
#     .num_envs · .reset(seed) · .step(numpy)→(obs, reward_np, term_np, trunc_np, info)
#     .call(name) · .unwrapped.metadata["render_fps"] · .call("render")  (see rollout/eval citations)
#
# HOW IT IS RUN (Phase 2c — userdata)
#   lerobot-eval --env.type=isaaclab_arena \
#                --env.hub_path=<this repo>:env.py            (or local-load; see Phase 2c) \
#                --env.state_keys="right_arm_pos,right_gripper_pos,left_arm_pos,left_gripper_pos" \
#                --env.camera_keys="base,left_wrist,right_wrist" \
#                --env.enable_cameras=true --env.headless=true --env.task="<folding instruction>" \
#                --env.action_dim=16 --env.state_dim=16 \
#                --policy.path=lerobot/folding_latest --policy.device=cuda \
#                --eval.batch_size=1 --eval.n_episodes=1 --trust_remote_code=true
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# [확인 필요] — RUNTIME FACTS to verify on FIRST GPU deploy (cannot be known without Isaac Sim):
#
#   (A1) IMPORT ORDER: AppLauncher MUST boot before ANY `isaaclab.envs` / openarm import. This file
#        keeps the module top isaaclab-free (only `from isaaclab.app import AppLauncher`, which is the
#        documented-safe import) and defers every other isaac import into make_env(), AFTER
#        `app_launcher.app`. Verify nothing in the lerobot import path booted omni first.
#
#   (A2) OVERLAY REGISTRATION: make_env imports the overlay module by dotted path to fire its
#        gym.register("Isaac-Reach-OpenArm-Bi-VLA-v0"). Requires `pip install -e source/openarm`
#        (Phase 2c) so `openarm.tasks...` is importable and the overlay file is copied into
#        `.../bimanual/reach/config/`. Verify the dotted path matches the installed package.
#
#   (A3) state_keys ORDER == 16-dim layout. IsaaclabArenaProcessorStep concatenates policy-group
#        terms in the ORDER GIVEN ON THE CLI (--env.state_keys), not the cfg order. The default
#        below — right_arm_pos, right_gripper_pos, left_arm_pos, left_gripper_pos — matches
#        folding_latest's [R_j1..7, R_grip, L_j1..7, L_grip]. Keep CLI and overlay term names in sync.
#
#   (A4) max_episode_length: stock bimanual Reach is episode_length_s≈24s @ 30Hz ≈ 720 steps —
#        long for a demo clip. We CAP rollout length at cfg.episode_length (default 300; override
#        ~150 for a ~5s @30fps clip) via `_max_episode_steps`. The Isaac env's own timeout is left
#        intact; capping only bounds the rollout loop (rollout marks done at max_steps).
# ─────────────────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import numpy as np
import torch  # SAFE before sim boot — torch does not import omni (cf. Isaac Lab rsl_rl/play.py top imports)

# SAFE top-level import: AppLauncher is the class that *boots* Isaac Sim; importing it does not.
# Every other isaaclab.envs / openarm import is deferred into make_env() — see (A1).
from isaaclab.app import AppLauncher

# Module-global handle to the running Isaac Sim app (kept alive; closed via adapter.close()).
_SIMULATION_APP = None


# ── Defaults (overridable via the IsaaclabArenaEnv cfg passed to make_env) ──────────────────────
_GYM_ID = "Isaac-Reach-OpenArm-Bi-VLA-v0"
_SUITE = "Isaac-Reach-OpenArm-Bi-VLA"
# Dotted path of the Phase 2a overlay once copied into the installed openarm package (see A2).
_OVERLAY_MODULE = (
    "openarm.tasks.manager_based.openarm_manipulation.bimanual.reach.config.openarm_bi_vla_env_cfg"
)
_DEFAULT_TASK = "fold the cloth"  # folding_latest is a cloth-folding policy; empty Reach scene = OOD (pipe-proof only)


def _boot_isaac(headless: bool, enable_cameras: bool, device: str) -> None:
    """Launch Isaac Sim exactly once. Must run before any isaaclab.envs / openarm import (A1)."""
    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        return
    # AppLauncher accepts a kwargs/dict of launcher args (app_launcher.py:59, :200 enable_cameras).
    # enable_cameras=True is REQUIRED whenever the scene has camera sensors or we render rgb_array.
    app_launcher = AppLauncher({"headless": headless, "enable_cameras": enable_cameras, "device": device})
    _SIMULATION_APP = app_launcher.app


def _cfg_get(cfg: Any, name: str, default):
    """Read an attribute off the IsaaclabArenaEnv cfg, tolerating None / missing."""
    if cfg is None:
        return default
    val = getattr(cfg, name, default)
    return default if val is None else val


class _LeRobotIsaacVecEnv:
    """Presents a single internally-vectorized Isaac ManagerBasedRLEnv as the gym.vector.VectorEnv
    surface that lerobot_eval.rollout() / eval_policy() touch. NOT a SyncVectorEnv (no double-batch).

    Surface used by rollout (scripts/lerobot_eval.py):
      .num_envs                              (:156, :311)
      .reset(seed=list|None) -> (obs, info)  (:159)
      .call("_max_episode_steps") -> list    (:157)
      .call("task_description"/"task")       (:174, :177)
      .step(numpy (N,A)) -> (obs, r, term, trunc, info)   (:198)
      .call("render") -> list[np.ndarray]    (render_frame :332)
      .unwrapped.metadata["render_fps"]      (:394, :422)
    """

    def __init__(self, isaac_env, task: str, max_episode_steps: int):
        self._env = isaac_env  # the ManagerBasedRLEnv (possibly under gym wrappers)
        self._u = isaac_env.unwrapped  # ManagerBasedRLEnv proper (has num_envs, device, render, metadata)
        self._task = task
        self._max_episode_steps = int(max_episode_steps)
        self.num_envs = int(self._u.num_envs)
        # Proxy metadata so .unwrapped.metadata["render_fps"] resolves (manager_based_rl_env.py:88).
        # Isaac sets metadata["render_fps"] = 1 / step_dt as a *float* (true division). lerobot's
        # write_video (io_utils.py:79) passes that fps straight into PyAV's
        # container.add_stream("libx264", rate=fps), and PyAV 15.x's to_avrational() rejects a bare
        # float (`AttributeError: 'float' object has no attribute 'numerator'`) — the 14th-deploy
        # video-encode failure. A plain setdefault() is a no-op here because Isaac already populated
        # the key, so coerce render_fps to an int unconditionally (libx264 rate is conventionally int).
        self.metadata = dict(getattr(self._u, "metadata", {}) or {})
        _fps = self.metadata.get("render_fps")
        if _fps is None:
            _fps = 1.0 / float(self._u.step_dt)
        self.metadata["render_fps"] = int(round(float(_fps)))

    # rollout reads env.unwrapped.metadata[...] — we ARE the unwrapped vec env.
    @property
    def unwrapped(self):
        return self

    @property
    def _device(self):
        return self._u.device

    def reset(self, seed=None, options=None):
        # rollout passes seed as a list/range (one per sub-env) or None. Isaac wants int|None.
        if seed is not None and not isinstance(seed, int):
            seed = next(iter(seed), None)
        obs, info = self._env.reset(seed=seed, options=options)
        # obs == {"policy": {term: torch}, "camera_obs": {cam: torch}} — passed through unchanged
        # by preprocess_observation's isaac branch (envs/utils.py:122).
        return obs, info

    def step(self, action):
        # rollout hands us numpy (N, action_dim); Isaac.step wants torch (N, A) on the sim device.
        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action)
        else:
            action_t = action
        action_t = action_t.to(device=self._device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = self._env.step(action_t)
        # rollout does torch.from_numpy(reward) / numpy boolean ops → convert torch→numpy here.
        reward_np = _to_numpy(reward)
        terminated_np = _to_numpy(terminated).astype(bool)
        truncated_np = _to_numpy(truncated).astype(bool)
        # Empty Reach scene has no task success → no "is_success" → rollout records success=False.
        # That is the HONEST label for a pipe-proof (motion) demo; do NOT fabricate success.
        return obs, reward_np, terminated_np, truncated_np, info

    def call(self, name: str, *args, **kwargs):
        """gym.vector.VectorEnv.call analogue: returns a per-sub-env list."""
        if name == "_max_episode_steps":
            return [self._max_episode_steps] * self.num_envs
        if name in ("task_description", "task"):
            return [self._task] * self.num_envs
        if name == "render":
            frame = self._u.render()  # rgb_array → (H, W, 3) uint8 numpy (manager_based_rl_env.py:243)
            return [frame] * self.num_envs
        # Fallback: try to read the attribute off the underlying env.
        attr = getattr(self._u, name)
        return [attr] * self.num_envs

    def get_attr(self, name: str):
        # check_env_attributes_and_types() probes task_description/task via get_attr (envs/utils.py:223).
        return self.call(name)

    def render(self):
        return self._u.render()

    def close(self):
        try:
            self._env.close()
        finally:
            global _SIMULATION_APP
            if _SIMULATION_APP is not None:
                _SIMULATION_APP.close()
                _SIMULATION_APP = None


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    # torch tensor
    return x.detach().to("cpu").numpy()


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg: Any = None):
    """LeRobot EnvHub entry point. Boots Isaac Sim, builds the OpenArm-Bi-VLA env, and returns it
    in the normalized {suite: {task_id: vec_env}} mapping LeRobot expects (_normalize_hub_result).

    Args:
        n_envs: number of parallel envs (= eval.batch_size). First bring-up should use 1 — pi05's
            action-chunk queue is shared across the batch, so n_envs>1 runs in lockstep (acceptable
            but unnecessary for a single demo clip).
        use_async_envs: ignored — Isaac Sim is a single in-process GPU app; async vec env is N/A.
        cfg: the IsaaclabArenaEnv config (envs/configs.py:641). Read for task / camera res / device /
            episode cap / enable_cameras.
    """
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")

    task = str(_cfg_get(cfg, "task", _DEFAULT_TASK))
    device = str(_cfg_get(cfg, "device", "cuda:0"))
    enable_cameras = bool(_cfg_get(cfg, "enable_cameras", True))
    headless = bool(_cfg_get(cfg, "headless", True))
    cam_h = int(_cfg_get(cfg, "camera_height", 224))
    cam_w = int(_cfg_get(cfg, "camera_width", 224))
    episode_cap = int(_cfg_get(cfg, "episode_length", 300))

    # ── (A1) boot Isaac Sim FIRST, then do all deferred isaac imports ───────────────────────────
    _boot_isaac(headless=headless, enable_cameras=enable_cameras, device=device)

    import gymnasium as gym  # noqa: E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

    # (A2) Trigger gym.register for the overlay. Importing the openarm tasks package auto-imports
    # subpackages (import_packages), but we import the overlay explicitly to be deterministic.
    import importlib  # noqa: E402
    import os  # noqa: E402
    import sys  # noqa: E402

    # (A2b) openarm's stock config modules use ABSOLUTE imports rooted at the REPO, e.g.
    #   `from source.openarm.openarm.tasks...assets.openarm_bimanual import OPEN_ARM_HIGH_PD_CFG`
    #   (bimanual/reach/config/joint_pos_env_cfg.py:25 — pulled in by our overlay's
    #   `from .joint_pos_env_cfg import OpenArmReachEnvCfg`). `source` has no __init__.py, so it only
    #   resolves as a PEP-420 namespace package when the repo ROOT is on sys.path. `pip install -e
    #   source/openarm` exposes the top-level `openarm` package but NOT `source`, and openarm's own
    #   docs always run from the repo root (cwd on sys.path) — our driver runs from /workspace/assets,
    #   so we must add the repo root explicitly. Derive it from the installed openarm package:
    #   <repo>/source/openarm/openarm/__init__.py  → repo root = parents[3] of openarm.__file__.
    import openarm  # noqa: E402  (top-level editable package)

    _openarm_init = getattr(openarm, "__file__", None)
    if _openarm_init:
        # .../source/openarm/openarm/__init__.py → up 3 dirs = .../source/openarm ; up 4 = repo root
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(_openarm_init)))))
        if os.path.isdir(os.path.join(_repo_root, "source", "openarm", "openarm")) and _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

    import openarm.tasks  # noqa: F401, E402  (fires base env registrations)

    importlib.import_module(_OVERLAY_MODULE)  # fires Isaac-Reach-OpenArm-Bi-VLA-v0 registration

    # Build the env cfg from the registry (loads OpenArmBiVLAReachEnvCfg) and override num_envs/cams.
    env_cfg = parse_env_cfg(_GYM_ID, device=device, num_envs=n_envs)
    # Camera resolution is owned by the overlay; mirror cfg here so a CLI override propagates.
    for cam in ("base_cam", "left_wrist_cam", "right_wrist_cam"):
        sensor = getattr(env_cfg.scene, cam, None)
        if sensor is not None:
            sensor.height = cam_h
            sensor.width = cam_w

    # rgb_array render_mode + enable_cameras is what makes env.render() / camera obs produce frames.
    isaac_env = gym.make(_GYM_ID, cfg=env_cfg, render_mode="rgb_array")

    # Cap rollout length for a demo-sized clip (A4); Isaac's own timeout is left intact.
    max_steps = min(int(isaac_env.unwrapped.max_episode_length), episode_cap)

    adapter = _LeRobotIsaacVecEnv(isaac_env, task=task, max_episode_steps=max_steps)
    return {_SUITE: {0: adapter}}
