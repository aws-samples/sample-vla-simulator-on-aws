#!/usr/bin/env python3
# Copyright 2026 — vla-simulator gr00t-g1 target (Step 2 trajectory collector)
#
# Collect (observation, action) trajectories from a GR00T N1.6 Unitree-G1 whole-body
# loco-manipulation rollout, writing the SUCCESSFUL episodes to HDF5 for downstream
# imitation fine-tuning of an N1.7 G1 adapter.
#
# WHAT THIS IS
#   A superset of NVIDIA's stock `gr00t/eval/rollout_policy.py` (pinned commit 77866395, the SAME
#   commit the gr00t-g1 userdata runs). We REUSE its env/policy factories by importing them, and we
#   re-implement ONLY the rollout loop so we can persist trajectories — no fork of the upstream file.
#
# ★ WHY WE RUN AT NATIVE n_action_steps (NOT 1) — the central design fact ★
#   First attempt forced n_action_steps=1 to get "one outer step == one clean (obs,action)". That
#   DESTROYED the policy: GR00T N1.6 G1 loco-manip scored 3/10 at n_action_steps=20 but 0/48 at
#   n_action_steps=1 (whole-body balance collapses — the WBC Balance/Walk ONNX assume a smooth,
#   chunk-consistent action stream; re-querying every sim step makes base/waist commands jitter and
#   the humanoid falls). action chunk size is NOT a data-format knob, it is PART OF THE POLICY.
#   (Durable lesson: knowledge/pai/concepts/gr00t/action-chunk-rollout-vs-collection.md.)
#
#   So we keep the policy at its native chunk (n_action_steps from the model/config, e.g. 20) and
#   capture per-timestep transitions from INSIDE the chunk via a recorder wrapper inserted BELOW
#   MultiStepWrapper. MultiStepWrapper.step() unrolls the chunk with a `for step in range(...)` loop
#   of single-step `super().step(act)` calls (multistep_wrapper.py:256-277); our recorder sits on
#   that inner boundary, so it sees single-step actions + single-frame obs = exactly per-timestep BC
#   pairs, while the policy still runs receding-horizon. Storage is per-step; execution is native.
#   The GR00T training loader rebuilds action chunks at train time via delta_indices (N1.7 G1 = 50).
#
# OBS / ACTION FORMAT — GR00T-NATIVE, DUMPED VERBATIM
#   Inference is delegated to the GR00T policy (ZMQ), so the chunk the policy emits is the GR00T
#   action dict (`action.left_arm`, ..., `action.base_height_command`, `action.navigate_command` for
#   UNITREE_G1). The recorder unpacks the per-step slice of that dict + the single-frame obs the base
#   env returns (GR00T modality dict: `video.ego_view`, `state.*`). Each modality key → its own HDF5
#   dataset, so the converter (gr00t_hdf5_to_lerobot.py) gets authoritative key names + per-key dims
#   with zero hand-assembly. We also snapshot policy.get_modality_config() into the HDF5 root attrs.
#
# OUTPUT — HDF5 (one file, --dataset_file), success-only
#   data
#     .attrs[env_name, embodiment_tag, modality_config_json, n_action_steps, video_fps, total]
#     demo_<i>  .attrs[num_samples=T, success=True]
#       obs/<modality_key>   (T, ...)   per-step obs (video uint8 HWC, state float32)
#       action/<action_key>  (T, d)     per-step action (float32)
#   The MP4 from VideoRecordingWrapper is kept too (uploaded separately by the userdata).
#
#   (obs,action) alignment: at inner sim step t we record the obs the base env just returned and the
#   action slice applied to reach it. We record action[t] paired with the obs BEFORE that action is
#   applied (pre-step pairing, standard BC convention) — see PerStepRecorder.step.
#
# ──────────────────────────────────────────────────────────────────────────────────────────────
# [확인 필요] — RUNTIME FACTS to verify on the FIRST collection run (cannot be known off-GPU):
#   (G1) Exact obs / action key NAMES and SHAPES — printed once at the first recorded step to stdout
#        (survives EC2 termination via the userdata log + S3). The dump is name-agnostic.
#   (G2) Per-step action slice shape. MultiStepWrapper feeds the base env `act[key] = value[step, :]`
#        (single-step, no chunk axis). The recorder stores it as-is; obs is single-frame from base env.
#   (G3) Success signal. We read terminations + the env `success`/`final_info` key (same extraction as
#        rollout_policy.py). Verify exported demo count > 0 on a run whose video shows a successful
#        place; if successes happen but 0 export, the success key path differs for this env.
#   (G4) Single env (n_envs=1), matching the verified 3/10 protocol. The recorder is per-env-instance.
#   (G5) Memory. One episode of 256x256 video buffered in RAM before flush (~280 MB @ 1440 steps).
#        Flushed per-episode so it never accumulates. terminate_on_success ends episodes promptly.
# ──────────────────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Upstream Isaac-GR00T is importable under the WBC venv (userdata clones to /home/ubuntu/Isaac-GR00T
# and pip-installs it editable). Import the rollout helpers by name — same pinned commit, zero drift.
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402,F401
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name  # noqa: E402
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper  # noqa: E402
from gr00t.eval.rollout_policy import (  # noqa: E402
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_gr00t_sim_policy,
    get_gym_env,
)

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402


def _to_numpy(x):
    """Coerce a (possibly torch) array-like to a contiguous numpy array."""
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    if hasattr(x, "detach"):  # torch tensor
        return np.ascontiguousarray(x.detach().cpu().numpy())
    return np.asarray(x)


class PerStepRecorder(gym.Wrapper):
    """Sits BELOW MultiStepWrapper, ABOVE the base GEAR-WBC env. MultiStepWrapper.step() unrolls the
    action chunk into single-step `super().step(act)` calls — each of those lands here, so we see the
    per-timestep (obs, action) pair while the policy above still runs at its native chunk size.

    We buffer the current episode's transitions; the driver reads/clears them at episode boundaries.
    Pre-step BC pairing: action[t] is paired with the obs the base env returned at t-1 (i.e. the obs
    the policy/SM saw before applying action[t]); on reset we seed `_last_obs` with the reset obs."""

    def __init__(self, env, video_keep: set[str] | None = None):
        super().__init__(env)
        self._last_obs: dict | None = None
        self.obs_buf: dict[str, list] = defaultdict(list)
        self.act_buf: dict[str, list] = defaultdict(list)
        self._printed = False
        # ★ Key filter (2026-06-17 main-run fix): the GEAR-WBC base env returns BOTH the registered
        #   GR00T modality obs (prefixed `state.*`, `video.*`) AND raw sim proprioception (`q`, `dq`,
        #   `ddq`, `tau_est`, `floating_base_*`, `wrist_pose`, `torso_*`) AND extra cameras
        #   (`ego_view_image`, `tpp_view_image`, `video.tpp_view`). Only the registered UNITREE_G1 keys
        #   feed N1.7 FT: state=7 keys (the `state.`-prefixed ones), video=ego_view. Dumping everything
        #   bloated the HDF5 to 138 GB and made state_dim=255 (vs registered 43) — the converter then
        #   mis-built modality.json. We keep ONLY:
        #     • obs keys with the `state.` modality prefix  (exactly the 7 registered legs/arms/hands/waist)
        #     • obs keys in `video_keep`                    (default {"video.ego_view"}; drops video.tpp_view)
        #   Raw sim keys carry NO modality prefix, so the prefix rule excludes them cleanly. Action keys
        #   are all registered already (7 keys, action.* prefix) → kept verbatim.
        self.video_keep: set[str] = video_keep if video_keep is not None else {"video.ego_view"}

    def _keep_obs_key(self, k: str) -> bool:
        if k.startswith("annotation"):
            return False  # language string stored once via --task, not a per-step array
        if k.startswith("state."):
            return True   # the 7 registered UNITREE_G1 state sub-keys (raw sim state has no `state.` prefix)
        return k in self.video_keep

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        # `action` here is a single-step dict (MultiStepWrapper sliced the chunk: act[key]=value[step,:]).
        obs, reward, terminated, truncated, info = self.env.step(action)
        prev = self._last_obs if self._last_obs is not None else obs

        if not self._printed:
            print("[collect] === per-step obs keys / shapes (G1) — [KEEP]/[drop] ===", flush=True)
            for k, v in prev.items():
                tag = "KEEP" if self._keep_obs_key(k) else "drop"
                print(f"[collect]   obs  [{tag}] {k}: {_to_numpy(v).shape} {_to_numpy(v).dtype}", flush=True)
            print("[collect] === per-step action keys / shapes (G1) ===", flush=True)
            for k, v in action.items():
                print(f"[collect]   act  {k}: {_to_numpy(v).shape} {_to_numpy(v).dtype}", flush=True)
            kept = [k for k in prev if self._keep_obs_key(k)]
            print(f"[collect] keeping {len(kept)} obs keys: {sorted(kept)}", flush=True)
            self._printed = True

        # obs[t] = what the policy saw before action[t] (pre-step pairing). Keep ONLY registered
        # GR00T modality keys (state.* + video_keep) — drop raw sim state + extra cameras (see ctor).
        for k, v in prev.items():
            if not self._keep_obs_key(k):
                continue
            self.obs_buf[k].append(_to_numpy(v))
        for k, v in action.items():
            self.act_buf[k].append(_to_numpy(v))

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def episode_len(self) -> int:
        return len(next(iter(self.act_buf.values()))) if self.act_buf else 0

    def drain(self) -> tuple[dict, dict]:
        """Return (obs_buf, act_buf) and clear for the next episode."""
        o, a = self.obs_buf, self.act_buf
        self.obs_buf, self.act_buf = defaultdict(list), defaultdict(list)
        self._last_obs = None
        return o, a


def _make_recording_env(env_name, recorder_holder, wrapper_configs, video_keep=None):
    """Mirror create_eval_env's wrapper stack but insert PerStepRecorder between the base env and
    MultiStepWrapper. (create_eval_env order: base -> [VideoRecordingWrapper] -> MultiStepWrapper.)
    We keep video recording too (MP4 deliverable), then the recorder, then MultiStepWrapper on top."""
    env = get_gym_env(env_name, env_idx=0, total_n_envs=1)

    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
        )

    # ★ per-step recorder — below MultiStepWrapper so it sees the unrolled single-step transitions.
    recorder = PerStepRecorder(env, video_keep=video_keep)
    recorder_holder.append(recorder)

    env = MultiStepWrapper(
        recorder,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def _extract_success(env_infos: dict, env_idx: int) -> bool:
    """Success extraction mirroring rollout_policy.py (live `success` + episode-end `final_info`)."""

    def _coerce(val) -> bool:
        if isinstance(val, list):
            return bool(np.any(val))
        if isinstance(val, np.ndarray):
            return bool(np.any(val))
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, np.integer)):
            return bool(int(val))
        raise ValueError(f"Unknown success dtype: {type(val)}")

    success = False
    if "success" in env_infos:
        success |= _coerce(env_infos["success"][env_idx])
    if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
        success |= _coerce(env_infos["final_info"][env_idx]["success"])
    return success


def collect(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    dataset_file: str,
    video_dir: str,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_action_steps: int = 20,
    video_fps: int = 20,
    video_crf: int = 22,
    steps_per_render: int = 2,
    max_total_episodes: int = 0,
    video_keep: set[str] | None = None,
) -> int:
    """Run a single-env rollout at NATIVE n_action_steps, dump every SUCCESSFUL episode's per-step
    transitions (captured below MultiStepWrapper) to HDF5.

    Returns the number of successful episodes exported. Stops at n_episodes successful demos OR
    max_total_episodes attempted (safety cap so a low success rate cannot loop forever)."""
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    # Native chunk (NOT 1) — preserves the verified success rate. The recorder below MultiStepWrapper
    # captures per-timestep transitions regardless of chunk size.
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
            fps=video_fps,
            crf=video_crf,
            steps_per_render=steps_per_render,
            n_action_steps=n_action_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    # Single env (n_envs=1), matching the verified 3/10 protocol. recorder_holder lets us reach the
    # PerStepRecorder instance from outside the SyncVectorEnv factory closure.
    recorder_holder: list[PerStepRecorder] = []
    env = gym.vector.SyncVectorEnv(
        [lambda: _make_recording_env(env_name, recorder_holder, wrapper_configs, video_keep=video_keep)]
    )
    recorder = recorder_holder[0]

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    # Snapshot the policy's modality config — authoritative key/horizon spec for the converter.
    modality_config_json = "{}"
    try:
        mc = policy.get_modality_config()
        modality_config_json = json.dumps(
            {k: getattr(v, "__dict__", str(v)) for k, v in mc.items()}, default=str
        )
        print(f"[collect] modality_config keys: {list(mc.keys())}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[collect] [warn] get_modality_config failed ({e!r}) — converter must infer keys", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(dataset_file)), exist_ok=True)

    observations, _ = env.reset()
    policy.reset()

    exported = 0
    attempted = 0
    total_transitions = 0
    if max_total_episodes <= 0:
        max_total_episodes = max(n_episodes * 10, n_episodes + 20)
    episode_success = False
    start_time = time.time()

    with h5py.File(dataset_file, "w") as h5:
        data_grp = h5.create_group("data")

        def _flush_episode():
            nonlocal exported, total_transitions, episode_success
            obs_buf, act_buf = recorder.drain()
            n_steps = len(next(iter(act_buf.values()))) if act_buf else 0
            if episode_success and n_steps > 0:
                demo = data_grp.create_group(f"demo_{exported}")
                demo.attrs["num_samples"] = n_steps
                demo.attrs["success"] = True
                ogrp = demo.create_group("obs")
                for key, steps in obs_buf.items():
                    ogrp.create_dataset(key, data=np.stack(steps, axis=0))
                agrp = demo.create_group("action")
                for key, steps in act_buf.items():
                    agrp.create_dataset(key, data=np.stack(steps, axis=0))
                exported += 1
                total_transitions += n_steps
                print(f"[collect] exported demo_{exported - 1}: {n_steps} steps "
                      f"({exported}/{n_episodes})", flush=True)
            episode_success = False

        outer_step = 0
        while exported < n_episodes:
            # Policy runs at its NATIVE chunk; MultiStepWrapper unrolls it; the recorder captures
            # each inner sim step. One outer env.step == n_action_steps sim steps.
            actions, _ = policy.get_action(observations)
            next_obs, rewards, terminations, truncations, env_infos = env.step(actions)

            episode_success |= _extract_success(env_infos, 0)

            done = bool(terminations[0] or truncations[0])
            if done:
                _flush_episode()
                attempted += 1
                if attempted >= max_total_episodes and exported < n_episodes:
                    print(f"[collect] [warn] hit max_total_episodes={max_total_episodes} with only "
                          f"{exported}/{n_episodes} successes — stopping (see G3 / success rate).",
                          file=sys.stderr, flush=True)
                    break

            observations = next_obs
            outer_step += 1
            if outer_step % 20 == 0:
                print(f"[collect] outer_step={outer_step} (~{outer_step * n_action_steps} sim steps) "
                      f"exported={exported}/{n_episodes} attempted={attempted}/{max_total_episodes} "
                      f"elapsed={time.time() - start_time:.0f}s", flush=True)

        data_grp.attrs["env_name"] = env_name
        data_grp.attrs["embodiment_tag"] = str(embodiment_tag)
        data_grp.attrs["modality_config_json"] = modality_config_json
        data_grp.attrs["n_action_steps"] = n_action_steps
        data_grp.attrs["video_fps"] = video_fps
        data_grp.attrs["total"] = total_transitions

    env.close()
    print(f"[collect] DONE: {exported} successful demos / {total_transitions} transitions "
          f"-> {dataset_file} (took {time.time() - start_time:.0f}s)", flush=True)
    return exported


def main() -> None:
    p = argparse.ArgumentParser(description="Collect GR00T G1 success-only trajectories to HDF5.")
    p.add_argument("--env_name", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=50,
                   help="Number of SUCCESSFUL demos to export before stopping.")
    p.add_argument("--max_episode_steps", type=int, default=1440)
    p.add_argument("--n_action_steps", type=int, default=20,
                   help="NATIVE policy action chunk size (do NOT set to 1 — breaks loco-manip balance; "
                        "per-step capture happens below MultiStepWrapper regardless).")
    p.add_argument("--dataset_file", type=str, default="./datasets/gr00t_g1.hdf5")
    p.add_argument("--video_dir", type=str, default="/tmp/gr00t-g1-collect-videos")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--policy_client_host", type=str, default="")
    p.add_argument("--policy_client_port", type=int, default=None)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--video_crf", type=int, default=22)
    p.add_argument("--steps_per_render", type=int, default=2)
    p.add_argument("--max_total_episodes", type=int, default=0,
                   help="Safety cap on ATTEMPTED episodes (0 = auto: 10x n_episodes).")
    p.add_argument("--video_keys", type=str, default="video.ego_view",
                   help="Comma-separated obs video keys to KEEP (default: video.ego_view — the only "
                        "camera N1.7 G1 FT uses). The GEAR-WBC env emits extra cameras (video.tpp_view, "
                        "ego_view_image, tpp_view_image) that bloat the HDF5/conversion ~4x; they are dropped. "
                        "state.* keys are always kept; raw sim state (q/dq/floating_base_*/...) always dropped.")
    args = p.parse_args()
    video_keep = {k.strip() for k in args.video_keys.split(",") if k.strip()}

    # Same policy-config validation as rollout_policy.py: EITHER model_path OR (host & port).
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: provide EITHER --model_path OR "
        "(--policy_client_host & --policy_client_port), not both."
    )

    n = collect(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        dataset_file=args.dataset_file,
        video_dir=args.video_dir,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_action_steps=args.n_action_steps,
        video_fps=args.video_fps,
        video_crf=args.video_crf,
        steps_per_render=args.steps_per_render,
        max_total_episodes=args.max_total_episodes,
        video_keep=video_keep,
    )
    if n == 0:
        print("[collect] [warn] exported 0 demos — check G3 (success key path) / rollout success rate",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
