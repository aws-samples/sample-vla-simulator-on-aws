#!/usr/bin/env python3
# Copyright 2026 — vla-simulator openarm-lift-act target (Phase 1c converter)
#
# Convert an Isaac Lab demo HDF5 (produced by collect_demos_sm.py) into a LeRobotDataset that
# `lerobot-train` can train an ACT policy on.
#
# NO GPU / NO Isaac Sim. Only deps: h5py, numpy, lerobot. Runs wherever lerobot is installed
# (on the GPU instance right after collection, or locally).
#
# INPUT — Isaac Lab HDF5 layout (verified against hdf5_dataset_file_handler.py / episode_data.py):
#   data                                         (group)
#     .attrs["env_args"]   = json {env_name, type, sim_args:{dt, decimation, render_interval,...}}
#     .attrs["total"]      = total step count
#     demo_0, demo_1, ...                        (one group per recorded SUCCESSFUL episode)
#       .attrs["num_samples"] = T (steps in this episode)
#       .attrs["success"]     = True            (EXPORT_SUCCEEDED_ONLY ⇒ always True here)
#       obs/                                     (the `policy` obs group, concatenate_terms=False)
#         arm_joint_pos  (T, 7)  float32         arm joint positions (absolute)
#         gripper_pos    (T, 1)  float32         single finger joint position
#         wrist          (T, H, W, 3) uint8      wrist camera RGB (channel-last)
#         table          (T, H, W, 3) uint8      table camera RGB (channel-last)
#       actions          (T, 8)  float32         [ee_pos(3), ee_quat_wxyz(4), gripper(1)]
#       processed_actions, initial_state, states ...  (recorded but NOT used by ACT)
#
#   NOTE on (obs, action) alignment: both obs and actions are recorded PRE-step
#   (PreStepFlatPolicyObservationsRecorder + PreStepActionsRecorder), so obs[t] is the observation
#   the policy saw and action[t] is the action it then took — the correct pairing for behavior
#   cloning. No off-by-one shift is applied.
#
# OUTPUT — a LeRobotDataset with features:
#   observation.state          (8,)        float32  = concat(arm_joint_pos(7), gripper_pos(1))
#   observation.images.wrist   (H, W, 3)   video    = obs/wrist
#   observation.images.table   (H, W, 3)   video    = obs/table
#   action                     (8,)        float32  = actions  [ee_pos(3), ee_quat(4), gripper(1)]
#   task                       (per-frame)  str     = --task
#
#   The 8-dim observation.state matches the overlay's `policy` state terms in field order
#   (arm_joint_pos then gripper_pos) — overlay L5. ACT's action == the SAME 8-dim env action the
#   ACT eval env (Isaac-Lift-Cube-OpenArm-ACT-v0) consumes, so train/eval distributions match.
#
# USAGE
#   python hdf5_to_lerobot.py \
#     --hdf5 ./datasets/openarm_lift_act.hdf5 \
#     --repo-id local/openarm-lift-act \
#     --root ./lerobot_datasets/openarm-lift-act \
#     --task "lift the cube"
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# [확인 필요] — verify on first real conversion (after the first GPU collection produces an HDF5):
#   (D1) State term order. We read obs/arm_joint_pos then obs/gripper_pos and concatenate in THAT
#        order. If the overlay's PolicyCfg field order ever changes, update _STATE_TERMS to match —
#        the ACT eval env builds observation.state the same way, so the two MUST agree.
#   (D2) fps. Derived as round(1 / (dt * decimation)) from env_args.sim_args. If env_args is absent
#        (older HDF5), falls back to --fps. Confirm against the printed value.
#   (D3) Image channel order. mdp.image(normalize=False) yields channel-last uint8 (H,W,3).
#        LeRobot's validate_feature_image_or_video accepts channel-last. We declare feature shape
#        from the actual array shape, so a (3,H,W) clone would still be declared correctly — but ACT
#        expects HWC video; if a future Isaac Lab returns CHW, transpose here.
# ─────────────────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np

# State obs terms read from obs/<term> and concatenated, IN THIS ORDER, into observation.state.
# MUST match the overlay PolicyCfg field order (arm_joint_pos, gripper_pos). See D1.
_STATE_TERMS = ["arm_joint_pos", "gripper_pos"]
# Camera obs terms → observation.images.<name>.
_IMAGE_TERMS = ["wrist", "table"]


def _derive_fps(data_group, fallback_fps: int) -> int:
    """fps = 1 / (sim.dt * decimation), read from data.attrs['env_args'] (see D2)."""
    env_args_raw = data_group.attrs.get("env_args")
    if env_args_raw is None:
        print(f"[warn] no env_args in HDF5 — falling back to --fps={fallback_fps}", file=sys.stderr)
        return fallback_fps
    try:
        env_args = json.loads(env_args_raw)
        sim = env_args["sim_args"]
        fps = round(1.0 / (float(sim["dt"]) * int(sim["decimation"])))
        print(f"[info] derived fps={fps} (dt={sim['dt']}, decimation={sim['decimation']})")
        return fps
    except (KeyError, ValueError, TypeError) as e:
        print(f"[warn] could not derive fps from env_args ({e!r}) — falling back to --fps={fallback_fps}",
              file=sys.stderr)
        return fallback_fps


def _sorted_demo_keys(data_group) -> list[str]:
    """Return demo_* group keys sorted numerically (demo_0, demo_1, ..., demo_10)."""
    keys = [k for k in data_group.keys() if k.startswith("demo_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _build_features(demo_group, state_dim: int, action_dim: int) -> dict:
    """Construct the LeRobot features dict, reading image H/W/C from the actual arrays (see D3)."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": [f"arm_joint_{i}" for i in range(state_dim - 1)] + ["gripper"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {"axes": ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz", "gripper"]}
            if action_dim == 8 else None,
        },
    }
    for cam in _IMAGE_TERMS:
        arr = demo_group["obs"][cam]
        # (T, H, W, C) channel-last
        _, h, w, c = arr.shape
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }
    return features


def convert(hdf5_path: str, repo_id: str, root: str, task: str, fallback_fps: int) -> None:
    # lerobot import is deferred so --help works without it installed
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"{hdf5_path} has no top-level 'data' group — not an Isaac Lab demo HDF5.")
        data = f["data"]
        demo_keys = _sorted_demo_keys(data)
        if not demo_keys:
            raise RuntimeError(f"{hdf5_path} contains zero demo_* episodes. Collection produced no successes.")
        print(f"[info] {len(demo_keys)} episodes in {hdf5_path}")

        fps = _derive_fps(data, fallback_fps)

        # Validate state/action term presence on the first demo + size the features.
        first = data[demo_keys[0]]
        for term in _STATE_TERMS:
            if term not in first["obs"]:
                raise RuntimeError(f"obs/{term} missing in {demo_keys[0]} — check overlay PolicyCfg field names (D1).")
        for cam in _IMAGE_TERMS:
            if cam not in first["obs"]:
                raise RuntimeError(f"obs/{cam} missing in {demo_keys[0]} — were cameras enabled (--enable_cameras)?")
        if "actions" not in first:
            raise RuntimeError(f"actions missing in {demo_keys[0]}.")

        state_dim = sum(int(first["obs"][t].shape[1]) for t in _STATE_TERMS)
        action_dim = int(first["actions"].shape[1])
        print(f"[info] state_dim={state_dim}, action_dim={action_dim}")
        if action_dim != 8:
            print(f"[warn] action_dim={action_dim} (expected 8 = ee_pose7 + gripper1). Proceeding, but "
                  "ACT eval must use the SAME action layout.", file=sys.stderr)

        features = _build_features(first, state_dim, action_dim)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type="openarm_unimanual",
            use_videos=True,
        )

        total_frames = 0
        for demo_key in demo_keys:
            demo = data[demo_key]
            num_samples = int(demo.attrs.get("num_samples", len(demo["actions"])))

            # Read whole-episode arrays once (T, …)
            state_parts = [np.asarray(demo["obs"][t], dtype=np.float32) for t in _STATE_TERMS]  # each (T, d)
            states = np.concatenate(state_parts, axis=1)                                          # (T, state_dim)
            actions = np.asarray(demo["actions"], dtype=np.float32)                               # (T, action_dim)
            images = {cam: np.asarray(demo["obs"][cam]) for cam in _IMAGE_TERMS}                  # (T, H, W, 3) uint8

            # Defensive length alignment: all streams must share T = num_samples.
            T = min(num_samples, len(states), len(actions), *(len(images[c]) for c in _IMAGE_TERMS))
            if T != num_samples:
                print(f"[warn] {demo_key}: length mismatch (num_samples={num_samples}, using T={T})", file=sys.stderr)

            for t in range(T):
                frame = {
                    "observation.state": states[t],
                    "action": actions[t],
                    "task": task,
                }
                for cam in _IMAGE_TERMS:
                    frame[f"observation.images.{cam}"] = images[cam][t]  # (H, W, 3) uint8, channel-last
                dataset.add_frame(frame)

            dataset.save_episode()
            total_frames += T
            print(f"[info] {demo_key}: saved {T} frames")

        dataset.finalize()
        print(f"[done] {len(demo_keys)} episodes / {total_frames} frames → {root}  (repo_id={repo_id}, fps={fps})")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Isaac Lab demo HDF5 → LeRobotDataset (ACT).")
    p.add_argument("--hdf5", required=True, help="Input Isaac Lab demo HDF5 path.")
    p.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. local/openarm-lift-act.")
    p.add_argument("--root", required=True, help="Output dataset directory.")
    p.add_argument("--task", default="lift the cube", help="Natural-language task string stored per frame.")
    p.add_argument("--fps", type=int, default=50, help="Fallback fps if env_args is missing (see D2).")
    args = p.parse_args()
    convert(args.hdf5, args.repo_id, args.root, args.task, args.fps)


if __name__ == "__main__":
    main()
