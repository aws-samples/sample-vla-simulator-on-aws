#!/usr/bin/env python3
# Copyright 2026 — vla-simulator gr00t-g1 target (Step 2 converter)
#
# Convert the success-only trajectory HDF5 produced by collect_trajectories.py into a
# GR00T-flavored LeRobotDataset (LeRobot data/video/meta layout + a meta/modality.json overlay
# + statistics.json) that a GR00T N1.7 G1 adapter fine-tune can train on.
#
# NO GPU / NO Isaac Sim needed for the LeRobot write itself. Deps: h5py, numpy, lerobot.
# Run it on the GPU instance right after collection (lerobot is already in the WBC venv), or
# locally wherever lerobot is installed.
#
# WHY THIS IS DATA-DRIVEN (no hand-assembled G1 dims)
#   The collector dumped each GR00T modality key as its OWN HDF5 dataset (obs/<key>, action/<key>)
#   AND snapshotted the policy server's get_modality_config() into data.attrs["modality_config_json"].
#   So this converter never guesses G1's action/state dimensions or key order — it reads the actual
#   array dims from the HDF5 and the key grouping/order from the modality-config snapshot. This is the
#   "match the verified lock" rule: the checkpoint's own processor config is the source of truth, not
#   a hard-coded layout. (The cloudwalk checkpoint's experiment_cfg/final_processor_config.json lists
#   UNITREE_G1 action keys: left_arm, right_arm, left_hand, right_hand, waist, base_height_command,
#   navigate_command — but we take the ORDER and dims from the live snapshot, not from this comment.)
#
# INPUT — collect_trajectories.py HDF5 layout
#   data
#     .attrs[env_name, embodiment_tag, modality_config_json, n_action_steps, video_fps, total]
#     demo_<i>
#       .attrs[num_samples=T, success=True]
#       obs/<modality_key>    (T, ...)    video.* → (T,H,W,3) uint8 ; state.* → (T, d) float32
#       action/<action_key>   (T, d)      float32
#
# OUTPUT — GR00T LeRobot dataset (documented in knowledge/pai/concepts/gr00t/gr00t-new-robot-training-guide.md §8)
#   <root>/
#     meta/modality.json        state.<k>/action.<k> → {start,end} index ranges into the flat vectors;
#                               video.<k> → {original_key}; annotation.human.task_description
#     meta/info.json, meta/episodes.jsonl, meta/stats (lerobot-managed)
#     statistics.json           GR00T per-key mean/std/min/max/q01/q99 computed from THIS data
#     data/chunk-000/episode_*.parquet   per-timestep flat observation.state + action + task + indices
#     videos/chunk-000/observation.images.<cam>/episode_*.mp4
#
#   The flat observation.state is concat(state sub-keys in modality-config order); flat action is
#   concat(action sub-keys in modality-config order). modality.json records each sub-key's [start,end)
#   so the GR00T data loader re-splits them. Video keys map video.<name> → observation.images.<name>.
#
# USAGE
#   python gr00t_hdf5_to_lerobot.py \
#     --hdf5 ./datasets/gr00t_g1.hdf5 \
#     --repo-id local/gr00t-g1-applepnp \
#     --root ./lerobot_datasets/gr00t-g1-applepnp \
#     --task "pick up the apple and place it on the plate"
#
# ──────────────────────────────────────────────────────────────────────────────────────────────
# H1-H4 RESOLVED by vla-ft (cc-58921, 2026-06-16), source = NVIDIA/Isaac-GR00T main@65cc4a192e6d
# (N1.7 EA) actual code. Durable recipe: knowledge/pai/concepts/gr00t/gr00t-n17-finetune-recipe.md.
#   (H1) LeRobot VERSION = v2.1, emitted on THIS (producer) side. The N1.7 training loader does NOT
#        assert codebase_version (reads data_path/chunks_size/fps/features only), but v2.1 is the
#        producer contract. We assert the created dataset reports v2.x and stamp meta/info.json's
#        codebase_version to v2.1 if the installed lerobot wrote a newer tag (defensive).
#   (H2) modality.json KEY NAMING = flat groups `state`/`action`/`video`/`annotation` (NOT
#        `observation.state.`-prefixed). Sub-keys MUST match the registered UNITREE_G1 config
#        (embodiment_configs.py:115-192): state=left_leg,right_leg,waist,left_arm,right_arm,left_hand,
#        right_hand ; action=left_arm,right_arm,left_hand,right_hand,waist,base_height_command,
#        navigate_command ; video=ego_view ; annotation=human.task_description (NO `.action.`).
#        Our cloudwalk-derived state/action sub-keys already match; only the annotation key was fixed.
#   (H3) ACTION horizon = 50 (registered UNITREE_G1 action delta_indices=range(50)). The loader builds
#        the 50-step chunk from our per-timestep (n_action_steps=1) store — do NOT pre-chunk (correct).
#        ⚠️ episodes shorter than ~50 frames get padded/clamped; prefer episode length ≥ ~50 frames.
#   (H4) Action rep: store RAW absolute as the policy emitted. The GR00T loader does relative
#        conversion itself (left_arm/right_arm are RELATIVE; hands/waist/base/navigate ABSOLUTE).
#        cloudwalk's use_relative_action=true is IGNORED here (no silent convert). vla-ft generates
#        meta/relative_stats.json on the FT side; we ship raw + statistics.json only.
# ──────────────────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np


def _sorted_demo_keys(data_group) -> list[str]:
    keys = [k for k in data_group.keys() if k.startswith("demo_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _key_order(group, prefix: str, modality_cfg: dict) -> list[str]:
    """Return the sub-keys under `prefix` (e.g. 'state', 'action') in modality-config order.

    Falls back to the HDF5's own dataset order (sorted) if the modality config does not enumerate them.
    The modality config snapshot maps a modality name to an object with `modality_keys` listing the
    fully-qualified keys (e.g. 'state.left_arm'). We honor that order; otherwise sort the HDF5 keys."""
    cfg_keys: list[str] = []
    if prefix in modality_cfg:
        mk = modality_cfg[prefix].get("modality_keys") if isinstance(modality_cfg[prefix], dict) else None
        if mk:
            cfg_keys = [k for k in mk if k in group]
    if cfg_keys:
        # append any HDF5 keys missing from the config (defensive), preserving config order first
        extra = sorted(k for k in group.keys() if k not in cfg_keys)
        return cfg_keys + extra
    return sorted(group.keys())


def _flatten_and_ranges(demo, sub_keys: list[str], grp_name: str) -> tuple[np.ndarray, dict]:
    """Concatenate sub-key arrays along the feature axis → (T, total_dim); return ranges {key:[s,e)}."""
    arrays = []
    ranges: dict[str, list[int]] = {}
    cursor = 0
    for k in sub_keys:
        a = np.asarray(demo[grp_name][k], dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        d = a.shape[1]
        ranges[k] = [cursor, cursor + d]
        cursor += d
        arrays.append(a)
    flat = np.concatenate(arrays, axis=1) if arrays else np.empty((0, 0), dtype=np.float32)
    return flat, ranges


def _accumulate_stats(acc: dict, name: str, arr: np.ndarray) -> None:
    """Collect raw rows per flat-vector name for later mean/std/min/max/q01/q99."""
    acc.setdefault(name, []).append(arr)


def _finalize_stats(acc: dict) -> dict:
    """GR00T statistics.json per flat vector (state, action): mean/std/min/max/q01/q99 over all rows."""
    out: dict = {}
    for name, parts in acc.items():
        allrows = np.concatenate(parts, axis=0)  # (sum_T, dim)
        out[name] = {
            "dim": int(allrows.shape[1]),
            "mean": allrows.mean(axis=0).tolist(),
            "std": allrows.std(axis=0).tolist(),
            "min": allrows.min(axis=0).tolist(),
            "max": allrows.max(axis=0).tolist(),
            "q01": np.quantile(allrows, 0.01, axis=0).tolist(),
            "q99": np.quantile(allrows, 0.99, axis=0).tolist(),
        }
    return out


def _short(k: str) -> str:
    """Registered GR00T modality sub-key = the LAST dotted segment (state.left_arm → left_arm,
    video.ego_view → ego_view). The N1.7 loader indexes modality.json by these SHORT names verbatim
    (verified vs NVIDIA/Isaac-GR00T@65cc4a192e6d lerobot_episode_loader.py:308-345; get_dataset_statistics
    at :518-534 does an UNGUARDED dict index → a prefixed key like `state.left_arm` raises KeyError at
    FT setup). So modality.json sub-keys MUST be short."""
    return k.split(".")[-1]


def convert(hdf5_path: str, repo_id: str, root: str, task: str, fallback_fps: int,
            video_keep: set[str] | None = None) -> None:
    # lerobot==0.1.0 (Isaac-GR00T WBC venv pins git a445d9c) lays the package out as
    # lerobot.common.datasets.* — the top-level `lerobot.datasets` / `src/` layouts are LATER.
    # Verified against the installed SHA (run#2 crashed on the wrong path). Deferred so --help works.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"{hdf5_path} has no top-level 'data' group — not a gr00t-g1 collect HDF5.")
        data = f["data"]
        demo_keys = _sorted_demo_keys(data)
        if not demo_keys:
            raise RuntimeError(f"{hdf5_path} has zero demo_* episodes — collection exported no successes (see G3).")
        print(f"[info] {len(demo_keys)} episodes in {hdf5_path}")

        fps = int(data.attrs.get("video_fps", fallback_fps))
        env_name = data.attrs.get("env_name", "")
        embodiment_tag = data.attrs.get("embodiment_tag", "")
        try:
            modality_cfg = json.loads(data.attrs.get("modality_config_json", "{}"))
        except (TypeError, json.JSONDecodeError):
            modality_cfg = {}
        print(f"[info] env={env_name} embodiment={embodiment_tag} fps={fps} "
              f"modality_cfg_keys={list(modality_cfg.keys())}")

        first = data[demo_keys[0]]
        if "obs" not in first or "action" not in first:
            raise RuntimeError(f"{demo_keys[0]} missing obs/ or action/ group.")

        # In the collector, state datasets live alongside video datasets under obs/. Separate them by
        # shape: video = 4-D (T,H,W,C); state = 2-D (T,d). Robust to either key prefixing.
        all_video_keys = [k for k in first["obs"].keys() if np.asarray(first["obs"][k]).ndim == 4]
        # ★ Key filter (2026-06-17 main-run fix): an HDF5 collected BEFORE the collector filter (the
        #   existing 138 GB run) also carries raw sim state (q/dq/floating_base_*/...) and extra cameras
        #   (video.tpp_view, ego_view_image, tpp_view_image). The N1.7 loader only consumes the
        #   REGISTERED UNITREE_G1 keys: state = the 7 `state.`-prefixed sub-keys; video = ego_view.
        #   Filter here too so this converter can re-convert an old over-collected HDF5 (no GPU needed)
        #   into a clean dataset. New HDF5s (post collector-fix) already contain only the kept keys, so
        #   the filter is a no-op there.
        #     • video: keep only keys in `video_keep` (default {"video.ego_view"}). Extra cameras dropped.
        #     • state: keep ONLY obs keys with the `state.` modality prefix (the 7 registered). Raw sim
        #       proprioception has no `state.` prefix → excluded. (If an HDF5 has NO `state.`-prefixed
        #       keys at all — unexpected — fall back to all non-video keys so we never emit an empty state.)
        keep_video = video_keep if video_keep is not None else {"video.ego_view"}
        video_keys = [k for k in all_video_keys if k in keep_video]
        if not video_keys:
            print(f"[warn] none of video_keep={sorted(keep_video)} present in HDF5 cameras "
                  f"{all_video_keys} — keeping all cameras as fallback", file=sys.stderr)
            video_keys = all_video_keys
        non_video = [k for k in first["obs"].keys() if k not in all_video_keys]
        registered_state = [k for k in non_video if k.startswith("state.")]
        if registered_state:
            state_keys = registered_state
        else:
            print(f"[warn] no `state.`-prefixed obs keys in HDF5 (keys={non_video}) — keeping all "
                  f"non-video obs as state (pre-prefix HDF5?)", file=sys.stderr)
            state_keys = non_video
        # Reorder state_keys by modality-config order where available; keep unknowns at the tail.
        ordered = _key_order(first["obs"], "state", modality_cfg)
        state_keys = [k for k in ordered if k in state_keys] + [k for k in state_keys if k not in ordered]
        action_keys = _key_order(first["action"], "action", modality_cfg)
        dropped_obs = [k for k in first["obs"].keys()
                       if k not in state_keys and k not in video_keys and not k.startswith("annotation")]
        if dropped_obs:
            print(f"[info] DROPPED {len(dropped_obs)} non-registered obs keys: {sorted(dropped_obs)}")

        # Size flat vectors + ranges from the first demo.
        _, state_ranges = _flatten_and_ranges(first, state_keys, "obs")
        _, action_ranges = _flatten_and_ranges(first, action_keys, "action")
        state_dim = state_ranges[state_keys[-1]][1] if state_keys else 0
        action_dim = action_ranges[action_keys[-1]][1] if action_keys else 0
        print(f"[info] state_dim={state_dim} ({state_keys}) | action_dim={action_dim} ({action_keys})")
        print(f"[info] video_keys={video_keys}")

        # LeRobot features: flat observation.state + flat action + one video per camera.
        features: dict = {
            "observation.state": {"dtype": "float32", "shape": (state_dim,),
                                  "names": {"axes": _expand_axis_names(state_keys, state_ranges)}},
            "action": {"dtype": "float32", "shape": (action_dim,),
                       "names": {"axes": _expand_axis_names(action_keys, action_ranges)}},
        }
        cam_shapes: dict[str, tuple] = {}
        for vk in video_keys:
            arr0 = np.asarray(first["obs"][vk])
            _, h, w, c = arr0.shape
            cam_name = vk.split(".")[-1]  # video.ego_view → ego_view
            cam_shapes[vk] = (h, w, c)
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channels"],
            }

        dataset = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, features=features, root=root,
            robot_type="unitree_g1_gear_wbc", use_videos=True,
        )

        stats_acc: dict = {}
        total_frames = 0
        for demo_key in demo_keys:
            demo = data[demo_key]
            num_samples = int(demo.attrs.get("num_samples", 0))
            states, _ = _flatten_and_ranges(demo, state_keys, "obs")
            actions, _ = _flatten_and_ranges(demo, action_keys, "action")
            videos = {vk: np.asarray(demo["obs"][vk]) for vk in video_keys}

            T = min(num_samples or len(actions), len(states), len(actions),
                    *(len(videos[v]) for v in video_keys)) if video_keys else min(len(states), len(actions))
            _accumulate_stats(stats_acc, "state", states[:T])
            _accumulate_stats(stats_acc, "action", actions[:T])

            for t in range(T):
                frame = {"observation.state": states[t], "action": actions[t], "task": task}
                for vk in video_keys:
                    cam_name = vk.split(".")[-1]
                    frame[f"observation.images.{cam_name}"] = videos[vk][t]  # (H,W,3) uint8
                dataset.add_frame(frame)
            dataset.save_episode()  # self-finalizing per episode at lerobot 0.1.0 (a445d9c):
            #   writes the parquet, encodes the mp4(s), computes per-episode stats, updates meta/info.json.
            total_frames += T
            print(f"[info] {demo_key}: {T} frames")

        # NOTE: lerobot 0.1.0 (a445d9c, the WBC-venv pin) has NO finalize()/consolidate() — save_episode()
        # already finalized each episode above. Calling finalize() here would AttributeError (the next
        # crash after the import bug). The v3.0 finalize() API does not exist at this SHA.

        # GR00T overlay files: meta/modality.json + statistics.json.
        import os
        meta_dir = os.path.join(root, "meta")
        os.makedirs(meta_dir, exist_ok=True)

        # ★ SHORT sub-keys (2026-06-17 fix): the N1.7 loader indexes modality.json's state/action groups
        #   by the REGISTERED short modality_keys (left_arm, ...) VERBATIM — NOT group+"."+key. A prefixed
        #   sub-key (`state.left_arm`) is therefore "not found": the data path silently drops the group,
        #   and get_dataset_statistics (unguarded dict index) raises KeyError at FT setup. Verified vs
        #   NVIDIA/Isaac-GR00T@65cc4a192e6d (lerobot_episode_loader.py:308-345, :518-534) + canonical
        #   demo_data/cube_to_bowl_5/meta/modality.json (single_arm/gripper short). Video was already short.
        modality_json = {
            "state": {_short(k): {"start": state_ranges[k][0], "end": state_ranges[k][1]} for k in state_keys},
            "action": {_short(k): {"start": action_ranges[k][0], "end": action_ranges[k][1]} for k in action_keys},
            "video": {_short(vk): {"original_key": f"observation.images.{_short(vk)}"}
                      for vk in video_keys},
            # H2 (vla-ft confirm 2026-06-16): registered UNITREE_G1 language modality_key is
            # `annotation.human.task_description` (NO `.action.`) — see embodiment_configs.py:115-192.
            # Must match exactly to use --embodiment-tag UNITREE_G1 without a custom modality config.
            "annotation": {"human.task_description": {}},
        }
        with open(os.path.join(meta_dir, "modality.json"), "w") as mf:
            json.dump(modality_json, mf, indent=2)

        statistics = {"state": _finalize_stats({"state": stats_acc["state"]})["state"],
                      "action": _finalize_stats({"action": stats_acc["action"]})["action"]}
        with open(os.path.join(root, "statistics.json"), "w") as sf:
            json.dump(statistics, sf, indent=2)

        # H1: GR00T N1.7 expects the LeRobot v2.x family. The training loader does not assert
        # codebase_version, but stamp meta/info.json to v2.1 (the producer contract) if the installed
        # lerobot wrote a newer tag. If it wrote v3.x, warn loudly — the producer lerobot should be a
        # v2.1-emitting build (vla-ft H1: "pin your converter's lerobot to a v2.1 emit version").
        info_path = os.path.join(meta_dir, "info.json")
        try:
            with open(info_path) as inf:
                info = json.load(inf)
            cv = str(info.get("codebase_version", ""))
            if cv.startswith("v3"):
                print(f"[warn] installed lerobot emitted codebase_version={cv} (v3) — GR00T N1.7 wants "
                      f"v2.1. Re-emitting tag to v2.1, but verify the on-disk layout matches v2.1 "
                      f"(chunked parquet/videos). Consider pinning a v2.1-emitting lerobot.",
                      file=sys.stderr)
            if cv != "v2.1":
                info["codebase_version"] = "v2.1"
                with open(info_path, "w") as inf:
                    json.dump(info, inf, indent=2)
                print(f"[info] stamped meta/info.json codebase_version v2.1 (was {cv or 'unset'})")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[warn] could not read/stamp meta/info.json ({e!r})", file=sys.stderr)

        print(f"[done] {len(demo_keys)} episodes / {total_frames} frames → {root}")
        print(f"[done] wrote meta/modality.json (flat groups, UNITREE_G1 keys, annotation="
              f"human.task_description) + statistics.json. LeRobot v2.1 contract (H1-H4 resolved).")


def _expand_axis_names(sub_keys: list[str], ranges: dict) -> list[str]:
    """Per-dim axis names like left_arm_0, left_arm_1, ... for the flat vector (cosmetic, aids debugging)."""
    names: list[str] = []
    for k in sub_keys:
        s, e = ranges[k]
        names.extend([f"{k.split('.')[-1]}_{i}" for i in range(e - s)])
    return names


def main() -> None:
    p = argparse.ArgumentParser(description="Convert gr00t-g1 collect HDF5 → GR00T LeRobotDataset.")
    p.add_argument("--hdf5", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--task", default="pick up the apple and place it on the plate")
    p.add_argument("--fps", type=int, default=20, help="Fallback fps if video_fps attr missing.")
    p.add_argument("--video-keys", default="video.ego_view",
                   help="Comma-separated obs video keys to KEEP (default: video.ego_view). Extra cameras "
                        "(video.tpp_view, *_image) are dropped. Lets this converter clean an old "
                        "over-collected HDF5 without a GPU re-run. Use the HDF5's full key names.")
    args = p.parse_args()
    video_keep = {k.strip() for k in args.video_keys.split(",") if k.strip()}
    convert(args.hdf5, args.repo_id, args.root, args.task, args.fps, video_keep=video_keep)


if __name__ == "__main__":
    main()
