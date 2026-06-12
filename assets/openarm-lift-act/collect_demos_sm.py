#!/usr/bin/env python3
# Copyright 2026 — vla-simulator openarm-lift-act target (Phase 1b demo collector)
#
# Autonomous (teleop-free) scripted pick-and-lift demo collector for OpenArm × ACT.
#
# WHAT THIS IS
#   A merge of two stock Isaac Lab reference scripts, adapted to the OpenArm ACT env
#   (`Isaac-Lift-Cube-OpenArm-ACT-v0`, registered by openarm_uni_lift_act_env_cfg.py):
#
#     • scripts/environments/state_machine/lift_cube_sm.py  — the GPU `warp` pick-lift state
#       machine (REST→APPROACH_ABOVE→APPROACH→GRASP→LIFT). It is END-EFFECTOR-POSE driven and
#       fully ENV-AGNOSTIC, so the kernel + `PickAndLiftSm` class are copied here VERBATIM. It
#       replaces human teleoperation: no `Se3Keyboard`, no hands.
#
#     • scripts/tools/record_demos.py  — the HDF5 demonstration recorder (`RecorderManager` +
#       `ActionStateRecorderManagerCfg` + `DatasetExportMode.EXPORT_SUCCEEDED_ONLY`). We keep its
#       recorder wiring but DROP its teleop loop and its manual num_success_steps debounce.
#
# WHY THE RECORDING IS SIMPLER THAN record_demos.py
#   record_demos.py POPS `terminations.success` and zeroes `time_out` because a HUMAN drives the
#   arm and wants manual control over when an episode concludes (debounced over N success steps,
#   reset on a keypress). An autonomous SM needs none of that. We instead lean on Isaac Lab's
#   built-in TERMINATION-DRIVEN auto-recording (verified in manager_based_rl_env.py:211-229):
#
#       env.step() → termination_manager.compute() sets reset_buf
#                  → recorder_manager.record_post_step()              (records this step's obs/action)
#                  → for each terminated/timed-out env:
#                        recorder_manager.record_pre_reset(env_ids)   (auto-marks success from the
#                                                                       "success" termination term,
#                                                                       then auto-exports the episode)
#                        env._reset_idx(env_ids)                       (env recycles for the next demo)
#
#   So with `EXPORT_SUCCEEDED_ONLY`, a successful lift (the cube reaching the goal pose →
#   `terminations.success` fires) is written to HDF5, while a dropped cube or a 5 s time-out is
#   silently discarded and the env immediately retries. This is multi-env safe with NO hand-rolled
#   per-env success counters — every parallel env contributes demos independently.
#
#   PREREQUISITE: the overlay MUST define `terminations.success` (it does — `object_reached_goal`).
#   This script asserts it at startup; without it nothing is ever marked successful.
#
# OUTPUT
#   One HDF5 file (`--dataset_file`) containing `data/demo_<i>` groups. Each group holds, per the
#   stock ActionStateRecorderManagerCfg (recorders_cfg.py):
#       obs/<term>          — the `policy` obs group, term-by-term: arm_joint_pos(7), gripper_pos(1),
#                             wrist(H,W,3 uint8), table(H,W,3 uint8)   [concatenate_terms=False]
#       actions             — the 8-dim action fed to the env: [pos(3), quat_wxyz(4), gripper(1)]
#       processed_actions   — post-IK joint targets (not used by ACT; recorded for completeness)
#       initial_state/states— full scene state (for exact replay; not used by ACT)
#   Phase 1c (hdf5_to_lerobot.py) reads obs/<term> + actions and writes a LeRobotDataset.
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# [확인 필요] — RUNTIME FACTS to verify on the FIRST GPU run (cannot be known without Isaac Sim).
#   These determine whether the SM actually GRASPS — if the first run reports ~0 successes, fix
#   these before scaling N up (and burning GPU):
#
#   (C1) IK-vs-SM frame coincidence (the overlay's L1). The SM reads scene["ee_frame"]
#        (target prim `openarm_ee_tcp`) as the gripper point and commands desired poses, which the
#        IK action realises on `openarm_hand` + `body_offset=_TCP_OFFSET`. If `openarm_ee_tcp`
#        and `openarm_hand` differ by a real offset and `_TCP_OFFSET` is still [0,0,0], every grasp
#        misses by that offset → near-zero success. Tune `_TCP_OFFSET` in the overlay first.
#
#   (C2) Grasp ORIENTATION. `desired_orientation = (w,x,y,z)=(0,1,0,0)` (180° about x) makes the
#        Franka panda_hand point straight down. Whether it points OpenArm's `openarm_hand`
#        approach axis down depends on that link's frame convention. If grasps approach sideways,
#        change `_DOWN_QUAT` below.
#
#   (C3) position_threshold (0.01 m) and the success threshold (0.04 m in the overlay) trade
#        success-rate against demo precision. Loosen if the SM stalls in APPROACH; tighten if
#        "successes" look sloppy in the recorded video.
#
#   (C4) Cameras REQUIRE `--enable_cameras` (forced True below). Without it the image obs terms
#        render nothing and the HDF5 images are unusable.
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""Launch Omniverse app first (must precede all isaaclab imports)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Autonomous scripted pick-lift demo collector (OpenArm × ACT).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric / use USD I/O.")
parser.add_argument("--num_envs", type=int, default=16, help="Parallel envs (more = faster collection).")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-OpenArm-ACT-v0", help="Registered task id.")
parser.add_argument(
    "--register_module",
    type=str,
    default="openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.openarm_uni_lift_act_env_cfg",
    help="Dotted module path of the overlay whose import fires gym.register for --task. Imported before gym.make.",
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/openarm_lift_act.hdf5", help="Output HDF5 path."
)
parser.add_argument("--num_demos", type=int, default=100, help="Stop after this many SUCCESSFUL demos are exported.")
parser.add_argument("--position_threshold", type=float, default=0.01, help="SM EE position-reached threshold (m).")
parser.add_argument("--max_steps", type=int, default=200000, help="Hard cap on env steps (safety stop).")
parser.add_argument(
    "--cam_res",
    type=int,
    default=224,
    help="Square camera resolution (px). 224 = ACT-training default; raise (e.g. 768) for a "
    "high-resolution beauty re-render. Exported to OPENARM_CAM_RES, which the overlay reads "
    "when it builds the TiledCameras — so it MUST be set before the overlay is imported.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Image observation terms need RTX cameras rendered → enable_cameras is MANDATORY (C4).
args_cli.enable_cameras = True

# Publish the camera resolution to the env BEFORE the overlay module is imported (main() imports it
# via --register_module, and the overlay reads OPENARM_CAM_RES at import time to size its cameras).
# Set here at parse time so the value is in os.environ no matter where the import fires.
import os as _os  # noqa: E402  (local alias; os is re-imported below with the rest of the stdlib)

_os.environ["OPENARM_CAM_RES"] = str(args_cli.cam_res)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else."""

import gymnasium as gym
import importlib
import os
import torch
from collections.abc import Sequence

import warp as wp

import omni.log

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401  (registers the stock Isaac-* tasks)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Grasp approach orientation (w, x, y, z): 180° about x → gripper points down (see C2).
_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)

# Root-cause diagnostic cadence (Phase 1b debug). Two consecutive GPU runs exported 0 demos even
# though USD physics-layer reads confirm the TCP offset (ee_tcp_joint.localPos0=(0,0,0.093)) and the
# grasp orientation are geometrically correct, and the gripper actuator is wired (stiffness 2e3). So
# the failure is a RUNTIME behavior (does the DLS IK reach the cube within position_threshold? does
# the 5 s episode time out before the pick-lift sequence finishes?) that cannot be settled off-GPU.
# This block logs WHERE the SM stalls every _DIAG_EVERY steps so ONE run is conclusive. The cube is
# reset-randomized per env over x∈0.4±0.1, y∈0±0.25 (lift_env_cfg EventCfg.reset_object_position), so
# the min/mean ee↔object distance across envs also reveals whether the FAR cubes (y≈±0.25) are simply
# out of the OpenArm IK workspace — a reachability ceiling, distinct from a per-pose tracking failure.
_DIAG_EVERY = 100

wp.init()


# ════════════════════════════════════════════════════════════════════════════════════════════════
# State machine — copied VERBATIM from isaaclab scripts/environments/state_machine/lift_cube_sm.py.
# It is end-effector-pose driven and entirely env-agnostic; nothing OpenArm-specific lives here.
# ════════════════════════════════════════════════════════════════════════════════════════════════
class GripperState:
    """States for the gripper. Binary action: <0 → close, ≥0 → open (binary_joint_actions.py:137)."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """Task-space pick-and-lift state machine (warp kernel). See lift_cube_sm.py for full docs."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach-above offset: 0.1 m above the object, identity rotation (x,y,z,w order for warp)
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        # (w,x,y,z) → (x,y,z,w) for warp transforms
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # (x,y,z,w) → (w,x,y,z) back to Isaac Lab convention
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Collection driver — recorder wiring grafted from record_demos.py onto the autonomous SM loop.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # 1) Fire gym.register for the ACT task by importing the overlay module (relative imports inside
    #    it require package-qualified import, not a bare file import). Mirrors the bimanual userdata.
    try:
        importlib.import_module(args_cli.register_module)
        omni.log.info(f"Imported overlay module to register task: {args_cli.register_module}")
    except Exception as e:  # noqa: BLE001
        omni.log.error(f"Failed to import overlay module '{args_cli.register_module}': {e!r}")
        raise

    # 2) Output dir
    output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 3) Parse env cfg. Keep terminations.success ACTIVE (auto-export gate) and time_out ACTIVE
    #    (failed episodes terminate → env recycles instead of hanging).
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task

    if not hasattr(env_cfg.terminations, "success") or env_cfg.terminations.success is None:
        raise RuntimeError(
            "env_cfg.terminations.success is missing — the demo recorder cannot mark any episode "
            "successful, so EXPORT_SUCCEEDED_ONLY would write nothing. The openarm-lift-act overlay "
            "must define terminations.success (object_reached_goal)."
        )

    # 4) Attach the stock action+state recorder, success-only export.
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # 5) Build env + SM
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    if env.action_space.shape[-1] != 8:
        omni.log.warn(
            f"Expected 8-dim action [pos(3),quat(4),gripper(1)] but got {env.action_space.shape[-1]}. "
            "The SM emits 8 dims; mismatch means the IK/gripper action wiring differs from expectations."
        )

    # action buffer: identity-rotation EE pose + open gripper to start (quat w at index 3)
    actions = torch.zeros(env.action_space.shape, device=env.device)
    actions[:, 3] = 1.0

    # fixed downward grasp orientation, broadcast to all envs (see C2)
    desired_orientation = torch.zeros((env.num_envs, 4), device=env.device)
    down = torch.tensor(_DOWN_QUAT, device=env.device)
    desired_orientation[:] = down

    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.num_envs,
        env.device,
        position_threshold=args_cli.position_threshold,
    )

    # print → stdout so it lands in run.log + the 60 s S3 progress stream (see the [diag] note below
    # for why omni.log.info is invisible there).
    print(
        f"Collecting up to {args_cli.num_demos} successful demos with {env.num_envs} envs "
        f"(cam_res={args_cli.cam_res}px) -> {args_cli.dataset_file}",
        flush=True,
    )

    # State id → name, for the diagnostic histogram (mirrors PickSmState constants above).
    _STATE_NAMES = {0: "REST", 1: "APPROACH_ABOVE", 2: "APPROACH", 3: "GRASP", 4: "LIFT"}

    step_count = 0
    last_reported = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # step → env auto-records this step and auto-exports any episode that terminates
            obs, _, terminated, truncated, _ = env.step(actions)

            # -- end-effector frame (target prim openarm_ee_tcp), env-relative position --
            ee_frame_sensor = env.scene["ee_frame"]
            tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
            tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame --
            object_data: RigidObjectData = env.scene["object"].data
            object_position = object_data.root_pos_w - env.scene.env_origins
            # -- goal position (object_pose command, env-relative) --
            desired_position = env.command_manager.get_command("object_pose")[..., :3]

            # ── ROOT-CAUSE DIAGNOSTIC (Phase 1b) ─────────────────────────────────────────────────
            # Answers, in ONE run, the only questions left open after the off-GPU USD analysis:
            #   • Does the IK-driven ee_tcp actually REACH the cube?  → min/mean ee↔object distance.
            #     If this never falls near position_threshold(0.01), the DLS IK cannot reach the cube
            #     at the init pose → a reachability fix is needed (init pose / dls iters / action
            #     scale), NOT another offset tweak.
            #   • WHERE does the SM stall?  → histogram of SM states across envs. Stuck in APPROACH_*
            #     ⇒ IK never converges; reaching GRASP/LIFT but no export ⇒ grasp slips or the
            #     success threshold/episode timing is the gate.
            #   • How close to SUCCESS?  → min object↔goal distance vs the 0.04 success threshold.
            if step_count % _DIAG_EVERY == 0:
                ee_obj = torch.norm(tcp_position - object_position, dim=-1)
                obj_goal = torch.norm(object_position - desired_position, dim=-1)
                states = pick_sm.sm_state.tolist()
                hist = {name: states.count(sid) for sid, name in _STATE_NAMES.items()}
                hist_str = " ".join(f"{n}={c}" for n, c in hist.items() if c)
                # MUST be print(flush) — NOT omni.log.info. Kit routes INFO to the kit_*.log file
                # sink only, never to stdout, so run_id 2026-06-09-1410 ran the full 20000 steps but
                # the diagnostic never reached run.log / the S3 progress stream and was lost when the
                # instance terminated. print → stdout → captured in run.log AND streamed 60 s to
                # s3://.../RUN_ID/progress/collect.log, so the diagnostic survives termination and is
                # observable live. (omni.log.warn DOES reach stdout — that is why "Hit max_steps" was
                # the only line that survived.)
                print(
                    f"[diag step={step_count}] ee<->obj min={ee_obj.min().item():.4f} "
                    f"mean={ee_obj.mean().item():.4f} m | obj<->goal min={obj_goal.min().item():.4f} m "
                    f"(success<0.04) | SM states: {hist_str} | exported={env.recorder_manager.exported_successful_episode_count}",
                    flush=True,
                )

            # advance SM → next action
            actions = pick_sm.compute(
                torch.cat([tcp_position, tcp_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # reset SM state for any env that just terminated OR timed out (env itself already reset)
            dones = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if dones.numel() > 0:
                pick_sm.reset_idx(dones)

            # progress + stop conditions
            exported = env.recorder_manager.exported_successful_episode_count
            if exported != last_reported:
                print(f"Exported successful demos: {exported}/{args_cli.num_demos}", flush=True)
                last_reported = exported
            if exported >= args_cli.num_demos:
                print(f"Reached {exported} successful demos -- stopping.", flush=True)
                break

            step_count += 1
            if step_count >= args_cli.max_steps:
                omni.log.warn(
                    f"Hit max_steps={args_cli.max_steps} with only {exported} successful demos. "
                    "Check C1–C3 (TCP offset / grasp orientation / thresholds)."
                )
                break

            if env.sim.is_stopped():
                break

    # Read the export count BEFORE env.close() — close() does `del self.recorder_manager`
    # (manager_based_env.py:507), so touching env.recorder_manager afterward raises AttributeError.
    # (This is exactly what aborted the run_id 2026-06-08-0526 tail with a misleading traceback at
    # the old line 417; the recorder itself worked fine throughout the loop.)
    final_count = env.recorder_manager.exported_successful_episode_count
    env.close()
    print(f"Done. {final_count} demos -> {args_cli.dataset_file}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
