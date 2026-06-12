# Copyright 2026 — vla-simulator openarm-lift-act target (Phase 1a overlay)
#
# OpenArm UNIMANUAL Lift-Cube × ACT (LeRobot) — Isaac Lab 2.3 env overlay.
#
# WHAT THIS IS
#   A drop-in overlay for the `enactic/openarm_isaac_lab` package. It registers NEW gym ids
#   derived from the stock `Isaac-Lift-Cube-OpenArm-v0` (`OpenArmCubeLiftEnvCfg`), changing
#   ONLY what an ACT (action-chunking transformer) imitation-learning loop needs:
#
#     1. ACTION   — replaces the stock joint-position arm action with an ABSOLUTE-pose
#                   Differential IK action (`DifferentialInverseKinematicsActionCfg`). This is
#                   REQUIRED so the scripted pick-lift state machine (`lift_cube_sm.py`), which
#                   commands EE poses, can drive this env to auto-generate success demos.
#                   The stock BinaryJointPositionActionCfg gripper is kept (inherited).
#     2. CAMERAS  — adds 2 RGB cameras (wrist + table) that ACT consumes as visual input.
#                   Stock env is state-only RL (zero cameras).
#     3. DICT OBS — exposes ONE `policy` obs group with `concatenate_terms=False`, holding BOTH
#                   the proprio state terms (arm_joint_pos, gripper_pos) AND the RGB image terms
#                   (wrist, table) side-by-side as a dict. Stock env is one flat `policy` tensor.
#
#                   WHY one group and not two (state group + camera group): the demo-collection
#                   path (Phase 1b) reuses Isaac Lab's stock RecorderManager. Its sole obs recorder
#                   term, `PreStepFlatPolicyObservationsRecorder`, captures ONLY `obs_buf["policy"]`
#                   — a separate `camera_obs` group would be silently dropped, yielding image-less
#                   demos (fatal for ACT). `record_demos.py` even hard-sets
#                   `observations.policy.concatenate_terms = False` so the policy group is a dict.
#                   This mirrors NVIDIA's reference visuomotor recording cfg
#                   (`stack_ik_rel_visuomotor_env_cfg.py`), which likewise packs state + cameras
#                   into the single `policy` group. Phase 1c concatenates the state terms into
#                   LeRobot `observation.state`; image terms map to `observation.images.<name>`.
#
#   It is NOT a new training reward. The parent's lift rewards/commands/terminations are left
#   INTACT and are actively USED:
#     - `commands.object_pose` (goal pose) is read by the state machine for the place target.
#     - lifting/goal-tracking rewards + object_dropping termination gate demo "success".
#
# WHY UNIMANUAL + IK (vs. the sibling bimanual π0.5 overlay `openarm_bi_vla_env_cfg.py`)
#   Fastest path to a TASK-SUCCESS video (score 70): reuse the only finished OpenArm PnP task
#   (unimanual Lift-Cube with success reward), collect demos with a teleop-free scripted SM,
#   train a lightweight ACT policy on them, then roll out. bimanual + language VLA is a later
#   upgrade (score 80) on a yet-to-be-built bimanual lift scene.
#
# HOW IT IS APPLIED (later phase, on GPU)
#   Copied into the cloned openarm_isaac_lab at:
#     source/openarm/openarm/tasks/manager_based/openarm_manipulation/unimanual/lift/config/
#   then imported once (its import triggers gym.register). Both the demo-collection SM and the
#   ACT eval rollout target `Isaac-Lift-Cube-OpenArm-ACT-v0` (or its -Play variant).
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# [확인 필요] — RUNTIME FACTS to verify on FIRST GPU deploy (cannot be known without Isaac Sim):
#
#   (L1) IK body_offset / TCP alignment. The scripted SM reads the `ee_frame` FrameTransformer
#        (target prim `openarm_ee_tcp`) as the EE pose, and commands desired EE poses back through
#        this IK action. For the SM loop to close, the IK-controlled point (body_name +
#        body_offset) MUST coincide with `openarm_ee_tcp`. `openarm_hand` is a known-valid body
#        (it is the parent's command/reward body_name); the offset from `openarm_hand` to
#        `openarm_ee_tcp` is a USD fact we do not have here. `_TCP_OFFSET` below is a first-pass
#        guess — verify against `robot.data.body_pos_w["openarm_hand"]` vs the ee_frame target,
#        and tune until the SM grasps reliably. (Franka's analogous panda_hand→TCP offset is
#        +0.107 m in z; OpenArm's hand geometry differs.)
#
#   (L2) Finger joint name/count. The regex `openarm_finger_joint.*` is inherited from the parent
#        for the gripper action (Binary = 1 dim regardless of finger count) and is reused here for
#        the proprio gripper-state term via `gripper_pos_single` (takes the FIRST matched finger →
#        1 dim). Robust to 1 or 2 mimic finger joints. Verify exact names via
#        `env.unwrapped.scene["robot"].joint_names`.
#
#   (L3) Wrist-camera mount prim path assumes the link prim is `{ENV_REGEX_NS}/Robot/openarm_hand`.
#        `openarm_hand` IS a valid body; if the USD nests it deeper the camera spawn fails →
#        adjust prim_path after inspecting the spawned prim tree.
#
#   (L4) Camera offsets/orientations are FIRST-PASS guesses framed for the lift workspace
#        (table @ [0.5,0,0], cube spawn @ [0.4,0,0.055]). Tune `OffsetCfg.pos/rot` after viewing
#        the first recorded frames — framing materially affects ACT success since vision carries
#        the object pose.
#
#   (L5) ACT proprio state == arm joint_pos (7) + 1 gripper value = 8-dim. The demo-collection
#        HDF5→LeRobot glue (Phase 1c) MUST record this SAME `policy` vector as observation.state
#        so train/eval distributions match.
# ─────────────────────────────────────────────────────────────────────────────────────────────

import os

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

# `mdp` here = unimanual/lift/mdp, which does `from isaaclab.envs.mdp import *`
# → provides image, joint_pos, JointPositionActionCfg, BinaryJointPositionActionCfg, etc.
from .. import mdp
from .joint_pos_env_cfg import OpenArmCubeLiftEnvCfg

# Stiffer-PD OpenArm config for IK tracking (mirrors Franka's FRANKA_PANDA_HIGH_PD_CFG switch).
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_unimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)


# ── Joint name lists (arm is known; gripper via regex — see L2) ─────────────────────────────────
_ARM = [f"openarm_joint{i}" for i in range(1, 8)]  # 7 arm joints, proprio/action indices 0–6
_FINGER = ["openarm_finger_joint.*"]               # gripper (regex robust to 1 or 2 finger joints)

# IK control point offset from `openarm_hand` toward `openarm_ee_tcp` (see L1).
# MEASURED off-GPU from the OpenArm unimanual USD (usdcat → ASCII, rest pose, root frame):
#   openarm_hand   translate.z = 0.6586
#   openarm_ee_tcp translate.z = 0.7516   → +0.093 m along the hand's local +z (approach) axis.
# Both prims carry identity orientation and are rigidly attached to the same wrist link, so this
# constant local offset holds at every arm pose. The stock SM commands `openarm_ee_tcp` to the cube
# (GRASP state: des_ee_pose = object_pose); without this offset the IK instead places `openarm_hand`
# (9.3 cm proximal) at the cube → the gripper closes 9.3 cm ABOVE it → zero grasps → zero demos.
# This was the demos=0 root cause on run_id 2026-06-08-0526 (NOT a recorder bug — recorder attached
# fine, 5 active terms; the cube was simply never reached). [확인 필요] confirm grasp on next GPU run.
_TCP_OFFSET = [0.0, 0.0, 0.093]

# Camera image resolution. ACT typically trains at 96–224 px; 224 matches common pretrained vision
# stacks and is resized internally if smaller is wanted. Square keeps aspect simple.
#
# OPENARM_CAM_RES env override: the SAME overlay serves two jobs — (a) ACT-training demo collection
# (224, cheap render, 16 envs) and (b) a high-resolution BEAUTY re-render for a shareable PnP video
# (e.g. 768). collect_demos_sm.py sets this env var from its --cam_res CLI arg BEFORE importing this
# module, so no source fork is needed. Higher res ⇒ more render VRAM per env, so a beauty run must
# drop num_envs (perfect square: 4/9) — see the yaml's video-mode block. Default stays 224 so any
# import without the env var (e.g. an ACT eval rollout) keeps the training-time resolution.
_CAM_RES = int(os.environ.get("OPENARM_CAM_RES", "224"))
_CAM_W = _CAM_RES
_CAM_H = _CAM_RES
_PINHOLE = sim_utils.PinholeCameraCfg(
    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
)


# ── Custom obs: single-finger gripper position (robust to 1 or 2 mimic finger joints, see L2) ──
def gripper_pos_single(env, asset_cfg: SceneEntityCfg):
    """Absolute joint position of the FIRST finger joint matched by ``asset_cfg`` → shape (B, 1).

    The OpenArm gripper may be modeled as 2 mimic finger joints; this collapses to a single
    representative value so the proprio state width is a stable 8 regardless of finger count.
    """
    asset = env.scene[asset_cfg.name]
    # asset_cfg.joint_ids is a list[int] resolved from the regex; take the first match only.
    return asset.data.joint_pos[:, asset_cfg.joint_ids][:, :1]


# ── Observations: ONE `policy` group (dict) holding proprio state + camera images ───────────────
@configclass
class ACTObservationsCfg:
    """Single `policy` group with `concatenate_terms=False` → returned as a dict of named terms:
    state terms {arm_joint_pos(7), gripper_pos(1)} + image terms {wrist, table}.

    Both kinds live in ONE group so the stock RecorderManager (which records only
    `obs_buf["policy"]`) captures state AND images in a single demo. Mirrors NVIDIA's
    `stack_ik_rel_visuomotor_env_cfg.py` recording cfg. Phase 1c concatenates the two state
    terms (in field order) into LeRobot `observation.state` (8-dim, see L5); each image term maps
    to `observation.images.<name>`."""

    @configclass
    class PolicyCfg(ObsGroup):
        """concatenate_terms=False → dict {term_name: tensor}. Field order below is the canonical
        order Phase 1c reads: state first (arm_joint_pos, gripper_pos), then images (wrist, table)."""

        # -- proprio state (concatenated by Phase 1c into observation.state) --
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,  # absolute joint position (state is absolute, not _rel)
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM, preserve_order=True)},
        )
        gripper_pos = ObsTerm(
            func=gripper_pos_single,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_FINGER)},
        )
        # -- camera images (each → observation.images.<name>) --
        wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        table = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # dict (mixes 8-dim state vectors with HxWx3 images)

    policy: PolicyCfg = PolicyCfg()


@configclass
class OpenArmCubeLiftACTEnvCfg(OpenArmCubeLiftEnvCfg):
    """ACT rollout/demo variant of the unimanual lift env: IK action + 2 cameras + dict obs."""

    observations: ACTObservationsCfg = ACTObservationsCfg()

    def __post_init__(self):
        # Parent sets OpenArm robot, cube object, ee_frame, joint_pos arm action + Binary gripper,
        # object_pose goal command, lift/goal rewards, object_dropping termination.
        super().__post_init__()

        # ── Switch to stiffer PD robot for accurate IK tracking ─────────────────────────────────
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Replace arm action with absolute-pose Differential IK (state machine drives EE pose) ─
        # Gripper action (BinaryJointPositionActionCfg) is INHERITED from the parent unchanged.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_joint.*"],
            body_name="openarm_hand",  # known-valid body (parent command/reward body_name); see L1
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=_TCP_OFFSET),  # L1
        )

        # ── Cameras (require AppLauncher --enable_cameras) ──────────────────────────────────────
        # wrist cam: mounted on the hand link, looking down the gripper approach axis.
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_hand/wrist_cam",  # [확인 필요] L3 prim nesting
            update_period=0.0,
            width=_CAM_W,
            height=_CAM_H,
            data_types=["rgb"],
            spawn=_PINHOLE,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="ros"  # L4
            ),
        )
        # table cam: world-frame third-person framing the table + cube workspace.
        self.scene.table_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            width=_CAM_W,
            height=_CAM_H,
            data_types=["rgb"],
            spawn=_PINHOLE,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"  # L4
            ),
        )

        # ── Camera rendering settings (match Franka visuomotor reference) ───────────────────────
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable DLSS for deterministic frames
        # Convenience list mirroring the Franka visuomotor convention (consumed by recorders).
        self.image_obs_list = ["table_cam", "wrist_cam"]

        # ── Success termination (the demo recorder gates HDF5 export on this) ───────────────────
        # The parent OpenArm lift env has ONLY time_out + object_dropping — no success signal.
        # The demo-collection driver (Phase 1b) follows record_demos.py: it pops
        # `terminations.success`, evaluates it each step, and exports the episode to HDF5 only
        # after the task has been continuously successful for N steps. `object_reached_goal` (the
        # cube reaching the goal command pose) is the canonical lift-task success criterion and is
        # exported by unimanual/lift/mdp.terminations. threshold=0.04 (vs the stock 0.02) is a
        # first-pass loosen so the scripted SM — whose LIFT target is the goal pose — registers
        # success reliably; [확인 필요] tighten after observing the first GPU run's success rate.
        self.terminations.success = DoneTerm(
            func=mdp.object_reached_goal,
            params={"command_name": "object_pose", "threshold": 0.04},
        )


@configclass
class OpenArmCubeLiftACTEnvCfg_PLAY(OpenArmCubeLiftACTEnvCfg):
    """Single-env play variant for ACT rollout / video recording."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ── gym registration (fires on import) ──────────────────────────────────────────────────────────
gym.register(
    id="Isaac-Lift-Cube-OpenArm-ACT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:OpenArmCubeLiftACTEnvCfg"},
)

gym.register(
    id="Isaac-Lift-Cube-OpenArm-ACT-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:OpenArmCubeLiftACTEnvCfg_PLAY"},
)
