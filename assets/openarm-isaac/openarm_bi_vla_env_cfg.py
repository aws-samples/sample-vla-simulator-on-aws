# Copyright 2026 — vla-simulator openarm-isaac target (Phase 2a overlay)
#
# OpenArm bimanual × π0.5 (LeRobot pi05 `folding_latest`) — Isaac Lab 2.3 env overlay.
#
# WHAT THIS IS
#   A drop-in overlay for the `enactic/openarm_isaac_lab` package. It registers a NEW gym id
#   `Isaac-Reach-OpenArm-Bi-VLA-v0` derived from the stock `Isaac-Reach-OpenArm-Bi-v0`
#   (`OpenArmReachEnvCfg`), changing ONLY what a Vision-Language-Action rollout needs:
#
#     1. CAMERAS  — adds 3 RGB cameras (base + left_wrist + right_wrist) the policy expects.
#                   Stock env is state-only RL (zero cameras).
#     2. DICT OBS — exposes obs as two groups named `policy` and `camera_obs`, matching the
#                   contract LeRobot's `preprocess_observation` Isaac Lab branch +
#                   `IsaaclabArenaProcessorStep` read. (Stock env is one flat `policy` tensor.)
#     3. ACTION   — replaces the 14-dim *relative* arm-only action with a 16-dim *absolute*
#                   joint-position action laid out [R_arm(7), R_grip(1), L_arm(7), L_grip(1)]
#                   to match folding_latest's action vector. (Stock env: scale=0.5,
#                   use_default_offset=True, no grippers.)
#
#   It is NOT a training env. Rewards/commands/curriculum from the parent are left intact but
#   unused — the VLA policy is driven open-loop via `lerobot-eval --env.type=isaaclab_arena`.
#
# HOW IT IS APPLIED (Phase 2c)
#   Copied into the cloned openarm_isaac_lab at:
#     source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/config/
#   then imported once (its import triggers gym.register). The userdata does:
#     python -c "import ...bimanual.reach.config.openarm_bi_vla_env_cfg"  (before lerobot-eval)
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# [확인 필요] — RUNTIME FACTS to verify on FIRST GPU deploy (cannot be known without Isaac Sim):
#
#   (G1) Gripper finger joints: is it 1 or 2 mimic joints per side, and the exact name
#        (`openarm_left_finger_joint1`? `..._joint`?). The regex `openarm_*_finger_joint.*` is
#        used everywhere here so it is ROBUST to either count for ACTION (BinaryJoint = 1 dim
#        regardless) and for STATE (gripper_pos_single takes the FIRST matched finger only → 1 dim).
#        Verify: `env.unwrapped.scene["robot"].joint_names`.
#
#   (G2) Wrist-camera mount prim path: assumes link prims live at
#        `{ENV_REGEX_NS}/Robot/openarm_<side>_hand/...`. `openarm_<side>_hand` IS a valid body
#        (it is the reward/command body_name in the parent), but the USD may nest it deeper.
#        Verify by inspecting the spawned prim tree; adjust prim_path if the camera fails to spawn.
#
#   (G3) Camera offsets/orientations are FIRST-PASS guesses. For a pipe-proof (motion demo) exact
#        framing does not matter — the real-robot-trained policy sees OOD sim images regardless.
#        Tune `OffsetCfg.pos/rot` after viewing the first RecordVideo output.
#
#   (G4) ActionManager concatenation order == config field-definition order. This is the
#        documented Isaac Lab behavior and the linchpin of the 16-dim layout below. If a future
#        Isaac Lab version reorders, the action mapping breaks — assert action_dim==16 at runtime.
# ─────────────────────────────────────────────────────────────────────────────────────────────

from dataclasses import MISSING

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

# `mdp` here = bimanual/reach/mdp, which does `from isaaclab.envs.mdp import *`
# → provides image, joint_pos, JointPositionActionCfg, BinaryJointPositionActionCfg.
from .. import mdp
from .joint_pos_env_cfg import OpenArmReachEnvCfg


# ── Joint name lists (arms are known; grippers via regex — see G1) ──────────────────────────────
_RIGHT_ARM = [f"openarm_right_joint{i}" for i in range(1, 8)]  # 7 joints, policy indices 0–6
_LEFT_ARM = [f"openarm_left_joint{i}" for i in range(1, 8)]   # 7 joints, policy indices 8–14
_RIGHT_FINGER = ["openarm_right_finger_joint.*"]               # policy index 7  (gripper)
_LEFT_FINGER = ["openarm_left_finger_joint.*"]                 # policy index 15 (gripper)


# ── Custom obs: single-finger gripper position (robust to 1 or 2 mimic finger joints, see G1) ──
def gripper_pos_single(env, asset_cfg: SceneEntityCfg):
    """Absolute joint position of the FIRST finger joint matched by ``asset_cfg`` → shape (B, 1).

    folding_latest's state expects ONE gripper value per side. The OpenArm gripper may be modeled
    as 2 mimic finger joints; this collapses to a single representative value so the 16-dim state
    layout holds regardless of finger-joint count.
    """
    asset = env.scene[asset_cfg.name]
    # asset_cfg.joint_ids is a list[int] resolved from the regex; take the first match only.
    return asset.data.joint_pos[:, asset_cfg.joint_ids][:, :1]


# ── Observations: two dict groups matching the IsaaclabArena contract ──────────────────────────
@configclass
class VLAObservationsCfg:
    """`policy` (16-dim state, as a dict) + `camera_obs` (3 RGB images, as a dict)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """State group. concatenate_terms=False → returned as dict {term_name: tensor}.

        LeRobot's IsaaclabArenaProcessorStep concatenates these by `state_keys` in the order:
            right_arm_pos, right_gripper_pos, left_arm_pos, left_gripper_pos   (= 7+1+7+1 = 16)
        Set on the lerobot side via: --env.state_keys="right_arm_pos,right_gripper_pos,left_arm_pos,left_gripper_pos"
        """

        right_arm_pos = ObsTerm(
            func=mdp.joint_pos,  # absolute joint position (NOT joint_pos_rel — policy state is absolute)
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_ARM, preserve_order=True)},
        )
        right_gripper_pos = ObsTerm(
            func=gripper_pos_single,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_FINGER)},
        )
        left_arm_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEFT_ARM, preserve_order=True)},
        )
        left_gripper_pos = ObsTerm(
            func=gripper_pos_single,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEFT_FINGER)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # dict, not flat tensor

    @configclass
    class CameraObsCfg(ObsGroup):
        """Camera group. Term names (base/left_wrist/right_wrist) → observation.images.<name>
        after IsaaclabArenaProcessorStep. MUST match folding_latest config.image_features."""

        base = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("base_cam"), "data_type": "rgb", "normalize": False},
        )
        left_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # dict {cam_name: uint8 (B,H,W,C)}

    policy: PolicyCfg = PolicyCfg()
    camera_obs: CameraObsCfg = CameraObsCfg()


# ── Actions: 16-dim absolute, field order == policy layout [R_arm, R_grip, L_arm, L_grip] (G4) ──
@configclass
class VLAActionsCfg:
    """Field definition order IS the action-vector concatenation order. Do not reorder."""

    right_arm_action: ActionTerm = MISSING      # dims 0–6
    right_gripper_action: ActionTerm = MISSING  # dim 7
    left_arm_action: ActionTerm = MISSING       # dims 8–14
    left_gripper_action: ActionTerm = MISSING   # dim 15


# ── Camera image resolution (square; policy resizes-with-pad to 224 internally) ─────────────────
_CAM_W = 224
_CAM_H = 224
_PINHOLE = sim_utils.PinholeCameraCfg(
    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
)


@configclass
class OpenArmBiVLAReachEnvCfg(OpenArmReachEnvCfg):
    """VLA rollout variant of the bimanual reach env: cameras + dict obs + 16-dim absolute action."""

    observations: VLAObservationsCfg = VLAObservationsCfg()
    actions: VLAActionsCfg = VLAActionsCfg()

    def __post_init__(self):
        # Parent sets up the robot (OPEN_ARM_HIGH_PD_CFG), sim dt/decimation, rewards, commands.
        super().__post_init__()

        # ── Cameras (require AppLauncher --enable_cameras) ──────────────────────────────────────
        # base: world-frame third-person looking at the workspace (arms reach toward +x).
        self.scene.base_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/base_cam",
            update_period=0.0,
            width=_CAM_W,
            height=_CAM_H,
            data_types=["rgb"],
            spawn=_PINHOLE,
            # [확인 필요] G3 — first-pass framing: ~0.8m in front, 0.6m high, tilted down toward origin.
            offset=TiledCameraCfg.OffsetCfg(pos=(0.9, 0.0, 0.6), rot=(0.9239, 0.0, 0.3827, 0.0), convention="world"),
        )
        # wrist cams: mounted on each hand link, looking down the gripper approach axis.
        self.scene.left_wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand/left_wrist_cam",  # [확인 필요] G2 prim nesting
            update_period=0.0,
            width=_CAM_W,
            height=_CAM_H,
            data_types=["rgb"],
            spawn=_PINHOLE,
            offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="ros"),
        )
        self.scene.right_wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand/right_wrist_cam",  # [확인 필요] G2
            update_period=0.0,
            width=_CAM_W,
            height=_CAM_H,
            data_types=["rgb"],
            spawn=_PINHOLE,
            offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="ros"),
        )

        # ── 16-dim absolute joint-position action ──────────────────────────────────────────────
        # Arms: scale=1.0, use_default_offset=False → target = action (absolute). preserve_order
        # forces the 7-vector to follow _RIGHT_ARM/_LEFT_ARM order (not Isaac Lab enumeration).
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=_RIGHT_ARM, scale=1.0, use_default_offset=False, preserve_order=True
        )
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=_LEFT_ARM, scale=1.0, use_default_offset=False, preserve_order=True
        )
        # Grippers: Binary action consumes exactly 1 dim regardless of finger-joint count (G1).
        # The policy's continuous gripper value is thresholded open/close — acceptable for pipe-proof.
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=_RIGHT_FINGER,
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=_LEFT_FINGER,
            open_command_expr={"openarm_left_finger_joint.*": 0.044},
            close_command_expr={"openarm_left_finger_joint.*": 0.0},
        )


@configclass
class OpenArmBiVLAReachEnvCfg_PLAY(OpenArmBiVLAReachEnvCfg):
    """Single-env play variant for VLA rollout (num_envs=1 avoids pi05 shared-chunk lockstep)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ── gym registration (fires on import) ──────────────────────────────────────────────────────────
gym.register(
    id="Isaac-Reach-OpenArm-Bi-VLA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:OpenArmBiVLAReachEnvCfg"},
)

gym.register(
    id="Isaac-Reach-OpenArm-Bi-VLA-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:OpenArmBiVLAReachEnvCfg_PLAY"},
)
