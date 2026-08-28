"""Checkpoint-compatible IGRIS-C environment for HugWBC inference."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg

from robros_lab.assets import IGRIS_C_WRIST_HAND_INDEPENDENT_CFG
from robros_lab.tasks.hug_wbc import mdp as hugwbc_mdp


LOWER_BODY_JOINTS = [
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Left",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Left",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Left",
    "Joint_Ankle_Roll_Right",
]
LOWER_BODY_SYMMETRY_JOINTS = [
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Left",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Left",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Left",
    "Joint_Ankle_Roll_Right",
]
LOWER_BODY_SYMMETRY_STATE_JOINTS = [
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Waist_Yaw",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Roll_Right",
    "Joint_Waist_Roll",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Waist_Pitch",
    "Joint_Elbow_Pitch_Left",
    "Joint_Elbow_Pitch_Right",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Left",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Left",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Left",
    "Joint_Ankle_Roll_Right",
]
TELEOPERATION_JOINTS = [
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
]
TELEOPERATION_TRACKED_JOINTS = [
    name
    for name in TELEOPERATION_JOINTS
    if name not in ("Joint_Waist_Roll", "Joint_Waist_Pitch")
]
LOWER_BODY_ACTION_OFFSET = {
    "Joint_Waist_Roll": 0.0,
    "Joint_Waist_Pitch": 0.0,
    "Joint_Hip_Pitch_Left": -0.2,
    "Joint_Hip_Pitch_Right": -0.2,
    "Joint_Hip_Roll_Left": 0.0,
    "Joint_Hip_Roll_Right": 0.0,
    "Joint_Hip_Yaw_Left": 0.0,
    "Joint_Hip_Yaw_Right": 0.0,
    "Joint_Knee_Pitch_Left": 0.3,
    "Joint_Knee_Pitch_Right": 0.3,
    "Joint_Ankle_Pitch_Left": -0.1,
    "Joint_Ankle_Pitch_Right": -0.1,
    "Joint_Ankle_Roll_Left": 0.0,
    "Joint_Ankle_Roll_Right": 0.0,
}
TELEOPERATION_RANGES = {
    "Joint_Waist_Yaw": (-1.5, 1.5),
    "Joint_Waist_Roll": (-0.2, 0.2),
    "Joint_Waist_Pitch": (-1.0, 0.28),
    "Joint_Shoulder_Pitch_Left": (-3.0, 1.0),
    "Joint_Shoulder_Roll_Left": (-0.15, 3.0),
    "Joint_Shoulder_Yaw_Left": (-1.5, 1.5),
    "Joint_Elbow_Pitch_Left": (-2.1, 0.0),
    "Joint_Shoulder_Pitch_Right": (-3.0, 1.0),
    "Joint_Shoulder_Roll_Right": (-3.0, 0.15),
    "Joint_Shoulder_Yaw_Right": (-1.5, 1.5),
    "Joint_Elbow_Pitch_Right": (-2.1, 0.0),
}


@configclass
class HugWBCSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = IGRIS_C_WRIST_HAND_INDEPENDENT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=IGRIS_C_WRIST_HAND_INDEPENDENT_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.98)
        ),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class LowerBodySymmetrySceneCfg(HugWBCSceneCfg):
    """Scene used by the lower-body symmetry checkpoint."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = IGRIS_C_WRIST_HAND_INDEPENDENT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        soft_joint_pos_limit_factor=0.97,
        init_state=IGRIS_C_WRIST_HAND_INDEPENDENT_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.98),
            joint_pos={
                **IGRIS_C_WRIST_HAND_INDEPENDENT_CFG.init_state.joint_pos,
                ".*_Ankle_Pitch_.*": -0.15,
            },
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = hugwbc_mdp.HugWBCBaseVelocityCommandCfg(
        command=(0.0, 0.0, 0.0),
        rel_standing_envs=0.0,
    )
    teleoperation_command = hugwbc_mdp.TeleoperationCommandCfg(
        asset_name="robot",
        joint_names=TELEOPERATION_JOINTS,
        ranges=TELEOPERATION_RANGES,
        interpolation_rate=1.5,
        resampling_time_range=(0.5, 3.5),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_BODY_JOINTS,
        scale=0.25,
        offset=LOWER_BODY_ACTION_OFFSET,
        preserve_order=True,
        use_default_offset=False,
    )
    gait_freq = hugwbc_mdp.GaitFreqActionCfg(asset_name="robot", lpf_alpha=0.3)
    teleop = hugwbc_mdp.CommandJointPositionActionCfg(
        asset_name="robot",
        joint_names=TELEOPERATION_TRACKED_JOINTS,
        command_name="teleoperation_command",
    )


@configclass
class LowerBodySymmetryActionsCfg:
    """Thirteen-action layout used by ``model_59400.pt``."""

    gait_freq = hugwbc_mdp.GaitFreqActionCfg(
        asset_name="robot",
        lpf_alpha=0.3,
        reset_processed_action=0.0,
    )
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_BODY_SYMMETRY_JOINTS,
        scale=0.25,
        preserve_order=True,
        use_default_offset=True,
    )
    teleop = hugwbc_mdp.CommandJointPositionActionCfg(
        asset_name="robot",
        joint_names=TELEOPERATION_TRACKED_JOINTS,
        command_name="teleoperation_command",
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        phase = ObsTerm(
            func=hugwbc_mdp.GaitSignalObservation,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=LOWER_BODY_JOINTS, preserve_order=True
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=LOWER_BODY_JOINTS, preserve_order=True
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        torso_command = ObsTerm(
            func=hugwbc_mdp.TeleoperationTorsoCommand,
            params={
                "command_name": "teleoperation_command",
                "joint_names": ("Joint_Waist_Roll", "Joint_Waist_Pitch"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LowerBodySymmetryObservationsCfg:
    """Actor observations from the source ``WalkingRobotPlayEnvCfg``."""

    @configclass
    class PolicyCfg(ObsGroup):
        gait_signal = ObsTerm(
            func=hugwbc_mdp.GaitSignalObservation,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=AdditiveGaussianNoiseCfg(std=0.05),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=AdditiveGaussianNoiseCfg(std=0.02),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LOWER_BODY_SYMMETRY_STATE_JOINTS,
                    preserve_order=True,
                )
            },
            noise=AdditiveGaussianNoiseCfg(std=0.02),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LOWER_BODY_SYMMETRY_STATE_JOINTS,
                    preserve_order=True,
                )
            },
            noise=AdditiveGaussianNoiseCfg(std=0.4),
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class IGRISCHugWBCFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: HugWBCSceneCfg = HugWBCSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 1.0e9
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class IGRISCHugWBCLowerBodySymmetryEnvCfg(IGRISCHugWBCFlatEnvCfg):
    """Playback port of ``WalkingRobotPlayEnvCfg`` for ``model_59400.pt``."""

    scene: LowerBodySymmetrySceneCfg = LowerBodySymmetrySceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: LowerBodySymmetryObservationsCfg = LowerBodySymmetryObservationsCfg()
    actions: LowerBodySymmetryActionsCfg = LowerBodySymmetryActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 1.0 / 250.0
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


__all__ = [
    "IGRISCHugWBCFlatEnvCfg",
    "IGRISCHugWBCLowerBodySymmetryEnvCfg",
    "LOWER_BODY_JOINTS",
    "LOWER_BODY_SYMMETRY_JOINTS",
    "LOWER_BODY_SYMMETRY_STATE_JOINTS",
    "TELEOPERATION_JOINTS",
    "TELEOPERATION_RANGES",
]
