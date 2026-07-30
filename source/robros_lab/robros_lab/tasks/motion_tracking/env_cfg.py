"""Free-base IGRIS-C motion tracking environment."""

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .action_adapter import MotionTrackingActionsCfg
from .mdp import ReferenceCommandCfg
from .observations import MotionTrackingObservationsCfg
from .scene_cfg import IGRISCMotionTrackingSceneCfg
from .student_contract import STUDENT_JOINT_NAMES


@configclass
class MotionTrackingCommandsCfg:
    """Replaceable motion reference consumed by the student observations."""

    motion: ReferenceCommandCfg = ReferenceCommandCfg(
        asset_name="robot",
        joint_names=list(STUDENT_JOINT_NAMES),
        body_names=[
            ".*base_link.*",
            ".*Hand.*",
            ".*Ankle_Roll.*",
            ".*Elbow.*",
            ".*Shoulder_Pitch.*",
            ".*Knee.*",
            ".*Hip_Pitch.*",
            ".*Neck_Pitch.*",
        ],
        anchor_body_name="base_link",
    )


@configclass
class MotionTrackingRewardsCfg:
    """No rewards are required for checkpoint inference."""

    pass


@configclass
class MotionTrackingTerminationsCfg:
    """No automatic terminations are required for continuous motion inference."""

    pass


@configclass
class IGRISCMotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """IGRIS-C environment for Student Policy motion tracking."""

    scene: IGRISCMotionTrackingSceneCfg = IGRISCMotionTrackingSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )
    observations: MotionTrackingObservationsCfg = MotionTrackingObservationsCfg()
    actions: MotionTrackingActionsCfg = MotionTrackingActionsCfg()
    commands: MotionTrackingCommandsCfg = MotionTrackingCommandsCfg()
    rewards: MotionTrackingRewardsCfg = MotionTrackingRewardsCfg()
    terminations: MotionTrackingTerminationsCfg = MotionTrackingTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.6, 0.0, 0.9)


class IGRISCMotionTrackingEnv(ManagerBasedRLEnv):
    """Registered motion tracking inference environment."""

    pass
