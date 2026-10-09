"""Free-base IGRIS-C motion tracking environment."""

import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .action_adapter import MotionTrackingActionsCfg
from .history_buffer import install_actuator_delay_buffers, install_student_history_buffers
from .mdp import ReferenceCommandCfg
from .observations import STUDENT_HISTORY_TERM_NAMES, MotionTrackingObservationsCfg
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

    def __init__(self, cfg: IGRISCMotionTrackingEnvCfg, **kwargs) -> None:
        super().__init__(cfg=cfg, **kwargs)
        install_student_history_buffers(self.observation_manager, STUDENT_HISTORY_TERM_NAMES)
        install_actuator_delay_buffers(self.scene["robot"])

    def step(self, action: torch.Tensor):
        """Advance an inference-only episode without checking empty done terms."""

        if (
            self.termination_manager.active_terms
            or self.reward_manager.active_terms
            or self.recorder_manager.active_terms
        ):
            return super().step(action)

        self.action_manager.process_action(action.to(self.device))
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reset_terminated.zero_()
        self.reset_time_outs.zero_()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
