"""Checkpoint-compatible actor observations for HugWBC."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg


class GaitSignalObservation(ManagerTermBase):
    """Return ``(frequency, sin(phase), cos(phase))`` in the training layout."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.global_gait_phase = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        del asset_cfg
        gait_freq = env.action_manager.get_term("gait_freq").processed_actions.squeeze(-1)
        self.global_gait_phase = torch.frac(self.global_gait_phase + self.dt * gait_freq)
        velocity_command = env.command_manager.get_command("base_velocity")
        small_velocity_command = torch.linalg.vector_norm(velocity_command, dim=-1) < 0.1
        self.global_gait_phase[small_velocity_command] = 0.125
        phase = 2.0 * torch.pi * self.global_gait_phase
        return torch.stack((gait_freq, torch.sin(phase), torch.cos(phase)), dim=-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.global_gait_phase[env_ids] = 0.0


class TeleoperationTorsoCommand(ManagerTermBase):
    """Select waist roll and pitch from the full teleoperation command."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_name = cfg.params["command_name"]
        self.joint_names = tuple(cfg.params["joint_names"])
        command_term = env.command_manager.get_term(self.command_name)
        self.command_ids = command_term.command_ids_for_joint_names(self.joint_names)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        joint_names: tuple[str, ...],
    ) -> torch.Tensor:
        del command_name, joint_names
        return env.command_manager.get_command(self.command_name)[:, self.command_ids]


__all__ = ["GaitSignalObservation", "TeleoperationTorsoCommand"]
