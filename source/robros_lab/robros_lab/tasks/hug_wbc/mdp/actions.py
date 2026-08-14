"""Action terms required by the lower-body HugWBC policy."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class GaitFreqAction(ActionTerm):
    """Store the filtered gait frequency encoded by the policy's last action."""

    cfg: GaitFreqActionCfg

    def __init__(self, cfg: GaitFreqActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._processed_actions = torch.full(
            (self.num_envs, 1), cfg.reset_processed_action, device=self.device
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        frequency = torch.tanh(actions) * self.cfg.freq_scale + self.cfg.freq_offset
        self._processed_actions.mul_(self.cfg.lpf_alpha).add_(
            frequency, alpha=1.0 - self.cfg.lpf_alpha
        )

    def apply_actions(self) -> None:
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = self.cfg.reset_raw_action
        self._processed_actions[env_ids] = self.cfg.reset_processed_action


@configclass
class GaitFreqActionCfg(ActionTermCfg):
    """Configuration for :class:`GaitFreqAction`."""

    class_type: type[ActionTerm] = GaitFreqAction
    lpf_alpha: float = 0.3
    freq_scale: float = 0.4
    freq_offset: float = 1.0
    reset_raw_action: float = 0.0
    reset_processed_action: float = 1.0


class CommandJointPositionAction(ActionTerm):
    """Apply teleoperation targets without consuming policy action dimensions."""

    cfg: CommandJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: CommandJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        joint_ids, joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if tuple(joint_names) != tuple(cfg.joint_names):
            raise ValueError(
                "Teleoperation action joints did not resolve in the requested order. "
                f"Expected {tuple(cfg.joint_names)}, got {tuple(joint_names)}."
            )
        self._joint_ids = joint_ids
        self._joint_names = tuple(joint_names)
        command_term = env.command_manager.get_term(cfg.command_name)
        self._command_ids = command_term.command_ids_for_joint_names(self._joint_names)
        self._raw_actions = torch.empty((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros(
            (self.num_envs, len(self._joint_ids)), device=self.device
        )
        self.clamped_target_count = 0

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 0:
            raise ValueError(f"Expected a zero-width teleoperation action, got {tuple(actions.shape)}.")
        command = self._env.command_manager.get_command(self.cfg.command_name)
        self._processed_actions.copy_(command[:, self._command_ids])
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        clamped = torch.logical_or(
            self._processed_actions < limits[..., 0],
            self._processed_actions > limits[..., 1],
        )
        self.clamped_target_count += int(clamped.sum().item())
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = self._asset.data.joint_pos[env_ids][
            :, self._joint_ids
        ]


@configclass
class CommandJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`CommandJointPositionAction`."""

    class_type: type[ActionTerm] = CommandJointPositionAction
    joint_names: list[str] = []
    command_name: str = "teleoperation_command"


__all__ = [
    "CommandJointPositionAction",
    "CommandJointPositionActionCfg",
    "GaitFreqAction",
    "GaitFreqActionCfg",
]
