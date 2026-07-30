from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.types import ArticulationActions

from .delayed_dc_motor import DelayedDCMotor

if TYPE_CHECKING:
    from .delayed_dc_motor_with_slack_cfg import DelayedDCMotorWithSlackCfg


class DelayedDCMotorWithSlack(DelayedDCMotor):
    """Explicit PD actuator with delayed commands and a feedback-side slack model."""

    cfg: DelayedDCMotorWithSlackCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedDCMotorWithSlackCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        slack_low, slack_high = self._resolve_slack_range(cfg.slack_range, self.joint_names, self._device)
        self._slack_low = slack_low
        self._slack_high = slack_high
        self._slack_half_gap = torch.zeros_like(self.computed_effort)

        self.joint_pos_feedback = torch.zeros_like(self.computed_effort)
        self.joint_vel_feedback = torch.zeros_like(self.computed_effort)
        self._joint_pos_feedback_initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._joint_vel_feedback_initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        self.reset(slice(None))

    def reset(self, env_ids):
        super().reset(env_ids)
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        self._slack_half_gap[env_ids_tensor] = self._sample_slack_half_gap(env_ids_tensor.numel())
        self.joint_pos_feedback[env_ids_tensor] = 0.0
        self.joint_vel_feedback[env_ids_tensor] = 0.0
        self._joint_pos_feedback_initialized[env_ids_tensor] = False
        self._joint_vel_feedback_initialized[env_ids_tensor] = False

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        delayed_positions, delayed_velocities, delayed_efforts = self._compute_delayed_commands(control_action)

        uninitialized_envs = ~self._joint_pos_feedback_initialized
        if torch.any(uninitialized_envs):
            self.joint_pos_feedback[uninitialized_envs] = joint_pos[uninitialized_envs]
            self.joint_vel_feedback[uninitialized_envs] = joint_vel[uninitialized_envs]
            self._joint_pos_feedback_initialized[uninitialized_envs] = True
            self._joint_vel_feedback_initialized[uninitialized_envs] = True

        inside_slack_band = torch.abs(delayed_positions - joint_pos) <= self._slack_half_gap
        self.joint_pos_feedback = torch.clamp(
            delayed_positions,
            min=joint_pos - self._slack_half_gap,
            max=joint_pos + self._slack_half_gap,
        )
        self.joint_vel_feedback = torch.where(inside_slack_band, delayed_velocities, joint_vel)

        error_pos = delayed_positions - self.joint_pos_feedback
        error_vel = delayed_velocities - self.joint_vel_feedback

        self._joint_vel[:] = joint_vel
        computed_effort = self.stiffness * error_pos + self.damping * error_vel + delayed_efforts
        self.computed_effort = torch.where(inside_slack_band, torch.zeros_like(computed_effort), computed_effort)
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _sample_slack_half_gap(self, num_envs: int) -> torch.Tensor:
        if torch.allclose(self._slack_low, self._slack_high):
            return self._slack_low.unsqueeze(0).expand(num_envs, -1)
        sample = torch.rand((num_envs, self.num_joints), device=self._device)
        return self._slack_low.unsqueeze(0) + sample * (self._slack_high - self._slack_low).unsqueeze(0)

    @staticmethod
    def _resolve_slack_range(
        slack_range: dict[str, tuple[float, float]],
        joint_names: list[str],
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(slack_range, dict):
            raise TypeError('slack_range must be a regex dictionary in the form {"regex": (min_value, max_value)}.')

        low = torch.zeros((len(joint_names),), device=device, dtype=torch.float)
        high = torch.zeros((len(joint_names),), device=device, dtype=torch.float)
        matched_patterns = [None] * len(joint_names)
        for joint_index, joint_name in enumerate(joint_names):
            for pattern, value in slack_range.items():
                if not re.fullmatch(pattern, joint_name):
                    continue
                if matched_patterns[joint_index] is not None:
                    raise ValueError(
                        f"Multiple slack_range patterns match joint '{joint_name}': "
                        f"'{matched_patterns[joint_index]}' and '{pattern}'."
                    )
                low_value, high_value = DelayedDCMotorWithSlack._validate_slack_bounds(*value)
                matched_patterns[joint_index] = pattern
                low[joint_index] = low_value
                high[joint_index] = high_value

        unmatched_joints = [
            joint_name
            for joint_name, matched_pattern in zip(joint_names, matched_patterns, strict=True)
            if matched_pattern is None
        ]
        if unmatched_joints:
            raise ValueError(f"slack_range did not match all actuator joints. Unmatched joints: {unmatched_joints}")
        return low, high

    @staticmethod
    def _validate_slack_bounds(low: float, high: float) -> tuple[float, float]:
        if low < 0.0 or high < 0.0 or high < low:
            raise ValueError(f"Slack range must satisfy 0.0 <= min_slack <= max_slack, got ({low}, {high}).")
        return float(low), float(high)
