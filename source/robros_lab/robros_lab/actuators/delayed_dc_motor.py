from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import IdealPDActuator
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .delayed_dc_motor_cfg import DelayedDCMotorCfg


class DelayedDCMotor(IdealPDActuator):
    """Explicit PD actuator with delayed commands and a rated/peak torque-speed envelope."""

    cfg: DelayedDCMotorCfg
    """The configuration for the actuator model."""

    def __init__(
        self,
        cfg: DelayedDCMotorCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        resolved_cfg = self._resolve_profile_cfg(cfg)
        super().__init__(
            resolved_cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            stiffness,
            damping,
            armature,
            friction,
            dynamic_friction,
            viscous_friction,
            effort_limit,
            velocity_limit,
        )

        self._validate_delay_range(cfg.min_delay, cfg.max_delay)

        self.torque_rated = self.effort_limit
        self.velocity_rated = self._parse_joint_parameter(cfg.velocity_rated, None)
        self.torque_limit = self._parse_joint_parameter(cfg.torque_limit, None)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._validate_profile_limits()

        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        DelayedDCMotor.reset(self, slice(None))

    def set_delay_steps(
        self, time_lags: int | torch.Tensor, env_ids: Sequence[int] | None = None, reset_buffers: bool = True
    ) -> None:
        """Set a shared actuator delay for the requested environments."""
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        if reset_buffers:
            self.positions_delay_buffer.reset(env_ids)
            self.velocities_delay_buffer.reset(env_ids)
            self.efforts_delay_buffer.reset(env_ids)

    def reset(self, env_ids: Sequence[int] | None):
        super().reset(env_ids)
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(env_ids_tensor.numel(),),
            dtype=torch.int,
            device=self._device,
        )
        self.set_delay_steps(time_lags, env_ids=env_ids_tensor, reset_buffers=True)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        delayed_positions, delayed_velocities, delayed_efforts = self._compute_delayed_commands(control_action)
        control_action.joint_positions = delayed_positions
        control_action.joint_velocities = delayed_velocities
        control_action.joint_efforts = delayed_efforts
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _compute_delayed_commands(
        self, control_action: ArticulationActions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.positions_delay_buffer.compute(control_action.joint_positions),
            self.velocities_delay_buffer.compute(control_action.joint_velocities),
            self.efforts_delay_buffer.compute(control_action.joint_efforts),
        )

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        joint_speed = torch.abs(self._joint_vel)
        motoring_limit = self._motoring_effort_limit(joint_speed)

        positive_limit = torch.where(self._joint_vel >= 0.0, motoring_limit, self.torque_limit)
        negative_limit = torch.where(self._joint_vel <= 0.0, motoring_limit, self.torque_limit)
        return torch.clip(effort, min=-negative_limit, max=positive_limit)

    def _motoring_effort_limit(self, joint_speed: torch.Tensor) -> torch.Tensor:
        # Below velocity_rated, overload torque tapers linearly from torque_limit to torque_rated.
        overload_limit = self.torque_limit - (self.torque_limit - self.torque_rated) * (
            joint_speed / self.velocity_rated
        )
        motoring_limit = torch.where(joint_speed < self.velocity_rated, overload_limit, self.torque_rated)
        motoring_limit = torch.where(
            joint_speed >= self.velocity_limit, torch.zeros_like(motoring_limit), motoring_limit
        )
        return torch.clamp(motoring_limit, min=torch.zeros_like(motoring_limit), max=self.torque_limit)

    def _validate_profile_limits(self) -> None:
        if not torch.all(torch.isfinite(self.velocity_limit)):
            raise ValueError("velocity_limit must be finite for all joints.")
        if torch.any(self.velocity_limit <= 0.0):
            raise ValueError("velocity_limit must be strictly positive for all joints.")
        if not torch.all(torch.isfinite(self.velocity_rated)):
            raise ValueError("velocity_rated must be finite for all joints.")
        if torch.any(self.velocity_rated <= 0.0):
            raise ValueError("velocity_rated must be strictly positive for all joints.")
        if torch.any(self.velocity_rated > self.velocity_limit):
            raise ValueError("velocity_rated must be less than or equal to velocity_limit for all joints.")
        if not torch.all(torch.isfinite(self.torque_rated)):
            raise ValueError("torque_rated must be finite for all joints.")
        if torch.any(self.torque_rated <= 0.0):
            raise ValueError("torque_rated must be strictly positive for all joints.")
        if not torch.all(torch.isfinite(self.torque_limit)):
            raise ValueError("torque_limit must be finite for all joints.")
        if torch.any(self.torque_limit < self.torque_rated):
            raise ValueError("torque_limit must be greater than or equal to torque_rated for all joints.")

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

    @classmethod
    def _resolve_profile_cfg(cls, cfg: DelayedDCMotorCfg) -> DelayedDCMotorCfg:
        effort_limit = cls._resolve_profile_alias(cfg.effort_limit, cfg.torque_rated, "effort_limit", "torque_rated")
        velocity_limit_sim = cfg.velocity_limit if cfg.velocity_limit_sim is None else cfg.velocity_limit_sim
        return replace(cfg, effort_limit=effort_limit, velocity_limit_sim=velocity_limit_sim)

    @staticmethod
    def _resolve_profile_alias(
        alias_value: float | dict[str, float] | None,
        canonical_value: float | dict[str, float],
        alias_name: str,
        canonical_name: str,
    ) -> float | dict[str, float]:
        if alias_value is None:
            return canonical_value
        if alias_value != canonical_value:
            raise ValueError(f"{alias_name} must match {canonical_name} when both are provided.")
        return alias_value

    @staticmethod
    def _validate_delay_range(min_delay: int, max_delay: int) -> None:
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError(f"Delay range must satisfy 0 <= min_delay <= max_delay, got ({min_delay}, {max_delay}).")
