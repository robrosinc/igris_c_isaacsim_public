"""Student action configuration and runtime adaptation for IGRIS-C."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.utils import configclass
from .controllers.base import EnvIds
from .student_contract import STUDENT_JOINT_COUNT, STUDENT_JOINT_NAMES


@configclass
class MotionTrackingActionsCfg:
    """Checkpoint-compatible joint-position actions for the 23-joint robot."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(STUDENT_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=True,
        scale=0.25,
    )


@dataclass(frozen=True)
class AdaptedAction:
    """Result of converting one student output into an environment action."""

    final_action: torch.Tensor
    applied_student_action: torch.Tensor
    clipped: torch.Tensor
    finite: torch.Tensor


class StudentActionAdapter:
    """Validate and safely adapt 23-joint student actions.

    The current fixed-neck robot uses the same 23-joint order as the checkpoint,
    so no joint permutation is performed. Optional magnitude, rate, and
    acceleration limits operate in normalized policy-action units. They default
    to disabled to preserve the checkpoint's original inference semantics.

    Non-finite actions are replaced per environment with the previous applied
    action. The hot path uses tensor operations only and does not synchronize
    CUDA tensors back to the CPU.
    """

    def __init__(
        self,
        *,
        student_joint_names: Sequence[str],
        robot_joint_names: Sequence[str],
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        action_limit: float | Sequence[float] | torch.Tensor | None = None,
        max_action_rate: float | Sequence[float] | torch.Tensor | None = None,
        max_action_acceleration: float | Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        student_names = tuple(student_joint_names)
        robot_names = tuple(robot_joint_names)
        self._validate_joint_contract(student_names, robot_names)

        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be floating-point, got {dtype}.")

        self.student_joint_names = student_names
        self.robot_joint_names = robot_names
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype

        self._action_limit = self._resolve_limit(action_limit, "action_limit")
        self._max_action_rate = self._resolve_limit(max_action_rate, "max_action_rate")
        self._max_action_acceleration = self._resolve_limit(
            max_action_acceleration,
            "max_action_acceleration",
        )

        state_shape = (self.num_envs, STUDENT_JOINT_COUNT)
        self._previous_action = torch.zeros(state_shape, device=self.device, dtype=self.dtype)
        self._previous_action_rate = torch.zeros_like(self._previous_action)

    def reset(self, env_ids: EnvIds = None) -> None:
        """Reset action and rate history for all or selected environments."""

        if env_ids is None:
            self._previous_action.zero_()
            self._previous_action_rate.zero_()
            return

        self._previous_action[env_ids] = 0.0
        self._previous_action_rate[env_ids] = 0.0

    def adapt(self, student_action: torch.Tensor, *, dt: float) -> AdaptedAction:
        """Return the safe action to pass to the Isaac Lab environment."""

        self._validate_runtime_input(student_action, dt)

        finite = torch.isfinite(student_action).all(dim=-1)
        finite_expanded = finite.unsqueeze(-1)
        candidate = torch.where(finite_expanded, student_action, self._previous_action)

        if self._action_limit is not None:
            candidate = torch.clamp(candidate, min=-self._action_limit, max=self._action_limit)

        desired_rate = (candidate - self._previous_action) / dt

        if self._max_action_rate is not None:
            desired_rate = torch.clamp(
                desired_rate,
                min=-self._max_action_rate,
                max=self._max_action_rate,
            )

        if self._max_action_acceleration is not None:
            maximum_rate_delta = self._max_action_acceleration * dt
            desired_rate = torch.clamp(
                desired_rate,
                min=self._previous_action_rate - maximum_rate_delta,
                max=self._previous_action_rate + maximum_rate_delta,
            )

        if self._max_action_rate is not None:
            desired_rate = torch.clamp(
                desired_rate,
                min=-self._max_action_rate,
                max=self._max_action_rate,
            )

        applied_action = self._previous_action + desired_rate * dt

        if self._action_limit is not None:
            applied_action = torch.clamp(
                applied_action,
                min=-self._action_limit,
                max=self._action_limit,
            )

        applied_action = torch.where(finite_expanded, applied_action, self._previous_action)
        applied_rate = (applied_action - self._previous_action) / dt
        clipped = torch.logical_or(
            ~finite_expanded,
            applied_action != student_action,
        )

        self._previous_action.copy_(applied_action)
        self._previous_action_rate.copy_(applied_rate)

        return AdaptedAction(
            final_action=applied_action,
            applied_student_action=applied_action,
            clipped=clipped,
            finite=finite,
        )

    def _resolve_limit(
        self,
        value: float | Sequence[float] | torch.Tensor | None,
        name: str,
    ) -> torch.Tensor | None:
        if value is None:
            return None

        limit = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if limit.ndim == 0:
            limit = limit.expand(STUDENT_JOINT_COUNT)
        elif tuple(limit.shape) != (STUDENT_JOINT_COUNT,):
            raise ValueError(
                f"{name} must be a scalar or have shape ({STUDENT_JOINT_COUNT},), "
                f"got {tuple(limit.shape)}."
            )

        if not torch.isfinite(limit).all().item():
            raise ValueError(f"{name} must contain only finite values.")
        if not torch.all(limit > 0.0).item():
            raise ValueError(f"{name} values must be positive.")

        return limit.unsqueeze(0)

    def _validate_runtime_input(self, student_action: torch.Tensor, dt: float) -> None:
        if not isinstance(student_action, torch.Tensor):
            raise TypeError(
                f"student_action must be a torch.Tensor, got {type(student_action).__name__}."
            )
        expected_shape = (self.num_envs, STUDENT_JOINT_COUNT)
        if tuple(student_action.shape) != expected_shape:
            raise ValueError(
                f"student_action must have shape {expected_shape}, "
                f"got {tuple(student_action.shape)}."
            )
        if student_action.device != self.device:
            raise ValueError(
                f"student_action is on {student_action.device}, expected {self.device}."
            )
        if student_action.dtype != self.dtype:
            raise TypeError(
                f"student_action has dtype {student_action.dtype}, expected {self.dtype}."
            )
        if not isinstance(dt, (float, int)) or not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError(f"dt must be a positive finite number, got {dt}.")

    @staticmethod
    def _validate_joint_contract(
        student_joint_names: tuple[str, ...],
        robot_joint_names: tuple[str, ...],
    ) -> None:
        if len(student_joint_names) != STUDENT_JOINT_COUNT:
            raise ValueError(
                f"The checkpoint requires {STUDENT_JOINT_COUNT} student joints, "
                f"got {len(student_joint_names)}."
            )
        if len(robot_joint_names) != STUDENT_JOINT_COUNT:
            raise ValueError(
                f"The fixed-neck robot must expose {STUDENT_JOINT_COUNT} action joints, "
                f"got {len(robot_joint_names)}."
            )
        if len(set(student_joint_names)) != STUDENT_JOINT_COUNT:
            raise ValueError("student_joint_names contains empty or duplicate names.")
        if len(set(robot_joint_names)) != STUDENT_JOINT_COUNT:
            raise ValueError("robot_joint_names contains empty or duplicate names.")
        if any(not name for name in student_joint_names):
            raise ValueError("student_joint_names contains an empty name.")
        if any(not name for name in robot_joint_names):
            raise ValueError("robot_joint_names contains an empty name.")
        if student_joint_names != robot_joint_names:
            raise ValueError(
                "Student and robot joint names/order must match exactly. "
                f"Student: {student_joint_names}; robot: {robot_joint_names}."
            )
