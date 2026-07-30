"""Standing reference source for the fixed-neck 23-joint IGRIS-C."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply_inverse

from ..student_contract import STUDENT_JOINT_COUNT
from .base import (
    EnvIds,
    MotionReferenceFrame,
    validate_motion_reference,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class StandingReferenceSource:
    """Capture and hold one standing reference from an initialized robot.

    Call :meth:`initialize` after the environment has been reset and its
    articulation data has been updated. The source captures the current joint
    pose, anchor pose, and FK-derived body positions exactly once. Target
    velocities are set to zero.

    The captured tensors are never updated from the live robot state, so a
    falling robot does not move its own balance target. Calling
    :meth:`get_reference` only creates a lightweight dataclass with a fresh
    timestamp and reuses the cached tensors.
    """

    def __init__(
        self,
        *,
        robot: Articulation,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        anchor_body_name: str = "base_link",
    ) -> None:
        self._robot = robot
        self._joint_names = tuple(joint_names)
        self._body_names = tuple(body_names)
        self._anchor_body_name = str(anchor_body_name)

        self._validate_name_contract()

        self._device = torch.device(robot.device)
        self._num_envs = int(robot.num_instances)
        self._joint_indices = self._resolve_exact_indices(
            available_names=tuple(robot.joint_names),
            requested_names=self._joint_names,
            description="joint",
        )
        self._body_indices = self._resolve_exact_indices(
            available_names=tuple(robot.body_names),
            requested_names=self._body_names,
            description="body",
        )
        self._anchor_body_index = tuple(robot.body_names).index(self._anchor_body_name)

        self._frame: MotionReferenceFrame | None = None
        self._closed = False

    @property
    def is_initialized(self) -> bool:
        """Whether a standing frame has been captured."""

        return self._frame is not None and not self._closed

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Checkpoint-ordered reference joint names."""

        return self._joint_names

    @property
    def body_names(self) -> tuple[str, ...]:
        """Tracking-ordered reference body names."""

        return self._body_names

    def initialize(self) -> None:
        """Capture the current robot pose as the immutable standing reference."""

        if not self._robot.is_initialized:
            raise RuntimeError("The robot articulation must be initialized before capturing a standing reference.")

        joint_pos = self._robot.data.joint_pos.index_select(1, self._joint_indices).detach().clone()
        joint_vel = torch.zeros_like(joint_pos)

        anchor_pos_w = self._robot.data.body_pos_w[:, self._anchor_body_index].detach()
        anchor_quat_w = self._robot.data.body_quat_w[:, self._anchor_body_index].detach().clone()
        quaternion_norm = torch.linalg.vector_norm(anchor_quat_w, dim=-1, keepdim=True)
        if torch.any(quaternion_norm <= torch.finfo(anchor_quat_w.dtype).eps).item():
            raise ValueError("Cannot capture a standing reference with a zero-norm anchor quaternion.")
        anchor_quat_w.div_(quaternion_norm)

        body_pos_w = self._robot.data.body_pos_w.index_select(1, self._body_indices).detach()
        body_delta_w = body_pos_w - anchor_pos_w.unsqueeze(1)
        expanded_anchor_quat = anchor_quat_w.unsqueeze(1).expand(-1, len(self._body_names), -1)
        body_pos_b = quat_apply_inverse(expanded_anchor_quat, body_delta_w).clone()

        anchor_pos_z = anchor_pos_w[:, 2:3].clone()
        anchor_lin_vel_b = torch.zeros(
            (self._num_envs, 3),
            device=self._device,
            dtype=joint_pos.dtype,
        )
        anchor_ang_vel_b = torch.zeros_like(anchor_lin_vel_b)

        frame = MotionReferenceFrame(
            timestamp=0.0,
            joint_names=self._joint_names,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            anchor_pos_z=anchor_pos_z,
            anchor_quat_wxyz=anchor_quat_w,
            anchor_lin_vel_b=anchor_lin_vel_b,
            anchor_ang_vel_b=anchor_ang_vel_b,
            body_names=self._body_names,
            body_pos_b=body_pos_b,
            valid=True,
        )
        validate_motion_reference(
            frame,
            num_envs=self._num_envs,
            expected_joint_names=self._joint_names,
            expected_body_names=self._body_names,
            expected_device=self._device,
        )

        self._frame = frame
        self._closed = False

    def reset(self, env_ids: EnvIds = None) -> None:
        """Keep the immutable standing target across environment resets.

        The environment is expected to reset the robot to the same nominal
        standing state. Use :meth:`initialize` explicitly if a new standing
        pose should replace the captured target.
        """

        del env_ids

    def get_reference(self, timestamp: float) -> MotionReferenceFrame:
        """Return the cached standing frame with the requested timestamp."""

        if not self.is_initialized or self._frame is None:
            raise RuntimeError("StandingReferenceSource must be initialized before use.")
        if not isinstance(timestamp, (float, int)) or not math.isfinite(float(timestamp)):
            raise ValueError(f"timestamp must be a finite number, got {timestamp}.")

        frame = self._frame
        return MotionReferenceFrame(
            timestamp=float(timestamp),
            joint_names=frame.joint_names,
            joint_pos=frame.joint_pos,
            joint_vel=frame.joint_vel,
            anchor_pos_z=frame.anchor_pos_z,
            anchor_quat_wxyz=frame.anchor_quat_wxyz,
            anchor_lin_vel_b=frame.anchor_lin_vel_b,
            anchor_ang_vel_b=frame.anchor_ang_vel_b,
            body_names=frame.body_names,
            body_pos_b=frame.body_pos_b,
            valid=frame.valid,
        )

    def close(self) -> None:
        """Release the cached frame."""

        self._frame = None
        self._closed = True

    def _validate_name_contract(self) -> None:
        if len(self._joint_names) != STUDENT_JOINT_COUNT:
            raise ValueError(
                f"Standing reference requires {STUDENT_JOINT_COUNT} checkpoint joints, "
                f"got {len(self._joint_names)}."
            )
        self._validate_unique_nonempty_names(self._joint_names, "joint")
        self._validate_unique_nonempty_names(self._body_names, "body")

        if self._anchor_body_name not in self._body_names:
            raise ValueError(
                f"Anchor body {self._anchor_body_name!r} must be included in the tracked body names."
            )

    def _resolve_exact_indices(
        self,
        *,
        available_names: tuple[str, ...],
        requested_names: tuple[str, ...],
        description: str,
    ) -> torch.Tensor:
        index_by_name = {name: index for index, name in enumerate(available_names)}
        missing_names = [name for name in requested_names if name not in index_by_name]
        if missing_names:
            raise ValueError(
                f"Standing reference {description} names are missing from the robot: {missing_names}. "
                f"Available names: {available_names}."
            )

        return torch.tensor(
            [index_by_name[name] for name in requested_names],
            dtype=torch.long,
            device=self._device,
        )

    @staticmethod
    def _validate_unique_nonempty_names(names: tuple[str, ...], description: str) -> None:
        if not names:
            raise ValueError(f"Standing reference {description} names must not be empty.")
        if any(not name for name in names):
            raise ValueError(f"Standing reference {description} names contain an empty entry.")
        if len(set(names)) != len(names):
            raise ValueError(f"Standing reference {description} names contain duplicates.")
