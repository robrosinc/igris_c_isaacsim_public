"""Shared data contracts for motion-reference controller backends.

The student controller consumes the same reference shape regardless of whether
the reference originates from a standing pose, GMR, MPC, or an offline replay.
This module intentionally contains no Isaac Lab, RSL-RL, or device-specific
control logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ..student_contract import STUDENT_JOINT_COUNT

EnvIds = Sequence[int] | torch.Tensor | None


@dataclass(frozen=True)
class MotionReferenceFrame:
    """One current motion-reference frame consumed by the student policy.

    All tensors include an environment batch dimension. The joint names must
    use the checkpoint's 23-joint order, while the body names must use the
    tracking observation's resolved body order.

    Tensor shapes:

    * joint_pos and joint_vel: (num_envs, 23)
    * anchor_pos_z: (num_envs, 1)
    * anchor_quat_wxyz: (num_envs, 4)
    * anchor_lin_vel_b and anchor_ang_vel_b: (num_envs, 3)
    * body_pos_b: (num_envs, num_bodies, 3)

    A frozen dataclass prevents field reassignment, but does not make tensors
    immutable. A reference source must not mutate a frame after publishing it.
    """

    timestamp: float
    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    anchor_pos_z: torch.Tensor
    anchor_quat_wxyz: torch.Tensor
    anchor_lin_vel_b: torch.Tensor
    anchor_ang_vel_b: torch.Tensor
    body_names: tuple[str, ...]
    body_pos_b: torch.Tensor
    valid: bool = True


@runtime_checkable
class MotionReferenceSource(Protocol):
    """Common lifecycle for standing, GMR, MPC, and replay references.

    Concrete sources receive their dependencies when constructed. Calling
    get_reference must be non-blocking; asynchronous sources should return
    their latest complete frame instead of waiting for new input.
    """

    def initialize(self) -> None:
        """Initialize resources and validate the source configuration."""
        ...

    def reset(self, env_ids: EnvIds = None) -> None:
        """Reset state for all or selected environments."""
        ...

    def get_reference(self, timestamp: float) -> MotionReferenceFrame:
        """Return the current reference frame without blocking."""
        ...

    def close(self) -> None:
        """Release resources owned by the source."""
        ...


def validate_motion_reference(
    reference: MotionReferenceFrame,
    *,
    num_envs: int,
    expected_joint_names: Sequence[str],
    expected_body_names: Sequence[str],
    expected_device: torch.device | str | None = None,
    quaternion_atol: float = 1.0e-3,
) -> None:
    """Validate a reference frame against the student checkpoint contract."""

    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")

    joint_names = tuple(expected_joint_names)
    body_names = tuple(expected_body_names)

    if len(joint_names) != STUDENT_JOINT_COUNT:
        raise ValueError(
            f"The student checkpoint expects {STUDENT_JOINT_COUNT} joints, "
            f"but received {len(joint_names)} expected joint names."
        )
    _validate_unique_names(joint_names, "expected joint")
    _validate_unique_names(body_names, "expected body")

    if tuple(reference.joint_names) != joint_names:
        raise ValueError(
            "Reference joint names/order do not match the student checkpoint: "
            f"expected {joint_names}, got {tuple(reference.joint_names)}."
        )
    if tuple(reference.body_names) != body_names:
        raise ValueError(
            "Reference body names/order do not match the tracking observation: "
            f"expected {body_names}, got {tuple(reference.body_names)}."
        )
    if not reference.valid:
        raise ValueError("Reference frame is marked invalid.")
    if not isinstance(reference.timestamp, (float, int)):
        raise TypeError(f"Reference timestamp must be numeric, got {type(reference.timestamp).__name__}.")
    if not torch.isfinite(torch.tensor(float(reference.timestamp))):
        raise ValueError(f"Reference timestamp must be finite, got {reference.timestamp}.")

    expected_shapes = {
        "joint_pos": (num_envs, STUDENT_JOINT_COUNT),
        "joint_vel": (num_envs, STUDENT_JOINT_COUNT),
        "anchor_pos_z": (num_envs, 1),
        "anchor_quat_wxyz": (num_envs, 4),
        "anchor_lin_vel_b": (num_envs, 3),
        "anchor_ang_vel_b": (num_envs, 3),
        "body_pos_b": (num_envs, len(body_names), 3),
    }
    resolved_device = torch.device(expected_device) if expected_device is not None else None

    for field_name, expected_shape in expected_shapes.items():
        value = getattr(reference, field_name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Reference {field_name} must be a torch.Tensor, got {type(value).__name__}.")
        if not value.is_floating_point():
            raise TypeError(f"Reference {field_name} must use a floating-point dtype, got {value.dtype}.")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Invalid reference {field_name} shape: expected {expected_shape}, got {tuple(value.shape)}."
            )
        if resolved_device is not None and value.device != resolved_device:
            raise ValueError(
                f"Reference {field_name} is on {value.device}, but expected device {resolved_device}."
            )
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Reference {field_name} contains NaN or Inf.")

    quaternion_norm = torch.linalg.vector_norm(reference.anchor_quat_wxyz, dim=-1)
    if not torch.allclose(
        quaternion_norm,
        torch.ones_like(quaternion_norm),
        atol=quaternion_atol,
        rtol=0.0,
    ):
        raise ValueError(
            "Reference anchor quaternion must be normalized; "
            f"got norms {quaternion_norm.detach().cpu().tolist()}."
        )


def _validate_unique_names(names: tuple[str, ...], description: str) -> None:
    """Reject empty or duplicate names in a resolved name sequence."""

    empty_names = [index for index, name in enumerate(names) if not name]
    if empty_names:
        raise ValueError(f"The {description} name list contains empty entries at indices {empty_names}.")

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"The {description} name list contains duplicates: {duplicates}.")
