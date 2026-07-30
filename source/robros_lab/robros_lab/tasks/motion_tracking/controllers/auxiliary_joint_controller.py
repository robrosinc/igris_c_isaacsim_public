"""Independent wrist and hand position control for IGRIS-C motion tracking."""

from __future__ import annotations
from collections.abc import Sequence


import torch
from isaaclab.assets import Articulation


WRIST_JOINT_NAMES = (
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
)

ACTIVE_HAND_JOINT_NAMES = (
    "l_0_joint_thumb_proximal",
    "l_1_joint_thumb_middle",
    "l_3_joint_index_middle",
    "l_5_joint_middle_middle",
    "l_7_joint_ring_middle",
    "l_9_joint_little_middle",
    "r_0_joint_thumb_proximal",
    "r_1_joint_thumb_middle",
    "r_3_joint_index_middle",
    "r_5_joint_middle_middle",
    "r_7_joint_ring_middle",
    "r_9_joint_little_middle",
)

DISTAL_COUPLING = (
    ("l_thumb", "l_1_joint_thumb_middle", "l_2_joint_thumb_distal", 1.285714286),
    ("l_index", "l_3_joint_index_middle", "l_4_joint_index_distal", 1.0),
    ("l_middle", "l_5_joint_middle_middle", "l_6_joint_middle_distal", 1.0),
    ("l_ring", "l_7_joint_ring_middle", "l_8_joint_ring_distal", 1.0),
    ("l_little", "l_9_joint_little_middle", "l_10_joint_little_distal", 1.0),
    ("r_thumb", "r_1_joint_thumb_middle", "r_2_joint_thumb_distal", 1.285714286),
    ("r_index", "r_3_joint_index_middle", "r_4_joint_index_distal", 1.0),
    ("r_middle", "r_5_joint_middle_middle", "r_6_joint_middle_distal", 1.0),
    ("r_ring", "r_7_joint_ring_middle", "r_8_joint_ring_distal", 1.0),
    ("r_little", "r_9_joint_little_middle", "r_10_joint_little_distal", 1.0),
)

AUXILIARY_JOINT_NAMES = WRIST_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES


class WristHandPoseController:
    """Hold the default wrist and active-hand pose outside the student action.

    The student checkpoint continues to control only its original 23 joints.
    The ten distal finger targets are computed from the active middle-joint
    targets immediately before commands are written to the articulation.
    """

    def __init__(self, robot: Articulation) -> None:
        self._robot = robot
        joint_ids, joint_names = robot.find_joints(
            list(AUXILIARY_JOINT_NAMES),
            preserve_order=True,
        )
        resolved_joint_names = tuple(joint_names)
        if resolved_joint_names != AUXILIARY_JOINT_NAMES:
            raise ValueError(
                "Wrist/hand joint names or order do not match the auxiliary controller: "
                f"expected {AUXILIARY_JOINT_NAMES}, got {resolved_joint_names}."
            )

        distal_joint_names = [distal_name for _, _, distal_name, _ in DISTAL_COUPLING]
        distal_joint_ids, resolved_distal_joint_names = robot.find_joints(
            distal_joint_names,
            preserve_order=True,
        )
        if tuple(resolved_distal_joint_names) != tuple(distal_joint_names):
            raise ValueError("Distal joint names or order do not match the robot articulation.")

        self._joint_ids = joint_ids
        self._distal_joint_ids = distal_joint_ids
        self._position_target = robot.data.default_joint_pos[:, joint_ids].clone()
        self._velocity_target = torch.zeros_like(self._position_target)
        self._middle_target_indices = [
            AUXILIARY_JOINT_NAMES.index(middle_name) for _, middle_name, _, _ in DISTAL_COUPLING
        ]
        self._distal_multipliers = self._position_target.new_tensor(
            [multiplier for _, _, _, multiplier in DISTAL_COUPLING]
        ).unsqueeze(0)
        distal_position_limits = robot.data.joint_pos_limits[:, distal_joint_ids, :]
        position_limits = robot.data.joint_pos_limits[:, joint_ids, :]
        self._lower_limits = position_limits[..., 0].clone()
        self._upper_limits = position_limits[..., 1].clone()

        self._distal_lower_limits = distal_position_limits[..., 0].clone()
        self._distal_upper_limits = distal_position_limits[..., 1].clone()
        if not torch.all(self._distal_lower_limits <= self._distal_upper_limits):
            raise ValueError("Distal hand joint position limits are invalid.")
    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the resolved wrist and active-hand joint order."""
        return AUXILIARY_JOINT_NAMES

    @property
    def position_target(self) -> torch.Tensor:
        """Return a copy of the held default joint positions."""
        return self._position_target.clone()
    def _compute_distal_position_target(self) -> torch.Tensor:

        """Compute and clamp distal targets from the authoritative middle targets."""
        middle_targets = self._position_target[:, self._middle_target_indices]
        distal_targets = middle_targets * self._distal_multipliers
        return torch.clamp(
            distal_targets,
            min=self._distal_lower_limits,
            max=self._distal_upper_limits,
        )

    def set_target(
        self,
        *,
        joint_names: Sequence[str],
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor | None = None,
    ) -> None:
        """Update wrist and active-hand targets in the controller's fixed order."""

        resolved_joint_names = tuple(joint_names)
        if resolved_joint_names != AUXILIARY_JOINT_NAMES:
            raise ValueError(
                "Wrist/hand target names or order do not match the controller: "
                f"expected {AUXILIARY_JOINT_NAMES}, got {resolved_joint_names}."
            )
        if joint_pos.shape != self._position_target.shape:
            raise ValueError(
                f"Wrist/hand position target shape must be {self._position_target.shape}, "
                f"got {joint_pos.shape}."
            )
        if joint_pos.device != self._position_target.device:
            raise ValueError(
                f"Wrist/hand position target is on {joint_pos.device}, "
                f"expected {self._position_target.device}."
            )
        if not torch.isfinite(joint_pos).all().item():
            raise ValueError("Wrist/hand position target contains NaN or Inf.")

        if joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos)
        if joint_vel.shape != self._velocity_target.shape:
            raise ValueError(
                f"Wrist/hand velocity target shape must be {self._velocity_target.shape}, "
                f"got {joint_vel.shape}."
            )
        if joint_vel.device != self._velocity_target.device:
            raise ValueError(
                f"Wrist/hand velocity target is on {joint_vel.device}, "
                f"expected {self._velocity_target.device}."
            )
        if not torch.isfinite(joint_vel).all().item():
            raise ValueError("Wrist/hand velocity target contains NaN or Inf.")

        self._position_target.copy_(
            torch.clamp(joint_pos, self._lower_limits, self._upper_limits)
        )
        self._velocity_target.copy_(joint_vel)

    def apply(self) -> None:
        """Write wrist, active-hand, and computed distal targets to the articulation."""
        self._robot.set_joint_position_target(
            self._position_target,
            joint_ids=self._joint_ids,
        )
        self._robot.set_joint_velocity_target(
            self._velocity_target,
            joint_ids=self._joint_ids,
        )
        distal_position_target = self._compute_distal_position_target()
        distal_velocity_target = (
            self._velocity_target[:, self._middle_target_indices] * self._distal_multipliers
        )
        self._robot.set_joint_position_target(
            distal_position_target,
            joint_ids=self._distal_joint_ids,
        )
        self._robot.set_joint_velocity_target(
            distal_velocity_target,
            joint_ids=self._distal_joint_ids,
        )

__all__ = [
    "ACTIVE_HAND_JOINT_NAMES",
    "AUXILIARY_JOINT_NAMES",
    "DISTAL_COUPLING",
    "WRIST_JOINT_NAMES",
    "WristHandPoseController",
]
