"""Observation terms for the replaceable motion tracking reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .reference_command import ReferenceCommand


def _get_reference_command(env: ManagerBasedEnv, command_name: str) -> ReferenceCommand:
    """Return the configured motion tracking reference command."""

    return env.command_manager.get_term(command_name)


def motion_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the current reference joint positions in checkpoint joint order."""

    command = _get_reference_command(env, command_name)
    return command.joint_pos.view(env.num_envs, -1)


def motion_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the current reference joint velocities in checkpoint joint order."""

    command = _get_reference_command(env, command_name)
    return command.joint_vel.view(env.num_envs, -1)


def motion_anchor_pos_z(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the reference anchor height in world coordinates."""

    command = _get_reference_command(env, command_name)
    return command.anchor_pos_w[:, 2:3]


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the yaw-invariant local anchor orientation as a flattened 6D matrix."""

    command = _get_reference_command(env, command_name)

    anchor_quat_local = quat_mul(
        quat_inv(yaw_quat(command.anchor_quat_w)),
        command.anchor_quat_w,
    )
    robot_anchor_quat_local = quat_mul(
        quat_inv(yaw_quat(command.robot_anchor_quat_w)),
        command.robot_anchor_quat_w,
    )
    orientation_local_b = quat_mul(
        quat_inv(robot_anchor_quat_local),
        anchor_quat_local,
    )

    orientation_matrix = matrix_from_quat(orientation_local_b)
    return orientation_matrix[..., :2].reshape(orientation_matrix.shape[0], -1)


def motion_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return tracked reference body positions relative to the anchor frame."""

    command = _get_reference_command(env, command_name)
    return command.body_pos_b.view(env.num_envs, -1)


def motion_anchor_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the reference anchor linear velocity in its local frame."""

    command = _get_reference_command(env, command_name)
    return command.anchor_lin_vel_b.view(env.num_envs, -1)


def motion_anchor_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the reference anchor angular velocity in its local frame."""

    command = _get_reference_command(env, command_name)
    return command.anchor_ang_vel_b.view(env.num_envs, -1)
