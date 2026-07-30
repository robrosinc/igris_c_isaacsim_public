"""Isaac Lab command term backed by a replaceable motion-reference source."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from ..controllers.base import (
    MotionReferenceFrame,
    MotionReferenceSource,
    validate_motion_reference,
)
from ..controllers.standing_reference import StandingReferenceSource

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ReferenceCommand(CommandTerm):
    """Expose a streaming motion reference through the tracking command API.

    The command starts with a frozen StandingReferenceSource. Runtime code may
    replace it with a GMR, MPC, or replay source through set_source. Observation
    terms continue to read the command named motion and do not depend on the
    selected source backend.
    """

    cfg: ReferenceCommandCfg

    def __init__(self, cfg: ReferenceCommandCfg, env: ManagerBasedRLEnv):
        self.robot = env.scene[cfg.asset_name]

        joint_indices, joint_names = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        body_indices, body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        anchor_indices, anchor_names = self.robot.find_bodies(cfg.anchor_body_name, preserve_order=True)

        if len(anchor_indices) != 1:
            raise ValueError(
                f"Expected exactly one anchor body for {cfg.anchor_body_name!r}, got {anchor_names}."
            )
        if not joint_indices:
            raise ValueError("Reference command must resolve at least one robot joint.")
        if not body_indices:
            raise ValueError("Reference command must resolve at least one tracked body.")

        self.joint_indexes = torch.tensor(joint_indices, dtype=torch.long, device=env.device)
        self.resolved_joint_names = tuple(joint_names)
        self.body_indexes = torch.tensor(body_indices, dtype=torch.long, device=env.device)
        self.resolved_body_names = tuple(body_names)
        self.robot_anchor_body_index = int(anchor_indices[0])
        self.robot_anchor_body_name = str(anchor_names[0])

        if self.robot_anchor_body_name not in self.resolved_body_names:
            raise ValueError(
                f"Anchor body {self.robot_anchor_body_name!r} must be included in the tracked body selection "
                f"{self.resolved_body_names}."
            )

        super().__init__(cfg, env)

        self._anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._reference: MotionReferenceFrame | None = None
        self._source: MotionReferenceSource | None = None

        standing_source = StandingReferenceSource(
            robot=self.robot,
            joint_names=self.resolved_joint_names,
            body_names=self.resolved_body_names,
            anchor_body_name=self.robot_anchor_body_name,
        )
        self.set_source(standing_source, initialize=True, close_previous=False)

    @property
    def source(self) -> MotionReferenceSource:
        """Currently active reference source."""

        if self._source is None:
            raise RuntimeError("Reference command does not have an active source.")
        return self._source

    @property
    def reference_timestamp(self) -> float:
        """Timestamp of the currently exposed reference frame."""

        return self._require_reference().timestamp

    @property
    def command(self) -> torch.Tensor:
        """Concatenated joint position and velocity reference."""

        frame = self._require_reference()
        return torch.cat((frame.joint_pos, frame.joint_vel), dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._require_reference().joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._require_reference().joint_vel

    @property
    def body_pos_b(self) -> torch.Tensor:
        return self._require_reference().body_pos_b

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._anchor_pos_w

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._require_reference().anchor_quat_wxyz

    @property
    def anchor_lin_vel_b(self) -> torch.Tensor:
        return self._require_reference().anchor_lin_vel_b

    @property
    def anchor_ang_vel_b(self) -> torch.Tensor:
        return self._require_reference().anchor_ang_vel_b

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def set_source(
        self,
        source: MotionReferenceSource,
        *,
        initialize: bool = True,
        close_previous: bool = True,
    ) -> None:
        """Atomically replace the active source after validating its first frame."""

        if not isinstance(source, MotionReferenceSource):
            raise TypeError("source must implement the MotionReferenceSource protocol.")
        if initialize:
            source.initialize()

        first_frame = source.get_reference(self._simulation_time())
        validate_motion_reference(
            first_frame,
            num_envs=self.num_envs,
            expected_joint_names=self.resolved_joint_names,
            expected_body_names=self.resolved_body_names,
            expected_device=self.device,
        )

        previous_source = self._source
        self._source = source
        self._apply_reference(first_frame, validate=False)

        if close_previous and previous_source is not None and previous_source is not source:
            previous_source.close()

    def close(self) -> None:
        """Close the active source and make the command unavailable."""

        if self._source is not None:
            self._source.close()
        self._source = None
        self._reference = None

    def _update_metrics(self) -> None:
        """The inference-only reference command currently has no metrics."""

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self.source.reset(env_ids)

    def _update_command(self) -> None:
        frame = self.source.get_reference(self._simulation_time())
        if not frame.valid:
            return

        self._apply_reference(
            frame,
            validate=self.cfg.validate_reference_every_step,
        )

    def _apply_reference(self, frame: MotionReferenceFrame, *, validate: bool) -> None:
        if validate:
            validate_motion_reference(
                frame,
                num_envs=self.num_envs,
                expected_joint_names=self.resolved_joint_names,
                expected_body_names=self.resolved_body_names,
                expected_device=self.device,
            )
        else:
            self._validate_runtime_layout(frame)

        self._reference = frame
        self._anchor_pos_w.zero_()
        self._anchor_pos_w[:, 2:3].copy_(frame.anchor_pos_z)

    def _validate_runtime_layout(self, frame: MotionReferenceFrame) -> None:
        """Check cheap structural invariants without synchronizing CUDA."""

        if tuple(frame.joint_names) != self.resolved_joint_names:
            raise ValueError("Reference joint names/order changed at runtime.")
        if tuple(frame.body_names) != self.resolved_body_names:
            raise ValueError("Reference body names/order changed at runtime.")

        expected_shapes = {
            "joint_pos": (self.num_envs, len(self.resolved_joint_names)),
            "joint_vel": (self.num_envs, len(self.resolved_joint_names)),
            "anchor_pos_z": (self.num_envs, 1),
            "anchor_quat_wxyz": (self.num_envs, 4),
            "anchor_lin_vel_b": (self.num_envs, 3),
            "anchor_ang_vel_b": (self.num_envs, 3),
            "body_pos_b": (self.num_envs, len(self.resolved_body_names), 3),
        }
        for field_name, expected_shape in expected_shapes.items():
            value = getattr(frame, field_name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Reference {field_name} must be a torch.Tensor.")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Reference {field_name} shape changed: expected {expected_shape}, "
                    f"got {tuple(value.shape)}."
                )
            if value.device != torch.device(self.device):
                raise ValueError(
                    f"Reference {field_name} is on {value.device}, expected {self.device}."
                )
            if not value.is_floating_point():
                raise TypeError(f"Reference {field_name} must use a floating-point dtype.")

    def _simulation_time(self) -> float:
        return float(self._env.common_step_counter) * float(self._env.step_dt)

    def _require_reference(self) -> MotionReferenceFrame:
        if self._reference is None:
            raise RuntimeError("Reference command has not received a valid frame.")
        return self._reference


@configclass
class ReferenceCommandCfg(CommandTermCfg):
    """Configuration for a replaceable IGRIS-C motion-reference command."""

    class_type: type = ReferenceCommand

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING
    anchor_body_name: str = MISSING

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
    validate_reference_every_step: bool = False
