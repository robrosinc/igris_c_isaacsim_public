"""Offline NPZ replay for the motion tracking reference command."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .base import EnvIds, MotionReferenceFrame, validate_motion_reference


_REQUIRED_KEYS = {
    "fps",
    "joint_names",
    "body_names",
    "local_frame_body_name",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_pos_b",
    "body_lin_vel_b",
    "body_ang_vel_b",
}

_AUXILIARY_KEYS = {
    "wrist_hand_joint_names",
    "wrist_hand_joint_pos",
    "wrist_hand_joint_vel",
}


@dataclass(frozen=True)
class AuxiliaryJointReferenceFrame:
    """One synchronized wrist/hand frame stored alongside the student reference."""

    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class NpzMotionReferenceSource:
    """Replay a processed tracking NPZ through :class:`ReferenceCommand`.

    The source uses the same processed arrays as the tracking ``MotionCommand``.
    All environments receive the same frame. The timeline either holds the last
    frame or loops when it reaches the end of the file.
    """

    def __init__(
        self,
        *,
        motion_file: str | Path,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        auxiliary_joint_names: Sequence[str] = (),
        anchor_body_name: str,
        num_envs: int,
        device: torch.device | str,
        loop: bool = False,
    ) -> None:
        self.motion_file = Path(motion_file).expanduser().resolve()
        self._joint_names = tuple(joint_names)
        self._body_names = tuple(body_names)
        self._anchor_body_name = str(anchor_body_name)
        self._auxiliary_joint_names = tuple(auxiliary_joint_names)
        self._num_envs = int(num_envs)
        self._device = torch.device(device)
        self._loop = bool(loop)

        if self._num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self._num_envs}.")
        if not self._joint_names:
            raise ValueError("joint_names must not be empty.")
        if not self._body_names:
            raise ValueError("body_names must not be empty.")
        if self._anchor_body_name not in self._body_names:
            raise ValueError(
                f"Anchor body {self._anchor_body_name!r} must be present in body_names."
            )

        self._fps = 0.0
        self._frame_count = 0
        self._joint_pos: torch.Tensor | None = None
        self._joint_vel: torch.Tensor | None = None
        self._anchor_pos_z: torch.Tensor | None = None
        self._auxiliary_joint_pos: torch.Tensor | None = None
        self._auxiliary_joint_vel: torch.Tensor | None = None
        self._anchor_quat_wxyz: torch.Tensor | None = None
        self._anchor_lin_vel_b: torch.Tensor | None = None
        self._anchor_ang_vel_b: torch.Tensor | None = None
        self._body_pos_b: torch.Tensor | None = None
        self._start_timestamp: float | None = None
        self._current_frame_index = 0
        self._is_finished = False
        self._closed = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        if self._fps <= 0.0 or self._frame_count <= 1:
            return 0.0
        return (self._frame_count - 1) / self._fps

    @property
    def has_auxiliary_reference(self) -> bool:
        """Return whether the NPZ contains a complete wrist/hand channel."""
        return self._auxiliary_joint_pos is not None

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    def initialize(self) -> None:
        """Load and validate the processed motion dataset."""

        auxiliary_joint_pos: np.ndarray | None = None
        auxiliary_joint_vel: np.ndarray | None = None

        if not self.motion_file.is_file():
            raise FileNotFoundError(f"Motion NPZ does not exist: {self.motion_file}")

        with np.load(self.motion_file, allow_pickle=False) as data:
            present_auxiliary_keys = _AUXILIARY_KEYS.intersection(data.files)
            if present_auxiliary_keys and present_auxiliary_keys != _AUXILIARY_KEYS:
                missing_auxiliary_keys = _AUXILIARY_KEYS - present_auxiliary_keys
                raise KeyError(
                    f"Motion NPZ {self.motion_file} has an incomplete wrist/hand "
                    f"channel; missing keys: {sorted(missing_auxiliary_keys)}."
                )

            missing_keys = _REQUIRED_KEYS - set(data.files)
            if missing_keys:
                raise KeyError(
                    f"Motion NPZ {self.motion_file} is missing keys: {sorted(missing_keys)}."
                )

            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"Motion fps must be positive and finite, got {fps}.")

            dataset_joint_names = tuple(
                str(name) for name in np.asarray(data["joint_names"]).reshape(-1).tolist()
            )
            dataset_body_names = tuple(
                str(name) for name in np.asarray(data["body_names"]).reshape(-1).tolist()
            )
            local_frame_body_name = str(
                np.asarray(data["local_frame_body_name"]).reshape(-1)[0]
            )

            joint_indices = self._resolve_indices(
                dataset_joint_names,
                self._joint_names,
                "joint",
            )
            body_indices = self._resolve_indices(
                dataset_body_names,
                self._body_names,
                "body",
            )
            try:
                anchor_body_index = dataset_body_names.index(self._anchor_body_name)
            except ValueError as exc:
                raise ValueError(
                    f"Anchor body {self._anchor_body_name!r} is missing from "
                    f"{self.motion_file}."
                ) from exc
            try:
                local_frame_body_index = dataset_body_names.index(local_frame_body_name)
            except ValueError as exc:
                raise ValueError(
                    f"Local-frame body {local_frame_body_name!r} is missing from "
                    f"{self.motion_file}."
                ) from exc

            joint_pos = self._read_float_array(data, "joint_pos", ndim=2)[:, joint_indices]
            if self._auxiliary_joint_names and present_auxiliary_keys:
                dataset_auxiliary_joint_names = tuple(
                    str(name)
                    for name in np.asarray(data["wrist_hand_joint_names"])
                    .reshape(-1)
                    .tolist()
                )
                auxiliary_joint_indices = self._resolve_indices(
                    dataset_auxiliary_joint_names,
                    self._auxiliary_joint_names,
                    "wrist/hand joint",
                )
                auxiliary_joint_pos = self._read_float_array(
                    data, "wrist_hand_joint_pos", ndim=2
                )[:, auxiliary_joint_indices]
                auxiliary_joint_vel = self._read_float_array(
                    data, "wrist_hand_joint_vel", ndim=2
                )[:, auxiliary_joint_indices]

            joint_vel = self._read_float_array(data, "joint_vel", ndim=2)[:, joint_indices]
            body_pos_w = self._read_float_array(data, "body_pos_w", ndim=3)
            body_quat_w = self._read_float_array(data, "body_quat_w", ndim=3)
            body_pos_b = self._read_float_array(data, "body_pos_b", ndim=3)[:, body_indices]
            body_lin_vel_b = self._read_float_array(data, "body_lin_vel_b", ndim=3)
            body_ang_vel_b = self._read_float_array(data, "body_ang_vel_b", ndim=3)

        frame_count = int(joint_pos.shape[0])
        if frame_count <= 0:
            raise ValueError(f"Motion NPZ contains no frames: {self.motion_file}")
        arrays = {
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_pos_b": body_pos_b,
            "body_lin_vel_b": body_lin_vel_b,
            "body_ang_vel_b": body_ang_vel_b,
        }
        if auxiliary_joint_pos is not None and auxiliary_joint_vel is not None:
            arrays["wrist_hand_joint_pos"] = auxiliary_joint_pos
            arrays["wrist_hand_joint_vel"] = auxiliary_joint_vel

        for name, array in arrays.items():
            if int(array.shape[0]) != frame_count:
                raise ValueError(
                    f"{name} has {array.shape[0]} frames, expected {frame_count}."
                )

        anchor_quat = body_quat_w[:, anchor_body_index]
        quaternion_norm = np.linalg.norm(anchor_quat, axis=-1, keepdims=True)
        if np.any(quaternion_norm <= np.finfo(np.float32).eps):
            raise ValueError("Motion NPZ contains a zero-norm anchor quaternion.")
        anchor_quat = anchor_quat / quaternion_norm

        self._fps = fps
        self._frame_count = frame_count
        self._joint_pos = self._to_tensor(joint_pos)
        self._joint_vel = self._to_tensor(joint_vel)
        self._anchor_pos_z = self._to_tensor(body_pos_w[:, anchor_body_index, 2:3])
        self._anchor_quat_wxyz = self._to_tensor(anchor_quat)
        self._anchor_lin_vel_b = self._to_tensor(
            body_lin_vel_b[:, local_frame_body_index]
        )
        self._anchor_ang_vel_b = self._to_tensor(
            body_ang_vel_b[:, local_frame_body_index]
        )
        self._body_pos_b = self._to_tensor(body_pos_b)
        self._start_timestamp = None
        self._current_frame_index = 0
        self._is_finished = False
        self._closed = False
        if auxiliary_joint_pos is None or auxiliary_joint_vel is None:
            self._auxiliary_joint_pos = None
            self._auxiliary_joint_vel = None
        else:
            self._auxiliary_joint_pos = self._to_tensor(auxiliary_joint_pos)
            self._auxiliary_joint_vel = self._to_tensor(auxiliary_joint_vel)


        validate_motion_reference(
            self._frame_at(0, timestamp=0.0),
            num_envs=self._num_envs,
            expected_joint_names=self._joint_names,
            expected_body_names=self._body_names,
            expected_device=self._device,
        )
    def reset(self, env_ids: EnvIds = None) -> None:
        """Restart the synchronized replay timeline from frame zero."""

        del env_ids
        self._require_initialized()
        self._start_timestamp = None
        self._current_frame_index = 0
        self._is_finished = False

    def get_reference(self, timestamp: float) -> MotionReferenceFrame:
        """Return the frame corresponding to the supplied simulation timestamp."""

        self._require_initialized()
        if not isinstance(timestamp, (float, int)) or not math.isfinite(float(timestamp)):
            raise ValueError(f"timestamp must be finite, got {timestamp}.")
        timestamp = float(timestamp)
        if self._start_timestamp is None:
            self._start_timestamp = timestamp

        elapsed = max(0.0, timestamp - self._start_timestamp)
        unbounded_index = int(math.floor(elapsed * self._fps + 1.0e-6))
        if self._loop:
            frame_index = unbounded_index % self._frame_count
            self._is_finished = False
        else:
            frame_index = min(unbounded_index, self._frame_count - 1)
            self._is_finished = unbounded_index >= self._frame_count - 1

        self._current_frame_index = frame_index
        return self._frame_at(frame_index, timestamp=timestamp)

    def get_auxiliary_reference(self) -> AuxiliaryJointReferenceFrame | None:
        """Return wrist/hand data synchronized to the current student frame."""

        self._require_initialized()
        if self._auxiliary_joint_pos is None or self._auxiliary_joint_vel is None:
            return None

        def _batched(array: torch.Tensor) -> torch.Tensor:
            return array[self._current_frame_index].unsqueeze(0).expand(
                self._num_envs, *array.shape[1:]
            )

        return AuxiliaryJointReferenceFrame(
            joint_names=self._auxiliary_joint_names,
            joint_pos=_batched(self._auxiliary_joint_pos),
            joint_vel=_batched(self._auxiliary_joint_vel),
        )

    def close(self) -> None:
        """Release loaded motion tensors."""

        self._joint_pos = None
        self._joint_vel = None
        self._anchor_pos_z = None
        self._anchor_quat_wxyz = None
        self._anchor_lin_vel_b = None
        self._anchor_ang_vel_b = None
        self._body_pos_b = None
        self._auxiliary_joint_pos = None
        self._auxiliary_joint_vel = None
        self._start_timestamp = None
        self._closed = True

    def _frame_at(self, frame_index: int, *, timestamp: float) -> MotionReferenceFrame:
        self._require_initialized()
        assert self._joint_pos is not None
        assert self._joint_vel is not None
        assert self._anchor_pos_z is not None
        assert self._anchor_quat_wxyz is not None
        assert self._anchor_lin_vel_b is not None
        assert self._anchor_ang_vel_b is not None
        assert self._body_pos_b is not None

        def _batched(array: torch.Tensor) -> torch.Tensor:
            return array[frame_index].unsqueeze(0).expand(self._num_envs, *array.shape[1:])

        return MotionReferenceFrame(
            timestamp=timestamp,
            joint_names=self._joint_names,
            joint_pos=_batched(self._joint_pos),
            joint_vel=_batched(self._joint_vel),
            anchor_pos_z=_batched(self._anchor_pos_z),
            anchor_quat_wxyz=_batched(self._anchor_quat_wxyz),
            anchor_lin_vel_b=_batched(self._anchor_lin_vel_b),
            anchor_ang_vel_b=_batched(self._anchor_ang_vel_b),
            body_names=self._body_names,
            body_pos_b=_batched(self._body_pos_b),
            valid=True,
        )

    def _require_initialized(self) -> None:
        if self._closed or self._joint_pos is None:
            raise RuntimeError("NpzMotionReferenceSource must be initialized before use.")

    @staticmethod
    def _resolve_indices(
        available_names: tuple[str, ...],
        requested_names: tuple[str, ...],
        description: str,
    ) -> list[int]:
        index_by_name = {name: index for index, name in enumerate(available_names)}
        missing_names = [name for name in requested_names if name not in index_by_name]
        if missing_names:
            raise ValueError(
                f"Requested {description} names are missing from the motion NPZ: "
                f"{missing_names}."
            )
        return [index_by_name[name] for name in requested_names]

    @staticmethod
    def _read_float_array(
        data: np.lib.npyio.NpzFile,
        key: str,
        *,
        ndim: int,
    ) -> np.ndarray:
        array = np.asarray(data[key], dtype=np.float32)
        if array.ndim != ndim:
            raise ValueError(f"{key} must have {ndim} dimensions, got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or Inf.")
        return array

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(array),
            dtype=torch.float32,
            device=self._device,
        )
