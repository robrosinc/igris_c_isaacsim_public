"""Synchronized NPZ command replay for HugWBC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


HUGWBC_COMMAND_SCHEMA = "igris_c_hugwbc_command_v1"
TELEOPERATION_JOINT_NAMES = (
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
)
TELEOPERATION_RANGES = np.asarray(
    [
        (-1.5, 1.5),
        (-0.2, 0.2),
        (-1.0, 0.28),
        (-3.0, 1.0),
        (-0.15, 3.0),
        (-1.5, 1.5),
        (-2.1, 0.0),
        (-3.0, 1.0),
        (-3.0, 0.15),
        (-1.5, 1.5),
        (-2.1, 0.0),
    ],
    dtype=np.float32,
)
WRIST_HAND_JOINT_NAMES = (
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
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
_REQUIRED_KEYS = {
    "schema_version",
    "fps",
    "base_velocity",
    "teleop_joint_names",
    "teleop_joint_pos",
}
_WRIST_HAND_KEYS = {
    "wrist_hand_joint_names",
    "wrist_hand_joint_pos",
    "wrist_hand_joint_vel",
}


@dataclass(frozen=True)
class HugWBCCommandFrame:
    frame_index: int
    base_velocity: torch.Tensor
    teleop_joint_names: tuple[str, ...]
    teleop_joint_pos: torch.Tensor


@dataclass(frozen=True)
class WristHandCommandFrame:
    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class NpzHugWBCCommandSource:
    """Load one command clip and expose a simulation-time-driven shared timeline."""

    def __init__(
        self,
        *,
        motion_file: str | Path,
        num_envs: int,
        device: torch.device | str,
        loop: bool = False,
    ) -> None:
        self.motion_file = Path(motion_file).expanduser().resolve()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.loop = bool(loop)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")
        self.fps = 0.0
        self.frame_count = 0
        self.teleop_joint_names = TELEOPERATION_JOINT_NAMES
        self.wrist_hand_joint_names: tuple[str, ...] = ()
        self._base_velocity: torch.Tensor | None = None
        self._teleop_joint_pos: torch.Tensor | None = None
        self._wrist_hand_joint_pos: torch.Tensor | None = None
        self._wrist_hand_joint_vel: torch.Tensor | None = None
        self._start_timestamp: float | None = None
        self.current_frame_index = 0
        self.is_finished = False

    @property
    def duration(self) -> float:
        return 0.0 if self.frame_count <= 1 else (self.frame_count - 1) / self.fps

    def initialize(self) -> None:
        if not self.motion_file.is_file():
            raise FileNotFoundError(f"HugWBC command NPZ does not exist: {self.motion_file}")
        with np.load(self.motion_file, allow_pickle=False) as data:
            missing = _REQUIRED_KEYS - set(data.files)
            if missing:
                raise KeyError(f"HugWBC command NPZ is missing keys: {sorted(missing)}.")
            schema = str(np.asarray(data["schema_version"]).reshape(-1)[0])
            if schema != HUGWBC_COMMAND_SCHEMA:
                raise ValueError(f"Expected schema {HUGWBC_COMMAND_SCHEMA!r}, got {schema!r}.")
            fps_array = np.asarray(data["fps"])
            if fps_array.size != 1:
                raise ValueError(f"fps must be a scalar, got shape {fps_array.shape}.")
            fps = float(fps_array.reshape(-1)[0])
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"fps must be positive and finite, got {fps}.")
            dataset_names = self._read_names(data, "teleop_joint_names")
            teleop_indices = self._resolve_indices(dataset_names, TELEOPERATION_JOINT_NAMES)
            base_velocity = self._read_float_array(data, "base_velocity", ndim=2)
            teleop_joint_pos = self._read_float_array(data, "teleop_joint_pos", ndim=2)
            if base_velocity.shape[1:] != (3,):
                raise ValueError(f"base_velocity must have shape (F, 3), got {base_velocity.shape}.")
            if teleop_joint_pos.shape[1] != len(dataset_names):
                raise ValueError(
                    "teleop_joint_pos width must match teleop_joint_names; "
                    f"got {teleop_joint_pos.shape[1]} and {len(dataset_names)}."
                )
            teleop_joint_pos = teleop_joint_pos[:, teleop_indices]
            frame_count = int(base_velocity.shape[0])
            if frame_count < 1 or teleop_joint_pos.shape[0] != frame_count:
                raise ValueError("All HugWBC command arrays must contain the same non-zero frame count.")
            outside_range = np.logical_or(
                teleop_joint_pos < TELEOPERATION_RANGES[:, 0],
                teleop_joint_pos > TELEOPERATION_RANGES[:, 1],
            )
            if outside_range.any():
                first_bad_frame, first_bad_joint = np.argwhere(outside_range)[0]
                raise ValueError(
                    "teleop_joint_pos exceeds the configured range at "
                    f"frame {int(first_bad_frame)}, joint "
                    f"{TELEOPERATION_JOINT_NAMES[int(first_bad_joint)]!r}."
                )
            if "teleop_joint_vel" in data.files:
                teleop_joint_vel = self._read_float_array(
                    data, "teleop_joint_vel", ndim=2
                )
                expected_velocity_shape = (frame_count, len(dataset_names))
                if teleop_joint_vel.shape != expected_velocity_shape:
                    raise ValueError(
                        "teleop_joint_vel must match the un-mapped teleoperation layout; "
                        f"expected {expected_velocity_shape}, got {teleop_joint_vel.shape}."
                    )

            present_wrist_keys = _WRIST_HAND_KEYS.intersection(data.files)
            if present_wrist_keys and present_wrist_keys != _WRIST_HAND_KEYS:
                raise KeyError(
                    "Wrist/hand keys are all-or-none; missing "
                    f"{sorted(_WRIST_HAND_KEYS - present_wrist_keys)}."
                )
            wrist_names: tuple[str, ...] = ()
            wrist_pos = wrist_vel = None
            if present_wrist_keys:
                dataset_wrist_names = self._read_names(data, "wrist_hand_joint_names")
                wrist_indices = self._resolve_named_indices(
                    dataset_wrist_names,
                    WRIST_HAND_JOINT_NAMES,
                    "wrist_hand_joint_names",
                )
                wrist_names = WRIST_HAND_JOINT_NAMES
                wrist_pos = self._read_float_array(data, "wrist_hand_joint_pos", ndim=2)
                wrist_vel = self._read_float_array(data, "wrist_hand_joint_vel", ndim=2)
                expected_shape = (frame_count, len(dataset_wrist_names))
                if wrist_pos.shape != expected_shape or wrist_vel.shape != expected_shape:
                    raise ValueError(
                        f"Wrist/hand arrays must have shape {expected_shape}, got "
                        f"{wrist_pos.shape} and {wrist_vel.shape}."
                    )
                wrist_pos = wrist_pos[:, wrist_indices]
                wrist_vel = wrist_vel[:, wrist_indices]

        self.fps = fps
        self.frame_count = frame_count
        self.wrist_hand_joint_names = wrist_names
        self._base_velocity = self._to_tensor(base_velocity)
        self._teleop_joint_pos = self._to_tensor(teleop_joint_pos)
        self._wrist_hand_joint_pos = None if wrist_pos is None else self._to_tensor(wrist_pos)
        self._wrist_hand_joint_vel = None if wrist_vel is None else self._to_tensor(wrist_vel)
        self.reset()

    def reset(self) -> None:
        self._require_initialized()
        self._start_timestamp = None
        self.current_frame_index = 0
        self.is_finished = False

    def frame_at(self, timestamp: float) -> HugWBCCommandFrame:
        self._require_initialized()
        if not isinstance(timestamp, (float, int)) or not math.isfinite(float(timestamp)):
            raise ValueError(f"timestamp must be finite, got {timestamp}.")
        timestamp = float(timestamp)
        if self._start_timestamp is None:
            self._start_timestamp = timestamp
        elapsed = max(0.0, timestamp - self._start_timestamp)
        unbounded_index = int(math.floor(elapsed * self.fps + 1.0e-6))
        if self.loop:
            frame_index = unbounded_index % self.frame_count
            self.is_finished = False
        else:
            frame_index = min(unbounded_index, self.frame_count - 1)
            self.is_finished = unbounded_index >= self.frame_count - 1
        self.current_frame_index = frame_index
        assert self._base_velocity is not None
        assert self._teleop_joint_pos is not None
        return HugWBCCommandFrame(
            frame_index=frame_index,
            base_velocity=self._batched(self._base_velocity, frame_index),
            teleop_joint_names=self.teleop_joint_names,
            teleop_joint_pos=self._batched(self._teleop_joint_pos, frame_index),
        )

    def get_wrist_hand_reference(self) -> WristHandCommandFrame | None:
        self._require_initialized()
        if self._wrist_hand_joint_pos is None or self._wrist_hand_joint_vel is None:
            return None
        return WristHandCommandFrame(
            joint_names=self.wrist_hand_joint_names,
            joint_pos=self._batched(self._wrist_hand_joint_pos, self.current_frame_index),
            joint_vel=self._batched(self._wrist_hand_joint_vel, self.current_frame_index),
        )

    def close(self) -> None:
        self._base_velocity = None
        self._teleop_joint_pos = None
        self._wrist_hand_joint_pos = None
        self._wrist_hand_joint_vel = None

    def _batched(self, array: torch.Tensor, index: int) -> torch.Tensor:
        return array[index].unsqueeze(0).expand(self.num_envs, *array.shape[1:])

    def _require_initialized(self) -> None:
        if self._base_velocity is None or self._teleop_joint_pos is None:
            raise RuntimeError("NpzHugWBCCommandSource must be initialized before use.")

    @staticmethod
    def _read_names(data: np.lib.npyio.NpzFile, key: str) -> tuple[str, ...]:
        names = tuple(str(value) for value in np.asarray(data[key]).reshape(-1).tolist())
        if not names or any(not name for name in names):
            raise ValueError(f"{key} must contain non-empty names.")
        if len(names) != len(set(names)):
            raise ValueError(f"{key} contains duplicate names.")
        return names

    @staticmethod
    def _resolve_indices(available: tuple[str, ...], requested: tuple[str, ...]) -> list[int]:
        return NpzHugWBCCommandSource._resolve_named_indices(
            available,
            requested,
            "teleop_joint_names",
        )

    @staticmethod
    def _resolve_named_indices(
        available: tuple[str, ...],
        requested: tuple[str, ...],
        key: str,
    ) -> list[int]:
        index_by_name = {name: index for index, name in enumerate(available)}
        missing = [name for name in requested if name not in index_by_name]
        if missing:
            raise ValueError(f"{key} is missing canonical joints: {missing}.")
        return [index_by_name[name] for name in requested]

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
        return np.ascontiguousarray(array)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)


__all__ = [
    "HUGWBC_COMMAND_SCHEMA",
    "HugWBCCommandFrame",
    "NpzHugWBCCommandSource",
    "TELEOPERATION_JOINT_NAMES",
    "TELEOPERATION_RANGES",
    "WRIST_HAND_JOINT_NAMES",
    "WristHandCommandFrame",
]
