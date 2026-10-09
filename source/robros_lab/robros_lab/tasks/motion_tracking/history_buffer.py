"""Student observation history without per-step device synchronization."""

from __future__ import annotations

import torch
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer


class TrackingHistoryBuffer(CircularBuffer):
    """Keep the fixed history length on the host and fill resets on the device."""

    def __init__(self, max_len: int, batch_size: int, device: str) -> None:
        super().__init__(max_len=max_len, batch_size=batch_size, device=device)
        self._length = max_len
        self._needs_fill = False

    @property
    def max_length(self) -> int:
        return self._length

    def reset(self, batch_ids=None) -> None:
        super().reset(batch_ids=batch_ids)
        self._needs_fill = True

    def append(self, data: torch.Tensor) -> None:
        if data.shape[0] != self.batch_size:
            raise ValueError(f"Expected batch size {self.batch_size}, got {data.shape[0]}.")
        data = data.to(self._device)
        if self._buffer is None:
            self._pointer = 0
            self._buffer = data.unsqueeze(0).expand(self._length, *data.shape).clone()
        else:
            self._pointer = (self._pointer + 1) % self._length
            self._buffer[self._pointer].copy_(data)
            if self._needs_fill:
                first = (self._num_pushes == 0).reshape(
                    (1, self.batch_size) + (1,) * (data.ndim - 1)
                )
                self._buffer.copy_(torch.where(first, data.unsqueeze(0), self._buffer))
        self._num_pushes += 1
        self._needs_fill = False

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        if len(key) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} lag indices, got {len(key)}.")
        if self._buffer is None or self._needs_fill:
            raise RuntimeError("Append data after reset before reading history.")
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        indices = torch.remainder(self._pointer - valid_keys, self._length)
        return self._buffer[indices, self._ALL_INDICES]


def _replace_buffer(original: CircularBuffer) -> TrackingHistoryBuffer:
    replacement = TrackingHistoryBuffer(
        max_len=original.max_length,
        batch_size=original.batch_size,
        device=original.device,
    )
    replacement._buffer = original._buffer
    replacement._pointer = original._pointer
    replacement._num_pushes = original._num_pushes
    replacement._needs_fill = original._buffer is not None
    return replacement


def install_student_history_buffers(observation_manager, term_names: tuple[str, ...]) -> None:
    """Replace only the student's buffers, preserving any samples already present."""

    histories = observation_manager._group_obs_term_history_buffer["student"]
    for term_name in term_names:
        histories[term_name] = _replace_buffer(histories[term_name])


def install_actuator_delay_buffers(robot) -> None:
    """Use the same device-resident buffer for delayed joint commands."""

    for actuator in robot.actuators.values():
        for name in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
            delay = getattr(actuator, name, None)
            if isinstance(delay, DelayBuffer):
                delay._circular_buffer = _replace_buffer(delay._circular_buffer)
