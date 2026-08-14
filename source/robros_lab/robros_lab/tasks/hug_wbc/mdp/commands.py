"""Base-velocity and upper-body command terms for HugWBC."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from ..controllers.npz_command_source import NpzHugWBCCommandSource


class HugWBCBaseVelocityCommand(CommandTerm):
    """Expose a fixed base command or a shared NPZ command timeline."""

    cfg: HugWBCBaseVelocityCommandCfg

    def __init__(self, cfg: HugWBCBaseVelocityCommandCfg, env):
        super().__init__(cfg, env)
        if cfg.rel_standing_envs != 0.0:
            raise ValueError("HugWBC walking playback requires rel_standing_envs=0.0.")
        self._command = torch.tensor(
            cfg.command,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._default_command = self._command.clone()
        self._is_standing_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._source: NpzHugWBCCommandSource | None = None

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def is_standing_env(self) -> torch.Tensor:
        return self._is_standing_env

    def set_source(self, source: NpzHugWBCCommandSource) -> None:
        self._source = source
        self._copy_source_frame()

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if self._source is None:
            self._command[env_ids] = self._default_command[env_ids]
        else:
            self._copy_source_frame()

    def _update_command(self) -> None:
        if self._source is not None:
            self._copy_source_frame()

    def _copy_source_frame(self) -> None:
        assert self._source is not None
        self._command.copy_(self._source.frame_at(self._simulation_time()).base_velocity)

    def _simulation_time(self) -> float:
        return float(self._env.common_step_counter) * float(self._env.step_dt)


@configclass
class HugWBCBaseVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`HugWBCBaseVelocityCommand`."""

    class_type: type[CommandTerm] = HugWBCBaseVelocityCommand
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rel_standing_envs: float = 0.0
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False


class TeleoperationCommand(CommandTerm):
    """Generate random poses or expose the shared NPZ command timeline."""

    cfg: TeleoperationCommandCfg

    def __init__(self, cfg: TeleoperationCommandCfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        joint_ids, joint_names = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        if tuple(joint_names) != tuple(cfg.joint_names):
            raise ValueError(
                "Teleoperation command joints did not resolve in the requested order. "
                f"Expected {tuple(cfg.joint_names)}, got {tuple(joint_names)}."
            )
        self.joint_ids = joint_ids
        self.joint_names = tuple(joint_names)
        self._joint_name_to_command_id = {
            name: index for index, name in enumerate(self.joint_names)
        }
        self._command = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self._command_start = self._command.clone()
        self._command_target = self._command.clone()
        self._command_interval = torch.ones(self.num_envs, device=self.device)
        lows, highs = [], []
        for name in self.joint_names:
            if name not in cfg.ranges:
                raise KeyError(f"No teleoperation range was configured for joint {name!r}.")
            low, high = cfg.ranges[name]
            if high < low:
                raise ValueError(f"Invalid teleoperation range for {name!r}: {(low, high)}.")
            lows.append(low)
            highs.append(high)
        self._range_low = torch.tensor(lows, device=self.device).unsqueeze(0)
        self._range_high = torch.tensor(highs, device=self.device).unsqueeze(0)
        self._source: NpzHugWBCCommandSource | None = None

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def command_ids_for_joint_names(self, joint_names: Sequence[str]) -> torch.Tensor:
        try:
            command_ids = [self._joint_name_to_command_id[name] for name in joint_names]
        except KeyError as exc:
            raise ValueError(f"Unknown teleoperation joint: {exc.args[0]!r}.") from exc
        return torch.tensor(command_ids, dtype=torch.long, device=self.device)

    def set_source(self, source: NpzHugWBCCommandSource) -> None:
        if tuple(source.teleop_joint_names) != self.joint_names:
            raise ValueError(
                "NPZ teleoperation order does not match the environment: "
                f"{source.teleop_joint_names} != {self.joint_names}."
            )
        self._source = source
        self._copy_source_frame()

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if self._source is not None:
            self._copy_source_frame()
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._command_start[env_ids] = self._command[env_ids]
        unit = torch.rand((env_ids.numel(), len(self.joint_names)), device=self.device)
        self._command_target[env_ids] = self._range_low + unit * (
            self._range_high - self._range_low
        )
        self._command_interval[env_ids] = self.time_left[env_ids].clamp_min(
            self._env.step_dt
        )

    def _update_command(self) -> None:
        if self._source is not None:
            self._copy_source_frame()
            return
        interval = self._command_interval.clamp_min(self._env.step_dt)
        elapsed = (interval - self.time_left).clamp_min(0.0)
        ratio = (
            self.cfg.interpolation_rate * elapsed / interval
        ).clamp(max=1.0).unsqueeze(-1)
        self._command[:] = torch.lerp(self._command_start, self._command_target, ratio)

    def _copy_source_frame(self) -> None:
        assert self._source is not None
        self._command.copy_(self._source.frame_at(self._simulation_time()).teleop_joint_pos)

    def _simulation_time(self) -> float:
        return float(self._env.common_step_counter) * float(self._env.step_dt)


@configclass
class TeleoperationCommandCfg(CommandTermCfg):
    """Configuration for :class:`TeleoperationCommand`."""

    class_type: type[CommandTerm] = TeleoperationCommand
    asset_name: str = "robot"
    joint_names: list[str] = []
    ranges: dict[str, tuple[float, float]] = {}
    interpolation_rate: float = 1.5


__all__ = [
    "HugWBCBaseVelocityCommand",
    "HugWBCBaseVelocityCommandCfg",
    "TeleoperationCommand",
    "TeleoperationCommandCfg",
]
