"""Observation terms required by the IGRIS-C Student Policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_shape,
)
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg


def _actuator_joint_pos_feedback(asset: Articulation) -> torch.Tensor:
    """Return actuator feedback, falling back to simulated joint positions."""
    joint_pos_feedback = asset.data.joint_pos.clone()
    for actuator in asset.actuators.values():
        actuator_feedback = getattr(actuator, "joint_pos_feedback", None)
        if actuator_feedback is None:
            continue

        initialized_mask = getattr(actuator, "_joint_pos_feedback_initialized", None)
        if initialized_mask is None:
            initialized_mask = torch.ones(asset.num_instances, dtype=torch.bool, device=asset.device)
        env_mask = initialized_mask.view(-1, 1)
        joint_indices = actuator.joint_indices

        if isinstance(joint_indices, slice):
            joint_pos_feedback = torch.where(env_mask, actuator_feedback, joint_pos_feedback)
        else:
            joint_pos_feedback[:, joint_indices] = torch.where(
                env_mask,
                actuator_feedback,
                joint_pos_feedback[:, joint_indices],
            )

    return joint_pos_feedback


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_feedback_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return actuator joint feedback relative to the default joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    feedback = _actuator_joint_pos_feedback(asset)
    return feedback[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


class finite_difference_obs(ManagerTermBase):
    """Differentiate an observation and optionally apply a first-order LPF."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if env.step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, but got {env.step_dt}.")

        self._dt = env.step_dt
        self._enable_lpf = bool(cfg.params.get("enable_lpf", False))
        self._lpf_alpha: float | None = None
        if self._enable_lpf:
            cutoff_frequency_hz = float(cfg.params.get("cutoff_frequency_hz", 0.0))
            if cutoff_frequency_hz <= 0.0:
                raise ValueError(
                    "cutoff_frequency_hz must be positive when LPF is enabled, "
                    f"but got {cutoff_frequency_hz}."
                )
            self._lpf_alpha = 1.0 - torch.exp(
                torch.tensor(-2.0 * torch.pi * cutoff_frequency_hz * env.step_dt)
            ).item()

        self._prev_obs: torch.Tensor | None = None
        self._filtered_obs: torch.Tensor | None = None
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._initialized[env_ids] = False
        if self._prev_obs is not None:
            self._prev_obs[env_ids] = 0.0
        if self._filtered_obs is not None:
            self._filtered_obs[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        obs_func: Callable[..., torch.Tensor],
        enable_lpf: bool = False,
        cutoff_frequency_hz: float = 0.0,
        observed_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del enable_lpf, cutoff_frequency_hz

        obs = obs_func(env, asset_cfg=observed_asset_cfg).clone()
        if self._prev_obs is None or self._prev_obs.shape != obs.shape:
            self._prev_obs = obs.clone()
            self._initialized.zero_()

        differentiated_obs = torch.zeros_like(obs)
        initialized_mask = self._initialized
        if torch.any(initialized_mask):
            differentiated_obs[initialized_mask] = (
                obs[initialized_mask] - self._prev_obs[initialized_mask]
            ) / self._dt
        self._prev_obs.copy_(obs)

        if not self._enable_lpf:
            self._initialized.fill_(True)
            return differentiated_obs

        if self._filtered_obs is None or self._filtered_obs.shape != differentiated_obs.shape:
            self._filtered_obs = differentiated_obs.clone()
            self._initialized.zero_()

        uninitialized_mask = ~self._initialized
        if torch.any(uninitialized_mask):
            self._filtered_obs[uninitialized_mask] = differentiated_obs[uninitialized_mask]
            self._initialized[uninitialized_mask] = True

        initialized_mask = ~uninitialized_mask
        if torch.any(initialized_mask):
            self._filtered_obs[initialized_mask] = (
                (1.0 - self._lpf_alpha) * self._filtered_obs[initialized_mask]
                + self._lpf_alpha * differentiated_obs[initialized_mask]
            )

        return self._filtered_obs.clone()
