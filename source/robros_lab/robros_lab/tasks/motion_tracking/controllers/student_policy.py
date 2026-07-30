"""Standalone deterministic inference for the IGRIS-C tracking student."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from ..student_contract import STUDENT_JOINT_COUNT
from .base import EnvIds


STUDENT_OBSERVATION_DIM = 1601
STUDENT_HIDDEN_DIMS = (512, 512, 256, 128)


class _EmpiricalNormalization(nn.Module):
    """Inference-only equivalent of the RSL empirical normalizer."""

    def __init__(self, observation_dim: int, eps: float = 1.0e-2) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(1, observation_dim))
        self.register_buffer("_var", torch.ones(1, observation_dim))
        self.register_buffer("_std", torch.ones(1, observation_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self._mean) / (self._std + self.eps)


class _GaussianDistributionState(nn.Module):
    """Checkpoint-compatible state for deterministic Gaussian output."""

    def __init__(self, action_dim: int, init_std: float = 0.001) -> None:
        super().__init__()
        self.std_param = nn.Parameter(init_std * torch.ones(action_dim))


class _StudentNetwork(nn.Module):
    """Exact deterministic subset of the training-time RSL MLP model."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.obs_normalizer = _EmpiricalNormalization(observation_dim)
        self.distribution = _GaussianDistributionState(action_dim)

        layer_dims = (observation_dim, *hidden_dims, action_dim)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(layer_dims[:-2], layer_dims[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized_observation = self.obs_normalizer(observation)
        return self.mlp(normalized_observation)


class StudentPolicy:
    """Load the student checkpoint and run its deterministic 23-action policy.

    Only the student model is constructed. Teacher, optimizer, rollout storage,
    and training state from the distillation checkpoint are intentionally not
    instantiated. The checkpoint observation normalizer is restored as part of
    the strict student state-dict load.

    Checkpoints are PyTorch pickle files and must therefore come from a trusted
    source.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str,
        expected_observation_dim: int = STUDENT_OBSERVATION_DIM,
        expected_action_dim: int = STUDENT_JOINT_COUNT,
        hidden_dims: Sequence[int] = STUDENT_HIDDEN_DIMS,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Student checkpoint does not exist: {self.checkpoint_path}")

        self.device = torch.device(device)
        self.expected_observation_dim = int(expected_observation_dim)
        self.expected_action_dim = int(expected_action_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self._validate_constructor_contract()

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = self._extract_student_state_dict(checkpoint)
        self._validate_checkpoint_layout(state_dict)

        checkpoint_dtype = state_dict["mlp.0.weight"].dtype
        model = _StudentNetwork(
            observation_dim=self.expected_observation_dim,
            action_dim=self.expected_action_dim,
            hidden_dims=list(self.hidden_dims),
        ).to(device=self.device, dtype=checkpoint_dtype)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._policy = model
        self.dtype = checkpoint_dtype
        self.iteration = checkpoint.get("iter") if isinstance(checkpoint, dict) else None

    @property
    def observation_dim(self) -> int:
        """Flattened student observation dimension expected by the checkpoint."""

        return self.expected_observation_dim

    @property
    def action_dim(self) -> int:
        """Number of normalized joint-position actions produced by the policy."""

        return self.expected_action_dim

    @torch.inference_mode()
    def infer(self, student_observation: torch.Tensor) -> torch.Tensor:
        """Return deterministic actions for a batched flattened observation."""

        self._validate_observation(student_observation)
        action = self._policy(student_observation)
        expected_shape = (student_observation.shape[0], self.expected_action_dim)
        if tuple(action.shape) != expected_shape:
            raise RuntimeError(
                f"Student policy returned shape {tuple(action.shape)}, expected {expected_shape}."
            )
        return action

    def reset(self, env_ids: EnvIds = None) -> None:
        """Reset policy state.

        The current student is an MLP and has no recurrent state. The method is
        retained so a runner can use the same lifecycle for future policies.
        """

        del env_ids

    def _validate_constructor_contract(self) -> None:
        if self.expected_observation_dim <= 0:
            raise ValueError(
                f"expected_observation_dim must be positive, got {self.expected_observation_dim}."
            )
        if self.expected_action_dim != STUDENT_JOINT_COUNT:
            raise ValueError(
                f"The checkpoint requires {STUDENT_JOINT_COUNT} actions, "
                f"got {self.expected_action_dim}."
            )
        if self.hidden_dims != STUDENT_HIDDEN_DIMS:
            raise ValueError(
                f"The checkpoint requires hidden dimensions {STUDENT_HIDDEN_DIMS}, "
                f"got {self.hidden_dims}."
            )

    @staticmethod
    def _extract_student_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Student checkpoint must contain a dictionary, got {type(checkpoint).__name__}."
            )
        state_dict = checkpoint.get("student_state_dict")
        if not isinstance(state_dict, dict):
            raise KeyError("Checkpoint does not contain a student_state_dict dictionary.")
        if not state_dict:
            raise ValueError("Checkpoint student_state_dict is empty.")
        if not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state_dict.items()):
            raise TypeError("student_state_dict must map string names to torch.Tensor values.")
        return state_dict

    def _validate_checkpoint_layout(self, state_dict: dict[str, torch.Tensor]) -> None:
        expected_shapes = {
            "obs_normalizer._mean": (1, self.expected_observation_dim),
            "obs_normalizer._var": (1, self.expected_observation_dim),
            "obs_normalizer._std": (1, self.expected_observation_dim),
            "distribution.std_param": (self.expected_action_dim,),
            "mlp.0.weight": (self.hidden_dims[0], self.expected_observation_dim),
            "mlp.0.bias": (self.hidden_dims[0],),
            "mlp.2.weight": (self.hidden_dims[1], self.hidden_dims[0]),
            "mlp.2.bias": (self.hidden_dims[1],),
            "mlp.4.weight": (self.hidden_dims[2], self.hidden_dims[1]),
            "mlp.4.bias": (self.hidden_dims[2],),
            "mlp.6.weight": (self.hidden_dims[3], self.hidden_dims[2]),
            "mlp.6.bias": (self.hidden_dims[3],),
            "mlp.8.weight": (self.expected_action_dim, self.hidden_dims[3]),
            "mlp.8.bias": (self.expected_action_dim,),
        }
        missing = tuple(name for name in expected_shapes if name not in state_dict)
        if missing:
            raise KeyError(f"Student checkpoint is missing required tensors: {missing}.")

        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(state_dict[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Checkpoint tensor {name!r} has shape {actual_shape}, "
                    f"expected {expected_shape}."
                )

        weight_dtype = state_dict["mlp.0.weight"].dtype
        if not weight_dtype.is_floating_point:
            raise TypeError(f"Student checkpoint weights must be floating-point, got {weight_dtype}.")
        for name, value in state_dict.items():
            if name == "obs_normalizer.count":
                continue
            if value.is_floating_point() and value.dtype != weight_dtype:
                raise TypeError(
                    f"Checkpoint tensor {name!r} uses {value.dtype}, expected {weight_dtype}."
                )

    def _validate_observation(self, observation: torch.Tensor) -> None:
        if not isinstance(observation, torch.Tensor):
            raise TypeError(
                f"student_observation must be a torch.Tensor, got {type(observation).__name__}."
            )
        if observation.ndim != 2 or observation.shape[-1] != self.expected_observation_dim:
            raise ValueError(
                "student_observation must have shape "
                f"(num_envs, {self.expected_observation_dim}), got {tuple(observation.shape)}."
            )
        if observation.device != self.device:
            raise ValueError(
                f"student_observation is on {observation.device}, expected {self.device}."
            )
        if observation.dtype != self.dtype:
            raise TypeError(
                f"student_observation has dtype {observation.dtype}, expected {self.dtype}."
            )
