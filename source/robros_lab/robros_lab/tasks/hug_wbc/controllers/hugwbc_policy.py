"""Deterministic loader for the HugWBC actor checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn


HUGWBC_OBSERVATION_DIM = 280
HUGWBC_ACTION_DIM = 15
HUGWBC_HIDDEN_DIMS = (512, 256, 128)
LOWER_BODY_SYMMETRY_OBSERVATION_DIM = 700
LOWER_BODY_SYMMETRY_ACTION_DIM = 13
LOWER_BODY_SYMMETRY_HIDDEN_DIMS = (256, 256, 256)


class _EmpiricalNormalization(nn.Module):
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
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.std_param = nn.Parameter(torch.ones(action_dim))


class _HugWBCActor(nn.Module):
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
        return self.mlp(self.obs_normalizer(observation))


class HugWBCPolicy:
    """Strictly load a supported HugWBC actor and return its deterministic mean."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str,
        expected_observation_dim: int | None = None,
        expected_action_dim: int | None = None,
        hidden_dims: Sequence[int] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"HugWBC checkpoint does not exist: {self.checkpoint_path}")
        self.device = torch.device(device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = self._extract_actor_state_dict(checkpoint)
        checkpoint_observation_dim, checkpoint_action_dim, checkpoint_hidden_dims = (
            self._infer_checkpoint_layout(state_dict)
        )
        if (
            expected_observation_dim is not None
            and int(expected_observation_dim) != checkpoint_observation_dim
        ):
            raise ValueError(
                f"Checkpoint has {checkpoint_observation_dim} observations, "
                f"expected {int(expected_observation_dim)}."
            )
        if expected_action_dim is not None and int(expected_action_dim) != checkpoint_action_dim:
            raise ValueError(
                f"Checkpoint has {checkpoint_action_dim} actions, "
                f"expected {int(expected_action_dim)}."
            )
        if hidden_dims is not None:
            requested_hidden_dims = tuple(int(dim) for dim in hidden_dims)
            if requested_hidden_dims != checkpoint_hidden_dims:
                raise ValueError(
                    f"Checkpoint hidden dimensions are {checkpoint_hidden_dims}, "
                    f"expected {requested_hidden_dims}."
                )
        self.expected_observation_dim = checkpoint_observation_dim
        self.expected_action_dim = checkpoint_action_dim
        self.hidden_dims = checkpoint_hidden_dims
        self._validate_checkpoint_layout(state_dict)
        checkpoint_dtype = state_dict["mlp.0.weight"].dtype
        model = _HugWBCActor(
            self.expected_observation_dim,
            self.expected_action_dim,
            self.hidden_dims,
        ).to(device=self.device, dtype=checkpoint_dtype)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._policy = model
        self.dtype = checkpoint_dtype
        self.iteration = checkpoint.get("iter")

    @property
    def observation_dim(self) -> int:
        return self.expected_observation_dim

    @property
    def action_dim(self) -> int:
        return self.expected_action_dim

    @torch.inference_mode()
    def infer(self, observation: torch.Tensor) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor):
            raise TypeError(f"observation must be a torch.Tensor, got {type(observation).__name__}.")
        expected_shape = (observation.shape[0], self.observation_dim) if observation.ndim else None
        if observation.ndim != 2 or tuple(observation.shape) != expected_shape:
            raise ValueError(
                f"observation must have shape (num_envs, {self.observation_dim}), "
                f"got {tuple(observation.shape)}."
            )
        if observation.device != self.device:
            raise ValueError(f"observation is on {observation.device}, expected {self.device}.")
        if observation.dtype != self.dtype:
            raise TypeError(f"observation has dtype {observation.dtype}, expected {self.dtype}.")
        action = self._policy(observation)
        expected_action_shape = (observation.shape[0], self.action_dim)
        if tuple(action.shape) != expected_action_shape:
            raise RuntimeError(
                f"HugWBC actor returned {tuple(action.shape)}, expected {expected_action_shape}."
            )
        if not torch.isfinite(action).all().item():
            raise RuntimeError("HugWBC actor returned NaN or Inf.")
        return action

    @staticmethod
    def _extract_actor_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError("HugWBC checkpoint must contain a dictionary.")
        state_dict = checkpoint.get("actor_state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise KeyError("Checkpoint does not contain a non-empty actor_state_dict.")
        if not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state_dict.items()
        ):
            raise TypeError("actor_state_dict must map string names to torch.Tensor values.")
        return state_dict

    @staticmethod
    def _infer_checkpoint_layout(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[int, int, tuple[int, ...]]:
        weight_names = ("mlp.0.weight", "mlp.2.weight", "mlp.4.weight", "mlp.6.weight")
        missing = [name for name in weight_names if name not in state_dict]
        if missing:
            raise KeyError(f"HugWBC actor is missing MLP weights: {missing}.")
        weights = [state_dict[name] for name in weight_names]
        if any(weight.ndim != 2 for weight in weights):
            raise ValueError("HugWBC MLP weights must all be rank-two tensors.")
        observation_dim = int(weights[0].shape[1])
        hidden_dims = tuple(int(weight.shape[0]) for weight in weights[:-1])
        action_dim = int(weights[-1].shape[0])
        return observation_dim, action_dim, hidden_dims

    def _validate_checkpoint_layout(self, state_dict: dict[str, torch.Tensor]) -> None:
        dims = self.hidden_dims
        expected_shapes = {
            "obs_normalizer._mean": (1, self.observation_dim),
            "obs_normalizer._var": (1, self.observation_dim),
            "obs_normalizer._std": (1, self.observation_dim),
            "obs_normalizer.count": (),
            "distribution.std_param": (self.action_dim,),
            "mlp.0.weight": (dims[0], self.observation_dim),
            "mlp.0.bias": (dims[0],),
            "mlp.2.weight": (dims[1], dims[0]),
            "mlp.2.bias": (dims[1],),
            "mlp.4.weight": (dims[2], dims[1]),
            "mlp.4.bias": (dims[2],),
            "mlp.6.weight": (self.action_dim, dims[2]),
            "mlp.6.bias": (self.action_dim,),
        }
        actual_keys = set(state_dict)
        expected_keys = set(expected_shapes)
        if actual_keys != expected_keys:
            raise KeyError(
                "Unexpected HugWBC actor state layout. "
                f"Missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}."
            )
        for name, expected_shape in expected_shapes.items():
            if tuple(state_dict[name].shape) != expected_shape:
                raise ValueError(
                    f"Checkpoint tensor {name!r} has shape {tuple(state_dict[name].shape)}, "
                    f"expected {expected_shape}."
                )


__all__ = [
    "HUGWBC_ACTION_DIM",
    "HUGWBC_HIDDEN_DIMS",
    "HUGWBC_OBSERVATION_DIM",
    "LOWER_BODY_SYMMETRY_ACTION_DIM",
    "LOWER_BODY_SYMMETRY_HIDDEN_DIMS",
    "LOWER_BODY_SYMMETRY_OBSERVATION_DIM",
    "HugWBCPolicy",
]
