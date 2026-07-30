from dataclasses import MISSING

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from .delayed_implicit import DelayedImplicitActuator


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 10
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""
