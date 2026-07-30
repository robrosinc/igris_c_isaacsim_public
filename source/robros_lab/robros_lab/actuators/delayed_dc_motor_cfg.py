from dataclasses import MISSING

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils import configclass

from .delayed_dc_motor import DelayedDCMotor


@configclass
class DelayedDCMotorCfg(IdealPDActuatorCfg):
    """Configuration for a delayed actuator with a rated/peak torque-speed envelope."""

    class_type: type = DelayedDCMotor

    velocity_limit: dict[str, float] | float = MISSING
    """Hard joint speed cap used by the actuator model and, by default, the solver."""

    velocity_rated: dict[str, float] | float = MISSING
    """Speed below which overload torque tapers from torque_rated to torque_limit."""

    torque_rated: dict[str, float] | float = MISSING
    """Continuous joint torque available throughout the normal operating speed range."""

    torque_limit: dict[str, float] | float = MISSING
    """Short-duration joint torque available at zero speed."""

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed."""
