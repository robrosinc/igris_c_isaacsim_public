from isaaclab.utils import configclass

from .delayed_dc_motor_cfg import DelayedDCMotorCfg
from .delayed_dc_motor_with_slack import DelayedDCMotorWithSlack


@configclass
class DelayedDCMotorWithSlackCfg(DelayedDCMotorCfg):
    """Configuration for a delayed rated/peak actuator with feedback-side slack."""

    class_type: type = DelayedDCMotorWithSlack

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed."""

    slack_range: dict[str, tuple[float, float]] = {".*": (0.0, 0.0)}
    """Half-gap range in radians sampled once per reset for each matching joint."""
