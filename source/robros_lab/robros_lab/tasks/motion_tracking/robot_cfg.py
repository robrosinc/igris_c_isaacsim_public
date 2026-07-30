"""Reusable IGRIS-C robot and robot-mounted sensor configurations."""

import math

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from robros_lab.assets import IGRIS_C_WRIST_HAND_INDEPENDENT_CFG

IGRIS_C_ROBOT_CFG = IGRIS_C_WRIST_HAND_INDEPENDENT_CFG.replace(
  prim_path="{ENV_REGEX_NS}/Robot"
)


def _quat_from_axis_angle(axis: tuple[float, float, float], angle_degrees: float) -> tuple[float, float, float, float]:
    """Convert an axis-angle rotation to a quaternion in ``(w, x, y, z)`` order."""

    axis_norm = math.sqrt(sum(component * component for component in axis))
    if axis_norm == 0.0:
        raise ValueError("Rotation axis must be non-zero.")

    half_angle = math.radians(angle_degrees) / 2.0
    scale = math.sin(half_angle) / axis_norm
    return (math.cos(half_angle), axis[0] * scale, axis[1] * scale, axis[2] * scale)


def _quat_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return the Hamilton product of two ``(w, x, y, z)`` quaternions."""

    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _camera_rotation(axis: tuple[float, float, float], angle_degrees: float) -> tuple[float, float, float, float]:
    """Apply a Link_Neck_Pitch-axis installation rotation to the base camera rotation."""

    base_rotation = (0.4056, -0.5792, 0.5792, -0.4056)
    installation_rotation = _quat_from_axis_angle(axis, angle_degrees)
    return _quat_multiply(installation_rotation, base_rotation)


IGRIS_C_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link_Neck_Pitch/HeadCamera",
    update_period=1.0 / 50.0,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0,
        focus_distance=4.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.112677, 0.0, 0.159748),
        rot=_camera_rotation((0.0, 1.0, 0.0), 54.0),
        convention="ros",
    ),
)

IGRIS_C_RIGHT_RGB_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link_Neck_Pitch/RightRgbCamera",
    update_period=1.0 / 50.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0,
        focus_distance=4.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.095516, -0.026076, 0.121000),
        rot=_camera_rotation((0.0, 0.0, 1.0), -14.0),
        convention="ros",
    ),
)

IGRIS_C_LEFT_RGB_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link_Neck_Pitch/LeftRgbCamera",
    update_period=1.0 / 50.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0,
        focus_distance=4.0,
        horizontal_aperture=20.955,
        clipping_range=(0.05, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.095516, 0.026076, 0.121000),
        rot=_camera_rotation((0.0, 0.0, 1.0), 14.0),
        convention="ros",
    ),
)


@configclass
class IGRISCRobotSceneCfg(InteractiveSceneCfg):
    """Reusable IGRIS-C robot with its mounted sensors."""

    robot = IGRIS_C_ROBOT_CFG
    head_camera = IGRIS_C_HEAD_CAMERA_CFG
    right_rgb_camera = IGRIS_C_RIGHT_RGB_CAMERA_CFG
    left_rgb_camera = IGRIS_C_LEFT_RGB_CAMERA_CFG
