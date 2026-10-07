"""Reusable IGRIS-C robot and robot-mounted sensor configurations."""

import math

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from robros_lab.assets import IGRIS_C_WRIST_HAND_INDEPENDENT_CFG

from .camera_cfg import calibrated_pinhole_camera_cfg

IGRIS_C_ROBOT_CFG = IGRIS_C_WRIST_HAND_INDEPENDENT_CFG.replace(
  prim_path="{ENV_REGEX_NS}/Robot"
)

HEAD_RGB_CAMERA_POS = (0.112677, 0.0325, 0.159748)
HEAD_LEFT_IR_CAMERA_POS = (0.112677, 0.0175, 0.159748)
HEAD_RIGHT_IR_CAMERA_POS = (0.112677, -0.0325, 0.159748)

HEAD_RGB_INTRINSIC = (
    (597.0, 0.0, 320.0),
    (0.0, 597.0, 240.0),
    (0.0, 0.0, 1.0),
)
HEAD_RGB_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)
HEAD_DEPTH_INTRINSIC = (
    (386.0, 0.0, 320.0),
    (0.0, 386.0, 240.0),
    (0.0, 0.0, 1.0),
)
HEAD_DEPTH_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)

LEFT_WRIST_INTRINSIC = (
    (115.669608, 0.0, 159.669481),
    (0.0, 115.546655, 96.948254),
    (0.0, 0.0, 1.0),
)
LEFT_WRIST_DISTORTION = (0.047115, -0.056155, 0.0, 0.0, 0.012162)
RIGHT_WRIST_INTRINSIC = (
    (115.459869, 0.0, 157.159381),
    (0.0, 115.422582, 91.929442),
    (0.0, 0.0, 1.0),
)
RIGHT_WRIST_DISTORTION = (0.044201, -0.057607, -0.000818, 0.000326, 0.012698)

# Both D435 optical centers are inside the neck-pitch render mesh. A 60 mm
# near plane clears that mesh only from the D435 render products while keeping
# the face visible to the eye cameras and the GUI viewport.
HEAD_CAMERA_CLIPPING_RANGE = (0.06, 4.0)


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

    base_rotation = (0.5, -0.5, 0.5, -0.5)
    installation_rotation = _quat_from_axis_angle(axis, angle_degrees)
    return _quat_multiply(installation_rotation, base_rotation)


HEAD_RGB_CAMERA_ROT = _camera_rotation((0.0, 1.0, 0.0), 54.0)


IGRIS_C_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link_Neck_Pitch/d435_camera",
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=calibrated_pinhole_camera_cfg(
        intrinsic_matrix=HEAD_RGB_INTRINSIC,
        distortion=HEAD_RGB_DISTORTION,
        width=640,
        height=480,
        clipping_range=HEAD_CAMERA_CLIPPING_RANGE,
    ),
    offset=CameraCfg.OffsetCfg(
        pos=HEAD_RGB_CAMERA_POS,
        rot=HEAD_RGB_CAMERA_ROT,
        convention="ros",
    ),
)

IGRIS_C_HEAD_DEPTH_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link_Neck_Pitch/d435_depth_camera",
    height=480,
    width=640,
    data_types=["depth"],
    spawn=calibrated_pinhole_camera_cfg(
        intrinsic_matrix=HEAD_DEPTH_INTRINSIC,
        distortion=HEAD_DEPTH_DISTORTION,
        width=640,
        height=480,
        clipping_range=HEAD_CAMERA_CLIPPING_RANGE,
    ),
    offset=CameraCfg.OffsetCfg(
        pos=HEAD_LEFT_IR_CAMERA_POS,
        rot=HEAD_RGB_CAMERA_ROT,
        convention="ros",
    ),
    depth_clipping_behavior="max",
)

IGRIS_C_HEAD_RIGHT_IR_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link_Neck_Pitch/d435_right_ir_camera",
    height=480,
    width=640,
    # Isaac Lab renders visible RGB here as a proxy for the right IR viewpoint.
    data_types=["rgb"],
    spawn=calibrated_pinhole_camera_cfg(
        intrinsic_matrix=HEAD_DEPTH_INTRINSIC,
        distortion=HEAD_DEPTH_DISTORTION,
        width=640,
        height=480,
        clipping_range=HEAD_CAMERA_CLIPPING_RANGE,
    ),
    offset=CameraCfg.OffsetCfg(
        pos=HEAD_RIGHT_IR_CAMERA_POS,
        rot=HEAD_RGB_CAMERA_ROT,
        convention="ros",
    ),
)

IGRIS_C_RIGHT_RGB_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link_Neck_Pitch/RightRgbCamera",
    update_period=1.0 / 30.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[433.3, 0.0, 320.0, 0.0, 433.3, 240.0, 0.0, 0.0, 1.0],
        width=640,
        height=480,
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
    update_period=1.0 / 30.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[433.3, 0.0, 320.0, 0.0, 433.3, 240.0, 0.0, 0.0, 1.0],
        width=640,
        height=480,
        clipping_range=(0.05, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.095516, 0.026076, 0.121000),
        rot=_camera_rotation((0.0, 0.0, 1.0), 14.0),
        convention="ros",
    ),
)

IGRIS_C_LEFT_WRIST_CAMERA_CFG = CameraCfg(
    # URDF l_palm is represented by the public asset's Left_Hand link.
    prim_path="{ENV_REGEX_NS}/Robot/Left_Hand/l_hand_camera_link",
    height=200,
    width=320,
    data_types=["rgb"],
    spawn=calibrated_pinhole_camera_cfg(
        intrinsic_matrix=LEFT_WRIST_INTRINSIC,
        distortion=LEFT_WRIST_DISTORTION,
        width=320,
        height=200,
        clipping_range=(0.001, 4.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(-0.024420, 0.010317, 0.034793),
        # URDF camera_link rpy=(-2.7823, -0.5795, 0.9576), converted to (w, x, y, z).
        rot=(0.281491451510162, -0.813343087073706, -0.479715245579771, -0.170613213422809),
        convention="world",
    ),
)

IGRIS_C_RIGHT_WRIST_CAMERA_CFG = CameraCfg(
    # URDF r_palm is represented by the public asset's Right_Hand link.
    prim_path="{ENV_REGEX_NS}/Robot/Right_Hand/r_hand_camera_link",
    height=200,
    width=320,
    data_types=["rgb"],
    spawn=calibrated_pinhole_camera_cfg(
        intrinsic_matrix=RIGHT_WRIST_INTRINSIC,
        distortion=RIGHT_WRIST_DISTORTION,
        width=320,
        height=200,
        clipping_range=(0.001, 4.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(-0.024420, -0.010317, 0.034793),
        # URDF camera_link rpy=(2.7845, -0.5769, -0.9588), converted to (w, x, y, z).
        rot=(0.280192038750642, 0.813790055664715, -0.479987262874607, 0.169854037927043),
        convention="world",
    ),
)


@configclass
class IGRISCRobotSceneCfg(InteractiveSceneCfg):
    """Reusable IGRIS-C robot with its mounted sensors."""

    robot = IGRIS_C_ROBOT_CFG
    head_camera = IGRIS_C_HEAD_CAMERA_CFG
    head_depth_camera = IGRIS_C_HEAD_DEPTH_CAMERA_CFG
    head_right_ir_camera = IGRIS_C_HEAD_RIGHT_IR_CAMERA_CFG
    right_rgb_camera = IGRIS_C_RIGHT_RGB_CAMERA_CFG
    left_rgb_camera = IGRIS_C_LEFT_RGB_CAMERA_CFG
    left_wrist_camera = IGRIS_C_LEFT_WRIST_CAMERA_CFG
    right_wrist_camera = IGRIS_C_RIGHT_WRIST_CAMERA_CFG
