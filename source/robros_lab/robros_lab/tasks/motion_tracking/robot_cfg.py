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

HEAD_RGB_CAMERA_POS = (0.111295551, 0.034652642, 0.165355680)
HEAD_RGB_CAMERA_ROT = (0.229563798, -0.674830130, 0.671676295, -0.201880443)
HEAD_RGB_TO_DEPTH_POS = (-0.014685470778083286, 0.0003668845572944308, -0.0002340585078071427)
HEAD_RGB_TO_DEPTH_ROT = (0.9999799842648904, -0.0004686728602067, -0.0016442685933542, 0.0060916168734206)

HEAD_RGB_INTRINSIC = (
    (602.2489649456799, 0.0, 319.55365209726017),
    (0.0, 601.5595194939405, 237.2499324251401),
    (0.0, 0.0, 1.0),
)
HEAD_RGB_DISTORTION = (
    0.10335624665662596,
    -0.08003090389475187,
    0.0005443642820743403,
    0.0011010907306839444,
    -0.44996319866742235,
)
HEAD_DEPTH_INTRINSIC = (
    (387.26192055231036, 0.0, 324.67217967864474),
    (0.0, 387.260573425404, 233.78868310827266),
    (0.0, 0.0, 1.0),
)
HEAD_DEPTH_DISTORTION = (
    0.0057938321877094474,
    -0.01100853164565892,
    -0.0006863295006149436,
    -0.000711123750070492,
    0.006960487411602116,
)

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


def _normalize_quaternion(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in quat))
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    return tuple(component / norm for component in quat)


def _rotate_vector(
    quat: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    quat = _normalize_quaternion(quat)
    conjugate = (quat[0], -quat[1], -quat[2], -quat[3])
    rotated = _quat_multiply(_quat_multiply(quat, (0.0, *vector)), conjugate)
    return rotated[1], rotated[2], rotated[3]


def _compose_pose(
    parent_pos: tuple[float, float, float],
    parent_rot: tuple[float, float, float, float],
    child_pos: tuple[float, float, float],
    child_rot: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rotated_child_pos = _rotate_vector(parent_rot, child_pos)
    pos = tuple(parent + child for parent, child in zip(parent_pos, rotated_child_pos))
    rot = _normalize_quaternion(_quat_multiply(parent_rot, child_rot))
    return pos, rot


def _invert_pose(
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rot = _normalize_quaternion(rot)
    inverse_rot = (rot[0], -rot[1], -rot[2], -rot[3])
    rotated_pos = _rotate_vector(inverse_rot, pos)
    return tuple(-component for component in rotated_pos), inverse_rot


def _camera_rotation(axis: tuple[float, float, float], angle_degrees: float) -> tuple[float, float, float, float]:
    """Apply a Link_Neck_Pitch-axis installation rotation to the base camera rotation."""

    base_rotation = (0.5, -0.5, 0.5, -0.5)
    installation_rotation = _quat_from_axis_angle(axis, angle_degrees)
    return _quat_multiply(installation_rotation, base_rotation)


HEAD_DEPTH_TO_RGB_POS, HEAD_DEPTH_TO_RGB_ROT = _invert_pose(
    HEAD_RGB_TO_DEPTH_POS,
    HEAD_RGB_TO_DEPTH_ROT,
)
HEAD_DEPTH_CAMERA_POS, HEAD_DEPTH_CAMERA_ROT = _compose_pose(
    HEAD_RGB_CAMERA_POS,
    HEAD_RGB_CAMERA_ROT,
    HEAD_DEPTH_TO_RGB_POS,
    HEAD_DEPTH_TO_RGB_ROT,
)


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
        pos=HEAD_DEPTH_CAMERA_POS,
        rot=HEAD_DEPTH_CAMERA_ROT,
        convention="ros",
    ),
    depth_clipping_behavior="max",
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
    # H1's Left_Hand frame is coincident with its wrist-pitch frame. The public
    # hand URDF adds a fixed hand transform, so attach to the common wrist frame.
    prim_path="{ENV_REGEX_NS}/Robot/Link_Wrist_Pitch_Left/left_wrist_camera",
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
        pos=(-0.000267322741459, 0.00415895499953, -0.129840975238),
        rot=(0.699215363677, 0.714806849091, 0.0114143028865, 0.00433097922337),
        convention="ros",
    ),
)

IGRIS_C_RIGHT_WRIST_CAMERA_CFG = CameraCfg(
    # H1's Right_Hand frame is coincident with its wrist-pitch frame. The public
    # hand URDF adds a fixed hand transform, so attach to the common wrist frame.
    prim_path="{ENV_REGEX_NS}/Robot/Link_Wrist_Pitch_Right/right_wrist_camera",
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
        pos=(0.000267322741459, -0.00415895499953, -0.129840975238),
        rot=(-0.000770306955654, -0.00879481673276, 0.715300349799, 0.698761380876),
        convention="ros",
    ),
)


@configclass
class IGRISCRobotSceneCfg(InteractiveSceneCfg):
    """Reusable IGRIS-C robot with its mounted sensors."""

    robot = IGRIS_C_ROBOT_CFG
    head_camera = IGRIS_C_HEAD_CAMERA_CFG
    head_depth_camera = IGRIS_C_HEAD_DEPTH_CAMERA_CFG
    right_rgb_camera = IGRIS_C_RIGHT_RGB_CAMERA_CFG
    left_rgb_camera = IGRIS_C_LEFT_RGB_CAMERA_CFG
    left_wrist_camera = IGRIS_C_LEFT_WRIST_CAMERA_CFG
    right_wrist_camera = IGRIS_C_RIGHT_WRIST_CAMERA_CFG
