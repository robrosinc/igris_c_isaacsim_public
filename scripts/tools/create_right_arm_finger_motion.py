"""Create an IGRIS-C right-arm and close/open finger sequence motion NPZ."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    REPOSITORY_ROOT
    / "source/robros_lab/robros_lab/assets/robots/robros/igris_c/from_urdf/urdf"
    / "igris_c_v2_wrist_hand_isaac.urdf"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "motion/right_arm_finger_sequence.npz"

FPS = 50.0
BASE_HEIGHT = 0.961
INITIAL_HOLD_SECONDS = 1.0
ARM_MOVE_SECONDS = 2.0
FINGER_MOVE_SECONDS = 1.0
CLOSED_HOLD_SECONDS = 1.0
FINAL_HOLD_SECONDS = 2.0

STUDENT_JOINT_NAMES = (
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Waist_Yaw",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Roll_Right",
    "Joint_Waist_Roll",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Waist_Pitch",
    "Joint_Elbow_Pitch_Left",
    "Joint_Elbow_Pitch_Right",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Left",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Left",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Left",
    "Joint_Ankle_Roll_Right",
)

BODY_NAMES = (
    "base_link",
    "Link_Neck_Yaw",
    "Link_Shoulder_Pitch_Left",
    "Link_Shoulder_Pitch_Right",
    "Link_Waist_Yaw",
    "Link_Neck_Pitch",
    "Link_Shoulder_Roll_Left",
    "Link_Shoulder_Roll_Right",
    "Link_Waist_Roll",
    "Link_Shoulder_Yaw_Left",
    "Link_Shoulder_Yaw_Right",
    "Link_Waist_Pitch",
    "Link_Elbow_Pitch_Left",
    "Link_Elbow_Pitch_Right",
    "Link_Hip_Pitch_Left",
    "Link_Hip_Pitch_Right",
    "Link_Wrist_Yaw_Left",
    "Link_Wrist_Yaw_Right",
    "Link_Hip_Roll_Left",
    "Link_Hip_Roll_Right",
    "Link_Wrist_Roll_Left",
    "Link_Wrist_Roll_Right",
    "Link_Hip_Yaw_Left",
    "Link_Hip_Yaw_Right",
    "Link_Wrist_Pitch_Left",
    "Link_Wrist_Pitch_Right",
    "Link_Knee_Pitch_Left",
    "Link_Knee_Pitch_Right",
    "Left_Hand",
    "Right_Hand",
    "Link_Ankle_Pitch_Left",
    "Link_Ankle_Pitch_Right",
    "Link_Ankle_Roll_Left",
    "Link_Ankle_Roll_Right",
)

WRIST_HAND_JOINT_NAMES = (
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
    "l_0_joint_thumb_proximal",
    "l_1_joint_thumb_middle",
    "l_3_joint_index_middle",
    "l_5_joint_middle_middle",
    "l_7_joint_ring_middle",
    "l_9_joint_little_middle",
    "r_0_joint_thumb_proximal",
    "r_1_joint_thumb_middle",
    "r_3_joint_index_middle",
    "r_5_joint_middle_middle",
    "r_7_joint_ring_middle",
    "r_9_joint_little_middle",
)


def _smooth_segment(times: np.ndarray, start: float, end: float, q0: float, q1: float) -> np.ndarray:
    fraction = np.clip((times - start) / (end - start), 0.0, 1.0)
    smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    return q0 + (q1 - q0) * smooth_fraction


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Revolute joint axis must be non-zero.")
    x, y, z = axis / norm
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale]
        )
    else:
        diagonal_index = int(np.argmax(np.diag(rotation)))
        next_index = (diagonal_index + 1) % 3
        last_index = (diagonal_index + 2) % 3
        scale = math.sqrt(
            1.0 + rotation[diagonal_index, diagonal_index]
            - rotation[next_index, next_index] - rotation[last_index, last_index]
        ) * 2.0
        quaternion = np.zeros(4, dtype=np.float64)
        quaternion[diagonal_index + 1] = 0.25 * scale
        quaternion[0] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / scale
        quaternion[next_index + 1] = (
            rotation[next_index, diagonal_index] + rotation[diagonal_index, next_index]
        ) / scale
        quaternion[last_index + 1] = (
            rotation[last_index, diagonal_index] + rotation[diagonal_index, last_index]
        ) / scale
    return quaternion / np.linalg.norm(quaternion)


def _parse_joints(urdf_path: Path) -> list[dict[str, object]]:
    root = ET.parse(urdf_path).getroot()
    joints: list[dict[str, object]] = []
    for element in root.findall("joint"):
        origin = element.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        axis_element = element.find("axis")
        axis = np.fromstring(
            axis_element.attrib.get("xyz", "1 0 0") if axis_element is not None else "1 0 0", sep=" "
        )
        limit = element.find("limit")
        limits = None if limit is None else (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        origin_transform = np.eye(4, dtype=np.float64)
        origin_transform[:3, :3] = _rpy_rotation(rpy)
        origin_transform[:3, 3] = xyz
        joints.append(
            {
                "name": element.attrib["name"],
                "type": element.attrib["type"],
                "parent": element.find("parent").attrib["link"],
                "child": element.find("child").attrib["link"],
                "axis": axis,
                "origin": origin_transform,
                "limits": limits,
            }
        )
    return joints


def _forward_kinematics(joints: list[dict[str, object]], positions: dict[str, float]) -> dict[str, np.ndarray]:
    transforms = {"base_link": np.eye(4, dtype=np.float64)}
    pending = list(joints)
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = str(joint["parent"])
            if parent not in transforms:
                continue
            angle = float(positions.get(str(joint["name"]), 0.0))
            limits = joint["limits"]
            if limits is not None and not limits[0] - 1.0e-6 <= angle <= limits[1] + 1.0e-6:
                raise ValueError(f"Joint {joint['name']} target {angle} exceeds URDF limits {limits}.")
            motion = np.eye(4, dtype=np.float64)
            if joint["type"] in ("revolute", "continuous"):
                motion[:3, :3] = _axis_rotation(np.asarray(joint["axis"]), angle)
            transforms[str(joint["child"])] = transforms[parent] @ np.asarray(joint["origin"]) @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            unresolved = [str(joint["name"]) for joint in pending]
            raise ValueError(f"URDF joint tree cannot be resolved from base_link: {unresolved}")
    return transforms


def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    arm_end = INITIAL_HOLD_SECONDS + ARM_MOVE_SECONDS
    finger_close_end = arm_end + 5.0 * FINGER_MOVE_SECONDS
    finger_open_start = finger_close_end + CLOSED_HOLD_SECONDS
    finger_open_end = finger_open_start + 5.0 * FINGER_MOVE_SECONDS
    arm_return_end = finger_open_end + ARM_MOVE_SECONDS
    final_time = arm_return_end + FINAL_HOLD_SECONDS
    times = np.arange(round(final_time * FPS) + 1, dtype=np.float64) / FPS

    initial_student = np.array(
        [0.13, 0.13, 0.0, 0.13, -0.13, 0.0, 0.0, 0.0, 0.0, -0.3, -0.3,
         -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, -0.1, -0.1, 0.0, 0.0],
        dtype=np.float64,
    )
    joint_pos = np.repeat(initial_student[None, :], len(times), axis=0)
    for joint_name in ("Joint_Shoulder_Pitch_Right", "Joint_Elbow_Pitch_Right"):
        index = STUDENT_JOINT_NAMES.index(joint_name)
        arm_position = _smooth_segment(
            times, INITIAL_HOLD_SECONDS, arm_end, initial_student[index], -math.pi / 2.0
        )
        return_position = _smooth_segment(
            times, finger_open_end, arm_return_end, -math.pi / 2.0, 0.0
        )
        return_mask = times >= finger_open_end
        arm_position[return_mask] = return_position[return_mask]
        joint_pos[:, index] = arm_position

    initial_wrist_hand = np.zeros(len(WRIST_HAND_JOINT_NAMES), dtype=np.float64)
    initial_wrist_hand[WRIST_HAND_JOINT_NAMES.index("Joint_Wrist_Roll_Left")] = math.pi / 6.0
    initial_wrist_hand[WRIST_HAND_JOINT_NAMES.index("Joint_Wrist_Roll_Right")] = -math.pi / 6.0
    wrist_hand_pos = np.repeat(initial_wrist_hand[None, :], len(times), axis=0)
    finger_targets = (
        ("r_1_joint_thumb_middle", math.radians(70.0)),
        ("r_3_joint_index_middle", math.pi / 2.0),
        ("r_5_joint_middle_middle", math.pi / 2.0),
        ("r_7_joint_ring_middle", math.pi / 2.0),
        ("r_9_joint_little_middle", math.pi / 2.0),
    )
    for finger_index, (joint_name, target) in enumerate(finger_targets):
        start = arm_end + finger_index * FINGER_MOVE_SECONDS
        end = start + FINGER_MOVE_SECONDS
        index = WRIST_HAND_JOINT_NAMES.index(joint_name)
        finger_position = _smooth_segment(times, start, end, 0.0, target)
        reverse_index = len(finger_targets) - 1 - finger_index
        open_start = finger_open_start + reverse_index * FINGER_MOVE_SECONDS
        open_end = open_start + FINGER_MOVE_SECONDS
        open_position = _smooth_segment(
            times, open_start, open_end, target, 0.0
        )
        open_mask = times >= open_start
        finger_position[open_mask] = open_position[open_mask]
        wrist_hand_pos[:, index] = finger_position

    dt = 1.0 / FPS
    joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=2)
    wrist_hand_vel = np.gradient(wrist_hand_pos, dt, axis=0, edge_order=2)

    joints = _parse_joints(args.urdf.resolve())
    body_pos_b = np.empty((len(times), len(BODY_NAMES), 3), dtype=np.float64)
    body_quat_b = np.empty((len(times), len(BODY_NAMES), 4), dtype=np.float64)
    for frame_index in range(len(times)):
        positions = dict(zip(STUDENT_JOINT_NAMES, joint_pos[frame_index]))
        positions.update(dict(zip(WRIST_HAND_JOINT_NAMES, wrist_hand_pos[frame_index])))
        transforms = _forward_kinematics(joints, positions)
        for body_index, body_name in enumerate(BODY_NAMES):
            transform = transforms[body_name]
            body_pos_b[frame_index, body_index] = transform[:3, 3]
            quaternion = _matrix_to_quaternion(transform[:3, :3])
            if frame_index > 0 and np.dot(quaternion, body_quat_b[frame_index - 1, body_index]) < 0.0:
                quaternion = -quaternion
            body_quat_b[frame_index, body_index] = quaternion

    body_pos_w = body_pos_b.copy()
    body_pos_w[..., 2] += BASE_HEIGHT
    body_quat_w = body_quat_b.copy()
    body_lin_vel_b = np.gradient(body_pos_b, dt, axis=0, edge_order=2)
    body_lin_vel_w = body_lin_vel_b.copy()
    quaternion_derivative = np.gradient(body_quat_b, dt, axis=0, edge_order=2)
    quaternion_conjugate = body_quat_b.copy()
    quaternion_conjugate[..., 1:] *= -1.0
    body_ang_vel_b = 2.0 * _quaternion_multiply(quaternion_conjugate, quaternion_derivative)[..., 1:]
    body_ang_vel_w = 2.0 * _quaternion_multiply(quaternion_derivative, quaternion_conjugate)[..., 1:]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        fps=np.float32(FPS),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        body_pos_b=body_pos_b.astype(np.float32),
        body_quat_b=body_quat_b.astype(np.float32),
        body_lin_vel_b=body_lin_vel_b.astype(np.float32),
        body_ang_vel_b=body_ang_vel_b.astype(np.float32),
        joint_names=np.asarray(STUDENT_JOINT_NAMES),
        body_names=np.asarray(BODY_NAMES),
        wrist_hand_joint_names=np.asarray(WRIST_HAND_JOINT_NAMES),
        wrist_hand_joint_pos=wrist_hand_pos.astype(np.float32),
        wrist_hand_joint_vel=wrist_hand_vel.astype(np.float32),
        clip_offsets=np.asarray([0, len(times)], dtype=np.int64),
        clip_names=np.asarray(["right_arm_finger_sequence"]),
        clip_weights=np.asarray([1.0], dtype=np.float32),
        local_frame_body_name=np.asarray("base_link"),
        source_motion_library=np.asarray(str(Path(__file__).resolve())),
    )
    print(f"Generated {args.output.resolve()} ({len(times)} frames, {final_time:.2f} s at {FPS:.1f} Hz)")


if __name__ == "__main__":
    main()
