"""Create a sample HugWBC command-v1 NPZ clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "motion/hugwbc_forward_arm_motion.npz"
FPS = 50.0
TELEOP_JOINT_NAMES = np.asarray(
    [
        "Joint_Waist_Yaw",
        "Joint_Waist_Roll",
        "Joint_Waist_Pitch",
        "Joint_Shoulder_Pitch_Left",
        "Joint_Shoulder_Roll_Left",
        "Joint_Shoulder_Yaw_Left",
        "Joint_Elbow_Pitch_Left",
        "Joint_Shoulder_Pitch_Right",
        "Joint_Shoulder_Roll_Right",
        "Joint_Shoulder_Yaw_Right",
        "Joint_Elbow_Pitch_Right",
    ]
)
TELEOP_RANGES = np.asarray(
    [
        (-1.5, 1.5),
        (-0.2, 0.2),
        (-1.0, 0.28),
        (-3.0, 1.0),
        (-0.15, 3.0),
        (-1.5, 1.5),
        (-2.1, 0.0),
        (-3.0, 1.0),
        (-3.0, 0.15),
        (-1.5, 1.5),
        (-2.1, 0.0),
    ],
    dtype=np.float32,
)
WRIST_HAND_JOINT_NAMES = np.asarray(
    [
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
    ]
)


def smooth_window(time: np.ndarray, start: float, end: float, edge: float = 0.5) -> np.ndarray:
    rise = np.clip((time - start) / edge, 0.0, 1.0)
    fall = np.clip((end - time) / edge, 0.0, 1.0)
    value = np.minimum(rise, fall)
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError(f"--duration must be positive and finite, got {args.duration}.")

    frame_count = int(round(FPS * args.duration)) + 1
    timeline = np.arange(frame_count, dtype=np.float32) / FPS
    base_velocity = np.zeros((frame_count, 3), dtype=np.float32)
    walk = smooth_window(timeline, start=2.0, end=min(8.0, args.duration), edge=1.0)
    base_velocity[:, 0] = 0.25 * walk

    teleop_joint_pos = np.zeros((frame_count, 11), dtype=np.float32)
    teleop_joint_pos[:, 3] = 0.13
    teleop_joint_pos[:, 4] = 0.13
    teleop_joint_pos[:, 6] = -0.30
    teleop_joint_pos[:, 7] = 0.13
    teleop_joint_pos[:, 8] = -0.13
    teleop_joint_pos[:, 10] = -0.30
    arm_motion = smooth_window(timeline, start=3.0, end=min(7.0, args.duration), edge=1.0)
    teleop_joint_pos[:, 3] -= 0.80 * arm_motion
    teleop_joint_pos[:, 6] -= 0.60 * arm_motion
    teleop_joint_pos[:, 7] -= 0.80 * arm_motion
    teleop_joint_pos[:, 10] -= 0.60 * arm_motion
    outside_range = np.logical_or(
        teleop_joint_pos < TELEOP_RANGES[:, 0],
        teleop_joint_pos > TELEOP_RANGES[:, 1],
    )
    if outside_range.any():
        indices = np.argwhere(outside_range)[:10].tolist()
        raise ValueError(f"Generated teleoperation targets exceed configured ranges: {indices}.")
    teleop_joint_vel = np.gradient(teleop_joint_pos, 1.0 / FPS, axis=0).astype(np.float32)

    wrist_hand_joint_pos = np.zeros((frame_count, 18), dtype=np.float32)
    hand_motion = smooth_window(
        timeline,
        start=4.0,
        end=min(8.0, args.duration),
        edge=1.0,
    )
    wrist_hand_targets = {
        "Joint_Wrist_Yaw_Left": 0.25,
        "Joint_Wrist_Roll_Left": 0.20,
        "Joint_Wrist_Pitch_Left": 0.20,
        "Joint_Wrist_Yaw_Right": -0.25,
        "Joint_Wrist_Roll_Right": -0.20,
        "Joint_Wrist_Pitch_Right": -0.20,
        "l_0_joint_thumb_proximal": 0.40,
        "l_1_joint_thumb_middle": 0.50,
        "l_3_joint_index_middle": 0.70,
        "l_5_joint_middle_middle": 0.70,
        "l_7_joint_ring_middle": 0.70,
        "l_9_joint_little_middle": 0.70,
        "r_0_joint_thumb_proximal": -0.40,
        "r_1_joint_thumb_middle": 0.50,
        "r_3_joint_index_middle": 0.70,
        "r_5_joint_middle_middle": 0.70,
        "r_7_joint_ring_middle": 0.70,
        "r_9_joint_little_middle": 0.70,
    }
    for joint_name, target in wrist_hand_targets.items():
        joint_index = int(np.flatnonzero(WRIST_HAND_JOINT_NAMES == joint_name)[0])
        wrist_hand_joint_pos[:, joint_index] = target * hand_motion
    wrist_hand_joint_vel = np.gradient(
        wrist_hand_joint_pos,
        1.0 / FPS,
        axis=0,
    ).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("igris_c_hugwbc_command_v1"),
        fps=np.asarray(FPS, dtype=np.float32),
        base_velocity=base_velocity,
        teleop_joint_names=TELEOP_JOINT_NAMES,
        teleop_joint_pos=teleop_joint_pos,
        teleop_joint_vel=teleop_joint_vel,
        wrist_hand_joint_names=WRIST_HAND_JOINT_NAMES,
        wrist_hand_joint_pos=wrist_hand_joint_pos,
        wrist_hand_joint_vel=wrist_hand_joint_vel,
        clip_name=np.asarray("hugwbc_forward_arm_motion"),
    )
    print(f"Wrote {args.output}: {frame_count} frames at {FPS:.1f} Hz")


if __name__ == "__main__":
    main()
