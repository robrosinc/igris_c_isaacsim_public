"""Static checkpoint contract for the 23-joint IGRIS-C student policy."""

from __future__ import annotations


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
STUDENT_JOINT_COUNT = len(STUDENT_JOINT_NAMES)

if STUDENT_JOINT_COUNT != 23:
    raise RuntimeError(
        f"The IGRIS-C student checkpoint requires 23 joints, got {STUDENT_JOINT_COUNT}."
    )
if len(set(STUDENT_JOINT_NAMES)) != STUDENT_JOINT_COUNT:
    raise RuntimeError("STUDENT_JOINT_NAMES must contain unique joint names.")


__all__ = ["STUDENT_JOINT_COUNT", "STUDENT_JOINT_NAMES"]
