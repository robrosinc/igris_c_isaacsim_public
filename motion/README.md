# Motion Dataset

This directory contains NPZ reference motions used by the IGRIS-C Motion Tracking task.
The 23 Student Policy joints remain compatible with the existing policy, while an
optional auxiliary channel can provide wrist and active-finger targets.

For project installation and general execution instructions, see the
[project README](../README.md). For the controller data flow and complete NPZ schema,
see the [Motion Tracking task README](../source/robros_lab/robros_lab/tasks/motion_tracking/README.md).

## Dataset Catalog

| File | FPS | Frames | Duration | Policy joints | Auxiliary joints | Description |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `recorded_arms.npz` | 50 | 362 | 7.22 s | 23 | None | Recorded upper-body/arm motion |
| `recorded_walk_2.npz` | 50 | 243 | 4.84 s | 23 | None | Recorded walking motion |
| `right_arm_finger_sequence.npz` | 50 | 901 | 18.00 s | 23 | 18 | Generated right-arm and sequential finger motion |

Duration is calculated as `(frame_count - 1) / fps`.

The two `recorded_*.npz` files do not contain auxiliary wrist or hand targets. During
their playback, the auxiliary controller therefore holds its configured default wrist
and open-hand pose. `right_arm_finger_sequence.npz` contains the optional auxiliary
channel and overrides those defaults for the joints listed in the file.

## Playback

Run commands from the repository root after completing the project installation.

### Recorded arm motion

```bash
./third_party/IsaacLab/isaaclab.sh -p \
  scripts/motion_tracking/igris_c_motion_tracking.py \
  --motion_file motion/recorded_arms.npz \
  --num_envs 1
```

### Recorded walking motion

```bash
./third_party/IsaacLab/isaaclab.sh -p \
  scripts/motion_tracking/igris_c_motion_tracking.py \
  --motion_file motion/recorded_walk_2.npz \
  --num_envs 1
```

### Right-arm and finger sequence

```bash
./third_party/IsaacLab/isaaclab.sh -p \
  scripts/motion_tracking/igris_c_motion_tracking.py \
  --motion_file motion/right_arm_finger_sequence.npz \
  --num_envs 1
```

Useful playback options:

- Omit `--motion_file` to keep sending the initial standing pose.
- Add `--loop` to repeat the selected motion.
- Add `--stop_at_end` to close the simulation when playback finishes.
- Without either option, the final frame is held after the motion finishes.
- Increase `--num_envs` to run multiple environments with the same reference motion.

The current player selects frames using `floor(elapsed_time * fps)` and does not
interpolate between frames. Use a suitable sampling rate when generating a new motion.

## Right-Arm and Finger Sequence

`right_arm_finger_sequence.npz` performs the following 18-second sequence:

| Time | Motion |
| --- | --- |
| 0–1 s | Hold the initial standing pose |
| 1–3 s | Move the right shoulder pitch and elbow pitch to -90 degrees |
| 3–4 s | Close the right thumb |
| 4–5 s | Close the right index finger |
| 5–6 s | Close the right middle finger |
| 6–7 s | Close the right ring finger |
| 7–8 s | Close the right little finger |
| 8–9 s | Hold the closed hand |
| 9–10 s | Open the right little finger |
| 10–11 s | Open the right ring finger |
| 11–12 s | Open the right middle finger |
| 12–13 s | Open the right index finger |
| 13–14 s | Open the right thumb |
| 14–16 s | Return the right shoulder and elbow to 0 degrees |
| 16–18 s | Hold the final standing pose |

Its initial wrist-roll targets are +30 degrees for the left wrist and -30 degrees for
the right wrist. Finger targets are otherwise initialized to the open position.

Regenerate the file with the repository's generator:

```bash
python3 scripts/tools/create_right_arm_finger_motion.py
```

To write a separate file instead of replacing the default output:

```bash
python3 scripts/tools/create_right_arm_finger_motion.py \
  --output motion/custom_right_arm_finger_sequence.npz
```

The generator reads the robot URDF, computes forward-kinematics body data, and writes
joint velocities from the generated position trajectory. Do not replace these fields
with arbitrary placeholder values.

## NPZ Structure

Every dataset in this directory contains the following common entries:

| Key | Purpose |
| --- | --- |
| `fps` | Playback frequency in Hz |
| `joint_pos`, `joint_vel` | Reference positions and velocities for the 23 policy joints |
| `joint_names` | Column names and ordering for the policy-joint arrays |
| `body_pos_w`, `body_quat_w` | Body poses in the world frame |
| `body_lin_vel_w`, `body_ang_vel_w` | Body velocities in the world frame |
| `body_pos_b`, `body_quat_b` | Body poses in the local body frame |
| `body_lin_vel_b`, `body_ang_vel_b` | Body velocities in the local body frame |
| `body_names` | Column names and ordering for body arrays |
| `clip_offsets`, `clip_names`, `clip_weights` | Clip boundaries and metadata |
| `local_frame_body_name` | Body used as the local-frame reference |
| `source_motion_library` | Provenance metadata for the source motion |

A motion with wrist or hand control must additionally provide all three auxiliary
entries:

| Key | Shape | Purpose |
| --- | --- | --- |
| `wrist_hand_joint_names` | `(A,)` | Names and ordering of auxiliary joints |
| `wrist_hand_joint_pos` | `(T, A)` | Auxiliary position targets in radians |
| `wrist_hand_joint_vel` | `(T, A)` | Auxiliary velocity targets in radians per second |

`T` is the number of frames and `A` is the number of auxiliary joints. The current
right-arm finger dataset uses 18 auxiliary joints.

The auxiliary channel stores active finger targets, not independently authored distal
targets. At runtime, `AuxiliaryJointController` calculates each distal target from its
corresponding middle-joint target and the configured coupling ratio. This guarantees
consistent coupled motion even when an NPZ file is provided by another user.

`source_motion_library` is provenance metadata only. Older recorded files may contain
an absolute path from the machine on which they were created; that path is not required
to replay the NPZ file.

## Inspecting a Dataset

Use NumPy to check keys and array shapes without starting Isaac Sim:

```bash
python3 - <<'PY'
import numpy as np

path = "motion/right_arm_finger_sequence.npz"
with np.load(path, allow_pickle=False) as data:
    print("keys:", data.files)
    print("fps:", float(data["fps"]))
    for key in ("joint_pos", "joint_vel", "body_pos_w"):
        print(f"{key}: {data[key].shape}")
    if "wrist_hand_joint_names" in data:
        print("auxiliary joints:", data["wrist_hand_joint_names"].tolist())
        print("auxiliary positions:", data["wrist_hand_joint_pos"].shape)
PY
```

## Adding a Motion

Before adding or replacing an NPZ file, verify the following:

1. `joint_names` exactly identifies every column in `joint_pos` and `joint_vel`.
2. The 23 policy-joint names remain compatible with the Student Policy checkpoint.
3. All time-series arrays have the same frame count and contain finite values.
4. Joint positions and velocities use radians and radians per second.
5. Body quaternion arrays are normalized and use the ordering expected by the task.
6. Auxiliary data includes all three auxiliary keys or none of them.
7. Active middle-joint targets are stored for coupled fingers; distal targets are left
   to `AuxiliaryJointController`.
8. Joint targets remain inside the robot's URDF limits.
9. The file is tested once with `--stop_at_end` and once without it to verify both
   termination and final-frame hold behavior.

NPZ files are binary datasets. Prefer regenerating them from a source trajectory or a
documented generator instead of editing individual values manually.
