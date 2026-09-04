# IGRIS-C Motion Tracking Task

Inference-only Isaac Lab task for replaying a standing or NPZ motion reference with the IGRIS-C Student Policy. The task preserves the checkpoint's original 23-joint contract and controls the additional wrist and articulated-hand joints through an independent auxiliary controller.

See the repository [README](../../../../../README.md) for installation and execution commands.

## Task summary

| Item | Value |
| --- | --- |
| Gym task ID | `Isaac-IGRISC-Motion-Tracking-v0` |
| Environment | `IGRISCMotionTrackingEnv` |
| Environment config | `IGRISCMotionTrackingEnvCfg` |
| Robot config | `IGRIS_C_WRIST_HAND_INDEPENDENT_CFG` |
| Student observation size | 1,601 |
| Student action size | 23 |
| Physics timestep | 0.005 s |
| Decimation | 4 |
| Policy/reference period | 0.02 s (50 Hz) |
| Default reference | Captured standing pose |
| Optional reference | Offline NPZ replay |

The scene contains the robot, six robot-mounted cameras, a ground plane, and a dome light. The calibrated head RGB/depth pair and wrist cameras mirror the H1 task settings; the two head-mounted side RGB cameras remain available. The scene does not create a table, box, or external environment USD.

## Layout

```text
motion_tracking/
├── README.md
├── __init__.py
├── action_adapter.py
├── camera_cfg.py
├── env_cfg.py
├── observations.py
├── robot_cfg.py
├── scene_cfg.py
├── student_contract.py
├── controllers/
│   ├── __init__.py
│   ├── auxiliary_joint_controller.py
│   ├── base.py
│   ├── npz_motion_reference.py
│   ├── standing_reference.py
│   └── student_policy.py
└── mdp/
    ├── __init__.py
    ├── reference_command.py
    └── reference_observations.py
```

| File | Responsibility |
| --- | --- |
| `__init__.py` | Registers the Gym task |
| `env_cfg.py` | Composes scene, action, observation, command, reward, and termination configs |
| `scene_cfg.py` | Adds ground and lighting to the reusable robot scene |
| `camera_cfg.py` | Applies the H1 OpenCV pinhole intrinsic and distortion model |
| `robot_cfg.py` | Selects the wrist/hand robot and defines the six mounted cameras |
| `student_contract.py` | Defines the immutable 23-joint checkpoint order |
| `observations.py` | Defines and validates the 1,601-element Student observation |
| `action_adapter.py` | Validates the 23 actions and applies the safe joint-position action config |
| `controllers/student_policy.py` | Restores and runs the deterministic Student MLP |
| `controllers/base.py` | Defines the common motion-reference protocol and frame contract |
| `controllers/standing_reference.py` | Captures and holds the initialized standing pose |
| `controllers/npz_motion_reference.py` | Loads, validates, maps, and replays NPZ references |
| `controllers/auxiliary_joint_controller.py` | Controls wrist/active-hand joints and computes distal targets |
| `mdp/reference_command.py` | Exposes a replaceable reference source to the observation manager |
| `mdp/reference_observations.py` | Converts the active reference into Student observation terms |

## Runtime data flow

```text
StandingReferenceSource or NpzMotionReferenceSource
                  │
                  ▼
          ReferenceCommand (motion)
                  │
                  ▼
        Student observation (1,601)
                  │
                  ▼
          StudentPolicy (23 actions)
                  │
                  ▼
          StudentActionAdapter
                  │
                  └──────────────► original 23 robot joints

NPZ wrist_hand_* channel (optional)
                  │
                  ▼
         WristHandPoseController
                  ├──────────────► 6 wrist joints
                  ├──────────────► 12 active hand joints
                  └──────────────► 10 computed distal joints
```

The Student Policy never receives or produces wrist/finger actions. This separation keeps existing checkpoints valid when the robot articulation contains additional joints.

## Student checkpoint contract

`student_108.pt` expects exactly 1,601 observations and produces exactly 23 normalized joint-position actions. The network architecture is fixed to hidden dimensions `(512, 512, 256, 128)`.

The canonical joint order is:

```text
 0  Joint_Shoulder_Pitch_Left
 1  Joint_Shoulder_Pitch_Right
 2  Joint_Waist_Yaw
 3  Joint_Shoulder_Roll_Left
 4  Joint_Shoulder_Roll_Right
 5  Joint_Waist_Roll
 6  Joint_Shoulder_Yaw_Left
 7  Joint_Shoulder_Yaw_Right
 8  Joint_Waist_Pitch
 9  Joint_Elbow_Pitch_Left
10  Joint_Elbow_Pitch_Right
11  Joint_Hip_Pitch_Left
12  Joint_Hip_Pitch_Right
13  Joint_Hip_Roll_Left
14  Joint_Hip_Roll_Right
15  Joint_Hip_Yaw_Left
16  Joint_Hip_Yaw_Right
17  Joint_Knee_Pitch_Left
18  Joint_Knee_Pitch_Right
19  Joint_Ankle_Pitch_Left
20  Joint_Ankle_Pitch_Right
21  Joint_Ankle_Roll_Left
22  Joint_Ankle_Roll_Right
```

Do not add wrist or finger joints to this list. Do not rename or reorder these joints without exporting and validating a new checkpoint.

### Observation term order

The declaration order is part of the checkpoint contract:

```text
joint_pos
joint_vel
projected_gravity
base_ang_vel
actions
motion_joint_pos
motion_joint_vel
motion_anchor_pos_z
motion_anchor_ori_b
motion_body_pos_b
motion_anchor_lin_vel_b
motion_anchor_ang_vel_b
```

The proprioceptive terms use a history length of 20. Training-time observation corruption is disabled during inference. `validate_student_observation_manager()` rejects a changed term order or concatenation mode at startup.

## Reference sources

`ReferenceCommand` owns one `MotionReferenceSource` and exposes it under the command name `motion`.

### Standing reference

The environment starts with `StandingReferenceSource`. After the robot is initialized, the source captures the checkpoint joints, anchor pose, tracked body positions, and zero target velocities. The captured tensors remain immutable, so a falling robot cannot move its own target.

When the runner is started without `--motion_file`, this standing frame remains active for the entire session.

### NPZ reference

When `--motion_file` is supplied, the runner replaces the standing source with `NpzMotionReferenceSource`. All environments receive the same frame.

- Frame selection uses `floor(elapsed_time * fps)`.
- Frames are not interpolated.
- Non-looping replay holds the final frame.
- Looping replay wraps with modulo `frame_count`.
- Dataset joints and bodies are mapped by name into the runtime order.
- All loaded floating-point arrays must be finite and have the same frame count.

## NPZ schema

Let `F` be the frame count, `J` the number of dataset joints, and `B` the number of dataset bodies.

### Required keys

| Key | Shape | Expected convention | Usage |
| --- | --- | --- | --- |
| `fps` | scalar | Hz, positive | Timeline sampling |
| `joint_names` | `(J,)` | string | Dataset-to-runtime joint mapping |
| `body_names` | `(B,)` | string | Dataset-to-runtime body mapping |
| `local_frame_body_name` | scalar | string | Selects local-frame anchor velocities |
| `joint_pos` | `(F, J)` | rad | Student joint reference positions |
| `joint_vel` | `(F, J)` | rad/s | Student joint reference velocities |
| `body_pos_w` | `(F, B, 3)` | m, world frame | Anchor height |
| `body_quat_w` | `(F, B, 4)` | unit quaternion `(w, x, y, z)` | Anchor orientation |
| `body_pos_b` | `(F, B, 3)` | m, anchor-local frame | Tracked body reference positions |
| `body_lin_vel_b` | `(F, B, 3)` | m/s, local frame | Anchor linear velocity |
| `body_ang_vel_b` | `(F, B, 3)` | rad/s, local frame | Anchor angular velocity |

The file may contain additional metadata or world-frame arrays. Extra keys are ignored by the replay source.

The schema does not store unit metadata. Producers must follow the conventions in the table.

### Optional wrist/hand keys

All three keys must be present together or all three must be omitted.

| Key | Shape | Expected convention |
| --- | --- | --- |
| `wrist_hand_joint_names` | `(A,)` | string names for auxiliary joints |
| `wrist_hand_joint_pos` | `(F, A)` | rad |
| `wrist_hand_joint_vel` | `(F, A)` | rad/s |

The current controller requests `A = 18` joints. Dataset columns are mapped by name, but using the canonical order below is recommended:

```text
Joint_Wrist_Yaw_Left
Joint_Wrist_Roll_Left
Joint_Wrist_Pitch_Left
Joint_Wrist_Yaw_Right
Joint_Wrist_Roll_Right
Joint_Wrist_Pitch_Right
l_0_joint_thumb_proximal
l_1_joint_thumb_middle
l_3_joint_index_middle
l_5_joint_middle_middle
l_7_joint_ring_middle
l_9_joint_little_middle
r_0_joint_thumb_proximal
r_1_joint_thumb_middle
r_3_joint_index_middle
r_5_joint_middle_middle
r_7_joint_ring_middle
r_9_joint_little_middle
```

If the optional keys are absent, the auxiliary controller holds the robot's default pose. The current defaults are left wrist roll `+30°`, right wrist roll `-30°`, all other wrist joints `0`, and all active hand joints `0`.

Do not store distal finger targets in the auxiliary channel. They are derived from the active middle-joint targets immediately before articulation commands are written.

## Finger coupling

The controller applies the following position relationship and clamps the result to the distal joint limit:

```text
distal_target = clamp(middle_target * multiplier, distal_lower, distal_upper)
```

| Side | Finger | Active joint | Computed distal joint | Multiplier |
| --- | --- | --- | --- | ---: |
| Left | Thumb | `l_1_joint_thumb_middle` | `l_2_joint_thumb_distal` | 1.285714286 |
| Left | Index | `l_3_joint_index_middle` | `l_4_joint_index_distal` | 1.0 |
| Left | Middle | `l_5_joint_middle_middle` | `l_6_joint_middle_distal` | 1.0 |
| Left | Ring | `l_7_joint_ring_middle` | `l_8_joint_ring_distal` | 1.0 |
| Left | Little | `l_9_joint_little_middle` | `l_10_joint_little_distal` | 1.0 |
| Right | Thumb | `r_1_joint_thumb_middle` | `r_2_joint_thumb_distal` | 1.285714286 |
| Right | Index | `r_3_joint_index_middle` | `r_4_joint_index_distal` | 1.0 |
| Right | Middle | `r_5_joint_middle_middle` | `r_6_joint_middle_distal` | 1.0 |
| Right | Ring | `r_7_joint_ring_middle` | `r_8_joint_ring_distal` | 1.0 |
| Right | Little | `r_9_joint_little_middle` | `r_10_joint_little_distal` | 1.0 |

The thumb proximal joints (`l_0_joint_thumb_proximal`, `r_0_joint_thumb_proximal`) remain independently controlled and are not part of the distal coupling table.

## Adding a motion

1. Generate all required Student reference arrays for the same frame count.
2. Include every canonical Student joint name. Keep the canonical order unless there is a specific reason to rely on name remapping.
3. Include every body required by `MotionTrackingCommandsCfg`, including `base_link`.
4. Use a positive `fps`; 50 Hz matches the current policy period.
5. Normalize all `body_quat_w` values and use `(w, x, y, z)` order.
6. Ensure every floating-point array contains no NaN or Inf.
7. Add all three `wrist_hand_*` keys only when the motion controls the wrist or hand.
8. Store active hand targets only; let `WristHandPoseController` generate distal targets.
9. Run the motion with `--stop_at_end` for a finite validation pass before enabling `--loop`.

The repository's right-arm/finger example is generated by:

```text
scripts/tools/create_right_arm_finger_motion.py
```

## Extending reference backends

A new live or offline backend must implement the `MotionReferenceSource` protocol:

```python
class MotionReferenceSource(Protocol):
    def initialize(self) -> None: ...
    def reset(self, env_ids=None) -> None: ...
    def get_reference(self, timestamp: float) -> MotionReferenceFrame: ...
    def close(self) -> None: ...
```

`get_reference()` must be non-blocking and return batched tensors on the environment device. Call `ReferenceCommand.set_source()` to replace the active backend without changing observation code.

## Invariants to preserve

- Keep the Student joint count and order at 23.
- Keep the flattened Student observation size at 1,601.
- Keep the observation term order and 20-step history unchanged.
- Keep policy and reference updates at 50 Hz unless a new checkpoint is validated.
- Keep wrist/finger commands outside the 23-action Student Policy path.
- Keep distal commands derived from active middle-joint targets.
- Reject partial auxiliary channels, missing names, shape mismatches, NaN, and Inf.
- Treat PyTorch checkpoints as trusted inputs only.

## Validation

From the repository root:

```bash
python3 -m py_compile \
    source/robros_lab/robros_lab/tasks/motion_tracking/*.py \
    source/robros_lab/robros_lab/tasks/motion_tracking/controllers/*.py \
    source/robros_lab/robros_lab/tasks/motion_tracking/mdp/*.py

git diff --check
```

For runtime validation, use the commands in the root [README](../../../../../README.md) with `--num_envs 1` first.
