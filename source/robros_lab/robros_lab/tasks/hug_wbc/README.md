# IGRIS-C HugWBC playback

The default task runs the 700-observation, 13-action lower-body symmetry actor ported
from `WalkingRobotPlayEnvCfg`. It uses a 10-frame observation history, one
gait-frequency action followed by twelve leg joint-position actions, a 250 Hz
simulation rate, and 50 Hz policy rate.
The wrist/hand-independent articulation keeps policy observations restricted to the
original 23 training joints while the auxiliary controller drives wrists and hands.

Run random upper-body teleoperation with a fixed zero base command:

```bash
./third_party/IsaacLab/isaaclab.sh -p scripts/hugwbc/igris_c_hugwbc.py \
  --num_envs 1
```

Replay the included synchronized base/upper-body command clip:

```bash
./third_party/IsaacLab/isaaclab.sh -p scripts/hugwbc/igris_c_hugwbc.py \
  --motion_file motion/hugwbc_forward_arm_motion.npz --num_envs 1 --loop
```

Append `--headless --max_steps 200` for a finite smoke test. The task keeps
standing-environment classification disabled so gait phase advances normally.

The earlier 280-observation, 15-action checkpoint remains supported through task
`Robros-IGRIS-C-Flat-Teleop-Lowerbody-Action` and checkpoint
`logs/rsl_rl/igris_c_hugwbc/model_16900.pt`. That task uses the wrist/hand-independent
robot and applies the optional auxiliary wrist/hand targets.
