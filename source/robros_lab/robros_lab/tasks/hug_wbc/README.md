# IGRIS-C HugWBC playback

This task runs the 280-observation, 15-action HugWBC actor independently from
the existing 1,601-observation Student motion-tracking task. The robot uses the
wrist/hand-independent articulation, while an auxiliary controller applies the six
wrist and twelve active-finger NPZ targets and derives ten distal-finger targets.

Run random upper-body teleoperation with a fixed zero base command:

```bash
./third_party/IsaacLab/isaaclab.sh -p scripts/hugwbc/igris_c_hugwbc.py \
  --checkpoint logs/rsl_rl/igris_c_hugwbc/model_16900.pt --num_envs 1
```

Replay the included synchronized base/upper-body command clip:

```bash
./third_party/IsaacLab/isaaclab.sh -p scripts/hugwbc/igris_c_hugwbc.py \
  --checkpoint logs/rsl_rl/igris_c_hugwbc/model_16900.pt \
  --motion_file motion/hugwbc_forward_arm_motion.npz --num_envs 1 --loop
```

Append `--headless --max_steps 200` for a finite smoke test. The task keeps
standing-environment classification disabled so gait phase advances normally.
