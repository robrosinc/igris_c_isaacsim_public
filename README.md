# IGRIS-C Isaac Sim

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76B900)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-submodule-76B900)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

Isaac Lab inference environment for the IGRIS-C humanoid robot. The project runs a fixed 23-joint Student Policy while controlling the wrist and articulated hand through a separate auxiliary motion channel.

The runtime supports either a fixed standing reference or an offline NPZ motion. Motions without wrist/hand data remain compatible: the Student Policy controls its original 23 joints and the auxiliary controller holds the default wrist and hand pose.

## Requirements

### Tested software stack

| Tool | Version |
| --- | --- |
| OS | Ubuntu 24.04, x86_64 |
| Python | 3.11 |
| NVIDIA Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0 + CUDA 12.8 wheels |
| TorchVision | 0.22.0 |
| NVIDIA driver | CUDA 12.8-capable NVIDIA 580 series |

This repository is pinned to Isaac Sim 5.1.0 and the included Isaac Lab submodule revision. Upgrade either dependency only after validating the task, robot articulation, cameras, and Student Policy observation contract.

### Recommended system

The following values are based on the official [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

| Component | Minimum | Recommended (`Good`) |
| --- | --- | --- |
| CPU | Intel Core i7 7th Gen / AMD Ryzen 5 | Intel Core i7 9th Gen / AMD Ryzen 7 |
| CPU cores | 4 | 8 |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB SSD | 500 GB SSD |
| GPU | NVIDIA GeForce RTX 4080 | NVIDIA GeForce RTX 5080 |
| VRAM | 16 GB | 16 GB or more |
| Linux driver | NVIDIA 580 series | NVIDIA 580 series |

An RTX GPU with RT Cores is required. Additional RAM and VRAM are recommended when increasing `--num_envs`, enabling multiple cameras, or rendering at higher resolutions.
Use a CUDA 12.8-capable NVIDIA 580-series Linux driver for this project. NVIDIA lists `580.65.06` as the tested Linux driver for Isaac Sim 5.1.0.


## Layout

```text
igris_c_isaacsim_public/
├── README.md
├── logs/rsl_rl/igris_c_tracking/
│   └── student_108.pt                 # Default Student Policy checkpoint
├── motion/
│   ├── recorded_arms.npz
│   ├── recorded_walk_2.npz
│   └── right_arm_finger_sequence.npz
├── scripts/
│   ├── motion_tracking/
│   │   └── igris_c_motion_tracking.py  # Main inference entry point
│   └── tools/                          # Repository utilities and motion generator
├── source/robros_lab/robros_lab/
│   ├── assets/                         # IGRIS-C robot assets and simulation config
│   └── tasks/motion_tracking/           # Task, observations, references, and controllers
└── third_party/
    ├── IsaacLab/
    └── IsaacSim/
```

## Installation

### 1. Install prerequisites

Install a CUDA 12.8-capable NVIDIA 580-series Linux driver, Miniconda or Conda, and Git before continuing.

### 2. Clone the repository and public submodules

```bash
git clone --recurse-submodules https://github.com/robrosinc/igris_c_isaacsim_public.git
cd igris_c_isaacsim_public
```

If the repository was cloned without submodules:

```bash
git submodule update --init third_party/IsaacLab third_party/IsaacSim
```

The Isaac Lab and Isaac Sim repositories are not stored directly in this repository.
Git reads their public upstream URLs and pinned commits from `.gitmodules` and fetches
them into `third_party/`.
### 3. Create the Python environment

```bash
conda create -y -n isaac_env python=3.11
conda activate isaac_env

python -m pip install --upgrade pip
```

### 4. Install Isaac Sim and PyTorch

```bash
pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

pip install -U torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install Isaac Lab and ROBROS Lab

```bash
export OMNI_KIT_ACCEPT_EULA=YES

cd third_party/IsaacLab
./isaaclab.sh --install none
cd ../..

pip install -e source/robros_lab
```

## Quick start

Activate the environment before each session:

```bash
conda activate isaac_env
export OMNI_KIT_ACCEPT_EULA=YES
```

Run without `--motion_file` to hold the initial standing reference:

```bash
./third_party/IsaacLab/isaaclab.sh -p \
    scripts/motion_tracking/igris_c_motion_tracking.py \
    --num_envs 1
```

The default checkpoint is:

```text
logs/rsl_rl/igris_c_tracking/student_108.pt
```

Use `--checkpoint <path>` to load another compatible checkpoint. Checkpoint files are loaded through PyTorch pickle and must come from a trusted source.

## Motion examples

All included motions run at 50 Hz. Without `--loop`, the final frame is held after playback finishes.

| Motion | Frames | Duration | Wrist/hand channel | Description |
| --- | ---: | ---: | --- | --- |
| `recorded_arms.npz` | 362 | 7.22 s | No | Recorded upper-body and arm reference; wrist and hand hold their default pose |
| `recorded_walk_2.npz` | 243 | 4.84 s | No | Recorded walking reference; wrist and hand hold their default pose |
| `right_arm_finger_sequence.npz` | 901 | 18.00 s | Yes | Raises the right arm, closes and reopens the fingers sequentially, then returns the arm to zero |

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

### Right arm and finger sequence

```bash
./third_party/IsaacLab/isaaclab.sh -p \
    scripts/motion_tracking/igris_c_motion_tracking.py \
    --motion_file motion/right_arm_finger_sequence.npz \
    --num_envs 1
```

Add `--loop` to repeat any motion:

```bash
./third_party/IsaacLab/isaaclab.sh -p \
    scripts/motion_tracking/igris_c_motion_tracking.py \
    --motion_file motion/right_arm_finger_sequence.npz \
    --num_envs 1 \
    --loop
```

## Runtime options

| Option | Description |
| --- | --- |
| `--motion_file PATH` | Optional NPZ reference; omit it to hold the initial pose |
| `--checkpoint PATH` | Override the default Student Policy checkpoint |
| `--num_envs N` | Number of parallel simulation environments |
| `--loop` | Replay the motion continuously |
| `--stop_at_end` | Close after the final frame instead of holding it |
| `--max_steps N` | Stop after a fixed number of policy steps |
| `--real_time` | Sleep when simulation runs faster than the 50 Hz policy period |
| `--headless` | Run without the graphical window |
| `--viewport_camera CAMERA` | Select `viewer`, `head_camera`, `right_rgb_camera`, or `left_rgb_camera` |
| `--disable_fabric` | Disable Fabric and use USD I/O operations |

`--loop` and `--stop_at_end` cannot be used together.

## Control architecture

The Student Policy contract remains fixed while additional robot joints are controlled independently:

```text
NPZ reference
├── joint_pos / joint_vel (23 joints)
│   └── Student Policy reference → 23 policy actions
└── wrist_hand_joint_pos / wrist_hand_joint_vel (optional, 18 joints)
    └── auxiliary controller
        ├── 6 wrist targets
        ├── 12 active hand targets
        └── 10 distal targets computed from active middle-joint targets
```

The Student Policy expects 1,601 observations and produces 23 normalized joint-position actions. The auxiliary controller does not alter that checkpoint contract. For coupled fingers, distal targets are calculated immediately before commands are written to the articulation; NPZ files therefore only need to provide the active hand targets.

## License

This project is licensed under the [Apache License 2.0](LICENSE). Third-party components retain their respective licenses.
