"""Run an IGRIS-C HugWBC policy with random or NPZ commands."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CURRENT_PACKAGE_ROOT = REPOSITORY_ROOT / "source/robros_lab"
sys.path.insert(0, str(CURRENT_PACKAGE_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT / "logs/rsl_rl/hugwbc_lowerbody_symmetry/model_31900.pt"
)
DEFAULT_TASK = "Robros-IGRIS-C-Flat-HugWBC-LowerBody-Symmetry"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK, help="Registered HugWBC task name.")
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--motion_file", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--loop", action="store_true")
parser.add_argument("--stop_at_end", action="store_true")
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robros_lab.tasks  # noqa: E402,F401
from robros_lab.tasks.hug_wbc.controllers import (  # noqa: E402
    HugWBCPolicy,
    NpzHugWBCCommandSource,
)
from robros_lab.tasks.hug_wbc.config.igris_c.env_cfg import (  # noqa: E402
    LOWER_BODY_JOINTS,
    LOWER_BODY_SYMMETRY_JOINTS,
)
from robros_lab.tasks.motion_tracking.controllers import (  # noqa: E402
    AUXILIARY_JOINT_NAMES,
    WristHandPoseController,
)


def _validate_runtime_contract(env, policy: HugWBCPolicy, observation: torch.Tensor) -> None:
    expected_observation_shape = (env.num_envs, policy.observation_dim)
    if tuple(observation.shape) != expected_observation_shape:
        raise ValueError(
            f"Expected policy observation {expected_observation_shape}, got {tuple(observation.shape)}."
        )
    if env.action_manager.total_action_dim != policy.action_dim:
        raise ValueError(
            f"Checkpoint requires {policy.action_dim} actions, but the environment has "
            f"{env.action_manager.total_action_dim}."
        )
    if abs(float(env.step_dt) - 0.02) > 1.0e-9:
        raise ValueError(f"HugWBC requires a 0.02 s policy step, got {env.step_dt}.")

    if policy.action_dim == 13:
        expected_action_terms = ("gait_freq", "joint_pos", "teleop")
        expected_action_term_dims = (1, 12, 0)
        expected_joint_names = LOWER_BODY_SYMMETRY_JOINTS
    elif policy.action_dim == 15:
        expected_action_terms = ("joint_pos", "gait_freq", "teleop")
        expected_action_term_dims = (14, 1, 0)
        expected_joint_names = LOWER_BODY_JOINTS
    else:
        raise ValueError(f"Unsupported HugWBC action dimension: {policy.action_dim}.")
    if tuple(env.action_manager.active_terms) != expected_action_terms:
        raise ValueError(
            "Action term order does not match the checkpoint. "
            f"Expected {expected_action_terms}, got {tuple(env.action_manager.active_terms)}."
        )
    if tuple(env.action_manager.action_term_dim) != expected_action_term_dims:
        raise ValueError(
            "Action term dimensions do not match the checkpoint. "
            f"Expected {expected_action_term_dims}, got "
            f"{tuple(env.action_manager.action_term_dim)}."
        )
    action_term = env.action_manager.get_term("joint_pos")
    resolved_joint_names = tuple(getattr(action_term, "_joint_names", ()))
    if resolved_joint_names != tuple(expected_joint_names):
        raise ValueError(
            "Lower-body action joint order does not match the checkpoint. "
            f"Expected {tuple(expected_joint_names)}, got {resolved_joint_names}."
        )


def _create_wrist_hand_controller(env) -> WristHandPoseController | None:
    robot = env.scene["robot"]
    if not set(AUXILIARY_JOINT_NAMES).issubset(robot.joint_names):
        return None
    _, resolved_joint_names = robot.find_joints(
        list(AUXILIARY_JOINT_NAMES),
        preserve_order=True,
    )
    if tuple(resolved_joint_names) != AUXILIARY_JOINT_NAMES:
        return None
    return WristHandPoseController(robot)


def main() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args_cli.num_envs}.")
    if args_cli.max_steps < 0:
        raise ValueError(f"--max_steps must be non-negative, got {args_cli.max_steps}.")
    if args_cli.loop and args_cli.stop_at_end:
        raise ValueError("--loop and --stop_at_end cannot be used together.")
    if args_cli.motion_file is None and (args_cli.loop or args_cli.stop_at_end):
        raise ValueError("--loop and --stop_at_end require --motion_file.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    source: NpzHugWBCCommandSource | None = None
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        if args_cli.motion_file is not None:
            source = NpzHugWBCCommandSource(
                motion_file=args_cli.motion_file,
                num_envs=unwrapped.num_envs,
                device=unwrapped.device,
                loop=args_cli.loop,
            )
            source.initialize()
            unwrapped.command_manager.get_term("base_velocity").set_source(source)
            unwrapped.command_manager.get_term("teleoperation_command").set_source(source)
            source.reset()
            observations, _ = env.reset()

        policy_observation = observations["policy"]
        policy = HugWBCPolicy(args_cli.checkpoint, device=unwrapped.device)
        _validate_runtime_contract(unwrapped, policy, policy_observation)
        wrist_hand_controller = _create_wrist_hand_controller(unwrapped)
        if policy_observation.dtype != policy.dtype:
            raise TypeError(
                f"Environment observation dtype {policy_observation.dtype} does not match "
                f"checkpoint dtype {policy.dtype}."
            )

        print(f"[INFO]: Checkpoint: {policy.checkpoint_path}")
        print(f"[INFO]: Checkpoint iteration: {policy.iteration}")
        print(f"[INFO]: Policy observation/action: {policy.observation_dim}/{policy.action_dim}")
        if wrist_hand_controller is None:
            print("[INFO]: Robot variant has fixed wrists/hands; auxiliary targets are ignored")
        if source is None:
            print("[INFO]: Command source: fixed base velocity + random teleoperation")
        else:
            print(f"[INFO]: Motion file: {source.motion_file}")
            print(
                f"[INFO]: Motion: {source.frame_count} frames at {source.fps:.3f} Hz "
                f"({source.duration:.3f} s)"
            )

        step_count = 0
        while simulation_app.is_running():
            start_time = time.perf_counter()
            with torch.inference_mode():
                action = policy.infer(policy_observation)
                if source is not None and wrist_hand_controller is not None:
                    wrist_hand_reference = source.get_wrist_hand_reference()
                    if wrist_hand_reference is not None:
                        wrist_hand_controller.set_target(
                            joint_names=wrist_hand_reference.joint_names,
                            joint_pos=wrist_hand_reference.joint_pos,
                            joint_vel=wrist_hand_reference.joint_vel,
                        )
                if wrist_hand_controller is not None:
                    wrist_hand_controller.apply()
                observations, _, _, _, _ = env.step(action)
                policy_observation = observations["policy"]
            step_count += 1
            if source is not None and args_cli.stop_at_end and source.is_finished:
                break
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
            sleep_time = float(unwrapped.step_dt) - (time.perf_counter() - start_time)
            if args_cli.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)
        print(f"[INFO]: Completed {step_count} policy steps.")
    finally:
        if source is not None:
            source.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
