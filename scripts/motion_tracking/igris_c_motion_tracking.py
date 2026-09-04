"""Run IGRIS-C motion tracking with a fixed pose or an optional NPZ reference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "logs/rsl_rl/igris_c_tracking/student_108.pt"
CAMERA_PRIM_PATHS = {
    "head_camera": "Link_Neck_Pitch/d435_camera",
    "head_depth_camera": "Link_Neck_Pitch/d435_depth_camera",
    "right_rgb_camera": "Link_Neck_Pitch/RightRgbCamera",
    "left_rgb_camera": "Link_Neck_Pitch/LeftRgbCamera",
    "left_wrist_camera": "Left_Hand/l_hand_camera_link",
    "right_wrist_camera": "Right_Hand/r_hand_camera_link",
}
LENS_DISTORTION_EXTENSION = "omni.usd.schema.omni_lens_distortion"


parser = argparse.ArgumentParser(description="Run a fixed initial pose or replay an optional motion NPZ.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-IGRISC-Motion-Tracking-v0",
    help="Registered motion-tracking task name.",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=DEFAULT_CHECKPOINT,
    help="Path to the distillation checkpoint.",
)
parser.add_argument(
    "--motion_file",
    type=Path,
    default=None,
    help="Optional processed tracking NPZ. Omit to hold the initial standing reference.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after this many policy steps. Zero runs until the app closes.",
)
parser.add_argument(
    "--loop",
    action="store_true",
    help="Loop to frame zero after the final reference frame.",
)
parser.add_argument(
    "--stop_at_end",
    action="store_true",
    help="Close the application when a non-looping reference reaches its final frame.",
)
parser.add_argument(
    "--real_time",
    action="store_true",
    help="Sleep when simulation is faster than the 50 Hz policy period.",
)

parser.add_argument(
    "--disable_fabric",
    action="store_true",
    help="Disable Fabric and use USD I/O operations.",
)
parser.add_argument(
    "--viewport_camera",
    choices=("viewer", *CAMERA_PRIM_PATHS),
    default="viewer",
    help="GUI viewport source.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
kit_args = args_cli.kit_args or ""
if f"--enable {LENS_DISTORTION_EXTENSION}" not in kit_args:
    args_cli.kit_args = f"{kit_args} --enable {LENS_DISTORTION_EXTENSION}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import robros_lab.tasks  # noqa: F401
from robros_lab.tasks.motion_tracking.action_adapter import StudentActionAdapter
from robros_lab.tasks.motion_tracking.controllers import (
    AUXILIARY_JOINT_NAMES,
    WristHandPoseController,
)
from robros_lab.tasks.motion_tracking.controllers.npz_motion_reference import (
    NpzMotionReferenceSource,
)
from robros_lab.tasks.motion_tracking.controllers.student_policy import StudentPolicy
from robros_lab.tasks.motion_tracking.observations import validate_student_observation_manager
from robros_lab.tasks.motion_tracking.student_contract import (
    STUDENT_JOINT_COUNT,
    STUDENT_JOINT_NAMES,
)


def _resolve_action_joint_names(env) -> tuple[str, ...]:
    action_term = env.action_manager.get_term("joint_pos")
    joint_names = getattr(action_term, "_joint_names", None)
    if joint_names is None:
        raise AttributeError("Action term joint_pos does not expose resolved joint names.")
    return tuple(joint_names)


def _configure_viewport(env) -> None:
    camera_name = args_cli.viewport_camera
    if camera_name == "viewer":
        return

    env.scene[camera_name]
    if args_cli.headless:
        print(
            f"[WARN]: --viewport_camera {camera_name} has no visible viewport in headless mode.",
            flush=True,
        )
        return

    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Could not resolve the active Isaac Sim GUI viewport.")
    camera_prim_path = f"/World/envs/env_0/Robot/{CAMERA_PRIM_PATHS[camera_name]}"
    viewport.set_active_camera(camera_prim_path)
    print(f"[INFO]: Active GUI viewport camera: {camera_prim_path}", flush=True)


def _validate_runtime_contract(env, policy: StudentPolicy) -> tuple[str, ...]:
    validate_student_observation_manager(env.observation_manager)
    if policy.observation_dim != 1601:
        raise ValueError(
            f"student_108.pt requires 1601 observations, got {policy.observation_dim}."
        )
    if policy.action_dim != STUDENT_JOINT_COUNT:
        raise ValueError(
            f"student_108.pt requires {STUDENT_JOINT_COUNT} actions, "
            f"got {policy.action_dim}."
        )
    if env.action_manager.total_action_dim != policy.action_dim:
        raise ValueError(
            f"Environment exposes {env.action_manager.total_action_dim} actions, "
            f"but the checkpoint produces {policy.action_dim}."
        )
    if abs(float(env.step_dt) - 0.02) > 1.0e-9:
        raise ValueError(f"Student policy requires a 0.02 s step, got {env.step_dt}.")

    action_joint_names = _resolve_action_joint_names(env)
    if action_joint_names != STUDENT_JOINT_NAMES:
        raise ValueError(
            "Action joint names/order do not match the checkpoint. "
            f"Expected {STUDENT_JOINT_NAMES}, got {action_joint_names}."
        )
    motion_command = env.command_manager.get_term("motion")
    reference_joint_names = tuple(motion_command.resolved_joint_names)
    if reference_joint_names != STUDENT_JOINT_NAMES:
        raise ValueError(
            "Reference joint names/order do not match the checkpoint. "
            f"Expected {STUDENT_JOINT_NAMES}, got {reference_joint_names}."
        )
    return STUDENT_JOINT_NAMES


def main() -> None:
    """Create the environment and run a fixed pose or optional NPZ reference."""

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
    unwrapped_env = env.unwrapped

    try:
        # The environment's built-in fixed reference captures and holds this initial pose.
        observations, _ = env.reset()
        _configure_viewport(unwrapped_env)

        policy = StudentPolicy(
            checkpoint_path=args_cli.checkpoint,
            device=unwrapped_env.device,
        )
        checkpoint_joint_names = _validate_runtime_contract(unwrapped_env, policy)
        source: NpzMotionReferenceSource | None = None
        if args_cli.motion_file is not None:
            motion_command = unwrapped_env.command_manager.get_term("motion")
            source = NpzMotionReferenceSource(
                motion_file=args_cli.motion_file,
                joint_names=motion_command.resolved_joint_names,
                body_names=motion_command.resolved_body_names,
                anchor_body_name=motion_command.robot_anchor_body_name,
                num_envs=unwrapped_env.num_envs,
                device=unwrapped_env.device,
                loop=args_cli.loop,
                auxiliary_joint_names=AUXILIARY_JOINT_NAMES,
            )
            motion_command.set_source(source)

            # This reset is consumed by the policy and starts at NPZ frame zero.
            observations, _ = env.reset()
            # reset() restarts the source but does not query it. Arm its timestamp
            # without recomputing observations or appending a duplicate sample.
            motion_command.set_source(
                source,
                initialize=False,
                close_previous=False,
            )

        student_observation = observations["student"]
        expected_shape = (unwrapped_env.num_envs, policy.observation_dim)
        if tuple(student_observation.shape) != expected_shape:
            raise ValueError(
                f"Environment returned student observation shape "
                f"{tuple(student_observation.shape)}, expected {expected_shape}."
            )

        action_adapter = StudentActionAdapter(
            student_joint_names=checkpoint_joint_names,
            robot_joint_names=_resolve_action_joint_names(unwrapped_env),
            num_envs=unwrapped_env.num_envs,
            device=unwrapped_env.device,
            dtype=policy.dtype,
        )

        auxiliary_joint_controller = WristHandPoseController(unwrapped_env.scene["robot"])
        print(f"[INFO]: Checkpoint: {policy.checkpoint_path}")
        print(f"[INFO]: Checkpoint iteration: {policy.iteration}")
        if source is None:
            print("[INFO]: Motion reference: fixed initial pose")
        else:
            print(f"[INFO]: Motion file: {source.motion_file}")
            print(
                f"[INFO]: Motion: {source.frame_count} frames at {source.fps:.3f} Hz "
                f"({source.duration:.3f} s)"
            )
            print(f"[INFO]: Replay mode: {'loop' if args_cli.loop else 'hold final frame'}")
        print(f"[INFO]: Policy step: {unwrapped_env.step_dt:.6f} s")

        step_count = 0
        report_wall_start = time.perf_counter()
        report_simulated_time = 0.0
        while simulation_app.is_running():
            tick_start = time.perf_counter()
            with torch.inference_mode():
                auxiliary_reference = (
                    source.get_auxiliary_reference() if source is not None else None
                )
                if auxiliary_reference is not None:
                    auxiliary_joint_controller.set_target(
                        joint_names=auxiliary_reference.joint_names,
                        joint_pos=auxiliary_reference.joint_pos,
                        joint_vel=auxiliary_reference.joint_vel,
                    )
                raw_action = policy.infer(student_observation)
                adapted_action = action_adapter.adapt(
                    raw_action,
                    dt=float(unwrapped_env.step_dt),
                )
                auxiliary_joint_controller.apply()
                observations, _, _, _, _ = env.step(adapted_action.final_action)
                student_observation = observations["student"]

            step_count += 1
            report_simulated_time += float(unwrapped_env.step_dt)
            report_wall_elapsed = time.perf_counter() - report_wall_start
            if report_wall_elapsed >= 1.0:
                rtf = report_simulated_time / report_wall_elapsed
                if source is None:
                    print(f"[INFO]: step={step_count} RTF={rtf:.2f}", flush=True)
                else:
                    print(
                        f"[INFO]: step={step_count} frame={source.current_frame_index}/"
                        f"{source.frame_count - 1} RTF={rtf:.2f}",
                        flush=True,
                    )
                report_wall_start = time.perf_counter()
                report_simulated_time = 0.0

            if args_cli.stop_at_end and source is not None and source.is_finished:
                break
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
            if args_cli.real_time:
                sleep_time = float(unwrapped_env.step_dt) - (
                    time.perf_counter() - tick_start
                )
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
