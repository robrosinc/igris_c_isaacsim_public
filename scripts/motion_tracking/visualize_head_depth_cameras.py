#!/usr/bin/env python3
"""Show IGRIS-C head cameras and intrinsics on the robot mesh.

Run from the repository root with:
    python scripts/motion_tracking/visualize_head_depth_cameras.py

Open http://127.0.0.1:8080 to inspect the camera mounts and intrinsics.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yourdfpy


ROBOT_ASSET = Path(
    "source/robros_lab/robros_lab/assets/robots/robros/igris_c/"
    "from_urdf/urdf/igris_c_v2_wrist_hand_isaac.urdf"
)
PUBLIC_CONFIG = Path(
    "source/robros_lab/robros_lab/tasks/motion_tracking/robot_cfg.py"
)


@dataclass(frozen=True)
class DepthCamera:
    key: str
    label: str
    color: tuple[int, int, int]
    position: tuple[float, float, float]  # Link_Neck_Pitch frame, metres
    wxyz: tuple[float, float, float, float]  # ROS optical frame in Link_Neck_Pitch
    width: int
    height: int
    intrinsic: np.ndarray
    distortion: tuple[float, ...]  # OpenCV [k1, k2, p1, p2, k3]


def _wxyz(rotation: Rotation) -> tuple[float, float, float, float]:
    x, y, z, w = rotation.as_quat()
    return float(w), float(x), float(y), float(z)


def _rotation(wxyz: tuple[float, float, float, float]) -> Rotation:
    w, x, y, z = wxyz
    return Rotation.from_quat((x, y, z, w))


def _literal_assignments(path: Path, names: set[str]) -> dict[str, object]:
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    missing = names - values.keys()
    if missing:
        raise ValueError(f"Missing public camera constants in {path}: {sorted(missing)}")
    return values


def _public_rotation(path: Path) -> tuple[float, float, float, float]:
    base_rotation = None
    rotation_call = None
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_camera_rotation":
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "base_rotation"
                    for target in statement.targets
                ):
                    base_rotation = ast.literal_eval(statement.value)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "HEAD_RGB_CAMERA_ROT"
            for target in node.targets
        ):
            rotation_call = node.value
    if base_rotation is None or not isinstance(rotation_call, ast.Call):
        raise ValueError(f"Cannot read public head camera rotation from {path}")
    if not isinstance(rotation_call.func, ast.Name) or rotation_call.func.id != "_camera_rotation":
        raise ValueError(f"Unexpected public head camera rotation in {path}")
    axis = np.asarray(ast.literal_eval(rotation_call.args[0]), dtype=float)
    angle = float(ast.literal_eval(rotation_call.args[1]))
    installation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.deg2rad(angle))
    return _wxyz(installation * _rotation(base_rotation))


def _load_public(config_path: Path) -> list[DepthCamera]:
    values = _literal_assignments(config_path, {
        "HEAD_RGB_CAMERA_POS", "HEAD_LEFT_IR_CAMERA_POS", "HEAD_RIGHT_IR_CAMERA_POS",
        "HEAD_RGB_INTRINSIC", "HEAD_RGB_DISTORTION",
        "HEAD_DEPTH_INTRINSIC", "HEAD_DEPTH_DISTORTION",
    })
    rotation = _public_rotation(config_path)
    return [
        DepthCamera(
            "public_rgb", "Public head RGB", (225, 84, 100),
            values["HEAD_RGB_CAMERA_POS"], rotation,
            640, 480, np.asarray(values["HEAD_RGB_INTRINSIC"], dtype=float),
            values["HEAD_RGB_DISTORTION"],
        ),
        DepthCamera(
            "public_left_ir", "Public left IR depth", (134, 91, 226),
            values["HEAD_LEFT_IR_CAMERA_POS"], rotation,
            640, 480, np.asarray(values["HEAD_DEPTH_INTRINSIC"], dtype=float),
            values["HEAD_DEPTH_DISTORTION"],
        ),
        DepthCamera(
            "public_right_ir", "Public right IR (RGB proxy)", (104, 192, 98),
            values["HEAD_RIGHT_IR_CAMERA_POS"], rotation,
            640, 480, np.asarray(values["HEAD_DEPTH_INTRINSIC"], dtype=float),
            values["HEAD_DEPTH_DISTORTION"],
        ),
    ]


def _undistorted_ray(camera: DepthCamera, u: float, v: float) -> np.ndarray:
    fx, fy = camera.intrinsic[0, 0], camera.intrinsic[1, 1]
    cx, cy = camera.intrinsic[0, 2], camera.intrinsic[1, 2]
    distorted_x, distorted_y = (u - cx) / fx, (v - cy) / fy
    x, y = distorted_x, distorted_y
    k1, k2, p1, p2, k3 = camera.distortion
    for _ in range(12):
        radius2 = x * x + y * y
        radial = 1.0 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
        offset_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        offset_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = (distorted_x - offset_x) / radial, (distorted_y - offset_y) / radial
    return np.array((x, y, 1.0), dtype=float)


def _image_plane_lines(camera: DepthCamera, distance: float) -> np.ndarray:
    segments: list[np.ndarray] = []

    def point(u: float, v: float) -> np.ndarray:
        return _undistorted_ray(camera, u, v) * distance

    def trace(start: tuple[float, float], end: tuple[float, float]) -> None:
        previous = point(*start)
        for fraction in np.linspace(1.0 / 16.0, 1.0, 16):
            current = point(
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            segments.append(np.stack((previous, current)))
            previous = current

    width, height = camera.width, camera.height
    for start, end in (
        ((0, 0), (width, 0)), ((width, 0), (width, height)),
        ((width, height), (0, height)), ((0, height), (0, 0)),
        ((width / 2, 0), (width / 2, height)),
        ((0, height / 2), (width, height / 2)),
    ):
        trace(start, end)
    origin = np.zeros(3)
    for u, v in ((0, 0), (width, 0), (width, height), (0, height)):
        segments.append(np.stack((origin, point(u, v))))
    segments.append(np.stack((origin, point(camera.intrinsic[0, 2], camera.intrinsic[1, 2]))))
    return np.asarray(segments, dtype=np.float32)


def _load_robot(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        path, filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent)
    )


def _show(server: viser.ViserServer, robot: yourdfpy.URDF, cameras: list[DepthCamera]) -> None:
    server.scene.set_up_direction("+z")
    server.scene.set_background_image(np.full((2, 2, 3), (21, 28, 39), dtype=np.uint8))
    server.gui.configure_theme(dark_mode=True, control_width="medium")
    server.scene.add_grid("/ground", width=2.5, height=2.5, cell_size=0.1, plane="xy")
    server.scene.add_frame("/robot", show_axes=False, position=(0.0, 0.0, 0.961))
    robot_view = ViserUrdf(
        server, robot, root_node_name="/robot", mesh_color_override=(0.65, 0.72, 0.78, 0.65)
    )
    robot_view.update_cfg(np.zeros(len(robot_view.get_actuated_joint_names())))
    neck_transform = robot.get_transform("Link_Neck_Pitch", robot.base_link)
    neck_rotation = Rotation.from_matrix(neck_transform[:3, :3])
    neck_position = neck_transform[:3, 3]
    server.scene.add_frame("/robot/depth_cameras", show_axes=False)
    server.gui.add_markdown(
        "## Head cameras\n"
        "Robot: public IGRIS-C URDF; all camera poses use its Link_Neck_Pitch frame. "
        "Colored rays use the full K matrix and OpenCV distortion. The rectangle is "
        "the image boundary; its middle lines cross at the image center, while the "
        "optical axis passes through (cx, cy). The right IR view is an RGB proxy."
    )
    for camera in cameras:
        camera_rotation = neck_rotation * _rotation(camera.wxyz)
        camera_position = neck_position + neck_rotation.apply(camera.position)
        root = f"/robot/depth_cameras/{camera.key}"
        frame = server.scene.add_frame(
            root, position=tuple(camera_position), wxyz=_wxyz(camera_rotation),
            axes_length=0.065, axes_radius=0.002, origin_color=camera.color,
            visible=True,
        )
        rays = server.scene.add_line_segments(
            f"{root}/intrinsic_rays", _image_plane_lines(camera, 0.24), camera.color,
            thickness=0.0025, thickness_units="world", visible=True,
        )
        marker = server.scene.add_icosphere(
            f"{root}/optical_center", radius=0.009, color=camera.color,
            visible=True,
        )
        fx, fy = camera.intrinsic[0, 0], camera.intrinsic[1, 1]
        cx, cy = camera.intrinsic[0, 2], camera.intrinsic[1, 2]
        server.gui.add_markdown(
            f"**{camera.label}**  \n"
            f"K: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f} px  \n"
            f"D: {', '.join(f'{value:.5g}' for value in camera.distortion)}  \n"
            f"Resolution: {camera.width}×{camera.height}; "
            f"neck position: ({', '.join(f'{value:.4f}' for value in camera.position)}) m"
        )
        checkbox = server.gui.add_checkbox(
            f"Show {camera.label}", initial_value=True
        )

        @checkbox.on_update
        def _toggle(event, handles=(frame, rays, marker)) -> None:
            for handle in handles:
                handle.visible = event.target.value

    server.initial_camera.position = (0.65, -0.7, 1.65)
    server.initial_camera.look_at = (0.1, 0.0, 1.4)
    server.initial_camera.up_direction = (0.0, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Validate inputs without starting Viser")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    cameras = _load_public(repo_root / PUBLIC_CONFIG)
    robot = _load_robot(repo_root / ROBOT_ASSET)
    if "Link_Neck_Pitch" not in robot.link_map or not robot.scene.geometry:
        raise RuntimeError("The public robot URDF must contain Link_Neck_Pitch and visual meshes.")
    for camera in cameras:
        print(f"{camera.label}: xyz={camera.position}, K={camera.intrinsic.tolist()}, D={camera.distortion}")
    if args.check:
        print(f"Loaded {len(robot.scene.geometry)} robot visual meshes.")
        return
    server = viser.ViserServer(host=args.host, port=args.port, label="IGRIS-C head cameras")
    _show(server, robot, cameras)
    print(f"Open http://{args.host}:{args.port} to inspect the cameras.", flush=True)
    server.sleep_forever()


if __name__ == "__main__":
    main()
