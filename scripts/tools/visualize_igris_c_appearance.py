"""Compare the public IGRIS-C visuals before and after the appearance pass."""

from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ROOT = (
    REPO_ROOT
    / "source/robros_lab/robros_lab/assets/robots/robros/igris_c"
)
URDF_PATH = ROBOT_ROOT / "from_urdf/urdf/igris_c_v2_wrist_hand_isaac.urdf"
APPEARANCE_DIR = ROBOT_ROOT / "appearance"
PAINTED_USD_PATH = (
    ROBOT_ROOT
    / "from_urdf/usd/extras/wrist_hand_independent/igris_c_colored.usda"
)
FACE_ROOT = "/IGRIS_C_WRIST_HAND/Link_Neck_Pitch/face_outer"
TORSO_ROOT = "/IGRIS_C_WRIST_HAND/base_link/torso_outer"


def read_materials(path: Path) -> dict[str, dict[str, float | tuple[float, ...]]]:
    """Read the copied Blender OBJ material colors and PBR factors."""
    materials: dict[str, dict[str, float | tuple[float, ...]]] = {}
    current: dict[str, float | tuple[float, ...]] | None = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "newmtl":
            current = {}
            materials[fields[1]] = current
        elif current is not None and fields[0] == "Kd":
            current["color"] = tuple(map(float, fields[1:4]))
        elif current is not None and fields[0] == "Pm":
            current["metallic"] = float(fields[1])
        elif current is not None and fields[0] == "Pr":
            current["roughness"] = float(fields[1])
        elif current is not None and fields[0] == "d":
            current["opacity"] = float(fields[1])
    return materials


def origin_matrix(origin: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4)
    if origin is None:
        return matrix
    roll, pitch, yaw = (float(x) for x in origin.get("rpy", "0 0 0").split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix[:3, :3] = (
        np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
        @ np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
        @ np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    )
    matrix[:3, 3] = tuple(float(x) for x in origin.get("xyz", "0 0 0").split())
    return matrix


def link_poses(robot: ET.Element) -> dict[str, np.ndarray]:
    """Compute the zero-joint URDF pose of each link."""
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    child_names = set()
    for joint in robot.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children.setdefault(parent, []).append((child, origin_matrix(joint.find("origin"))))
        child_names.add(child)
    roots = {link.get("name") for link in robot.findall("link")} - child_names
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root link, found {sorted(roots)}")
    root = roots.pop()
    poses = {root: np.eye(4)}
    pending = [root]
    while pending:
        parent = pending.pop()
        for child, transform in children.get(parent, []):
            poses[child] = poses[parent] @ transform
            pending.append(child)
    return poses


def load_spec() -> tuple[ET.Element, dict[str, str], dict[str, dict]]:
    robot = ET.parse(URDF_PATH).getroot()
    appearance = json.loads((APPEARANCE_DIR / "link_materials.json").read_text())
    mapping = appearance["links"]
    materials = read_materials(APPEARANCE_DIR / "materials.mtl")
    materials.update(appearance.get("materials", {}))
    links = {link.get("name"): link for link in robot.findall("link")}
    for name, material in mapping.items():
        if name not in links or links[name].find("visual") is None:
            raise ValueError(f"No public visual for {name}")
        if material not in materials:
            raise ValueError(f"Missing material {material}")
    return robot, mapping, materials


def load_selected_visuals(
    path: Path, root_path: str, hidden_visual_path: str,
) -> list[tuple[str, np.ndarray, np.ndarray, dict, bool]]:
    """Read the selected meshes composed into the painted robot USD."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(path)
    original = UsdGeom.Imageable(
        stage.GetPrimAtPath(hidden_visual_path)
    )
    if original.GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible:
        raise ValueError(f"The original visual is still visible: {hidden_visual_path}")
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Missing selected mesh root: {root_path}")

    visuals = []
    for prim in root.GetChildren():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            raise ValueError(f"Unexpected selected prim: {prim.GetPath()}")
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f"Non-triangular selected mesh: {prim.GetPath()}")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.uint32).reshape(-1, 3)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            raise ValueError(f"Missing selected material: {prim.GetPath()}")
        shader_prim = stage.GetPrimAtPath(str(material.GetPath()) + "/PreviewSurface")
        shader = UsdShade.Shader(shader_prim)
        if not shader:
            raise ValueError(f"Missing selected shader: {material.GetPath()}")
        paint = {
            "color": tuple(shader.GetInput("diffuseColor").Get()),
            "metallic": shader.GetInput("metallic").Get(),
            "roughness": shader.GetInput("roughness").Get(),
            "opacity": shader.GetInput("opacity").Get(),
        }
        visuals.append((
            prim.GetName(), points, faces, paint,
            bool(mesh.GetDoubleSidedAttr().Get()),
        ))
    if not visuals:
        raise ValueError(f"No selected render meshes: {path}")
    return visuals


def load_face_visuals() -> list[tuple[str, np.ndarray, np.ndarray, dict, bool]]:
    return load_selected_visuals(
        PAINTED_USD_PATH, FACE_ROOT, "/IGRIS_C_WRIST_HAND/Link_Neck_Pitch/visuals",
    )


def load_torso_visuals() -> list[tuple[str, np.ndarray, np.ndarray, dict, bool]]:
    return load_selected_visuals(
        PAINTED_USD_PATH, TORSO_ROOT, "/IGRIS_C_WRIST_HAND/base_link/visuals",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="validate assets without starting viser")
    args = parser.parse_args()

    robot, mapping, materials = load_spec()
    face_visuals = load_face_visuals()
    torso_visuals = load_torso_visuals()
    poses = link_poses(robot)
    visuals = []
    for link in robot.findall("link"):
        name = link.get("name")
        for visual in link.findall("visual"):
            mesh_spec = visual.find("geometry/mesh")
            if mesh_spec is None:
                continue
            mesh_path = (URDF_PATH.parent / mesh_spec.get("filename")).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            scale = np.array(tuple(float(x) for x in mesh_spec.get("scale", "1 1 1").split()))
            color_spec = visual.find("material/color")
            original = tuple(float(x) for x in color_spec.get("rgba").split()) if color_spec is not None else (1.0, 1.0, 1.0, 1.0)
            paint = materials[mapping[name]] if name in mapping else None
            transform = poses[name] @ origin_matrix(visual.find("origin"))
            visuals.append((name, mesh_path, scale, original, paint, transform))
    print(
        f"Validated {len(visuals)} URDF visuals, {len(face_visuals)} face meshes, "
        f"{len(torso_visuals)} torso meshes; "
        f"{len(mapping)} links receive materials"
    )
    if args.check:
        return

    import trimesh
    import viser

    server = viser.ViserServer(host=args.host, port=args.port, label="IGRIS-C appearance")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=3.0, height=3.0, position=(0, 0, -0.02))
    before = server.scene.add_frame("/Before", show_axes=False, position=(-0.7, 0, 0))
    after = server.scene.add_frame("/After", show_axes=False, position=(0.7, 0, 0))
    server.scene.add_label("/Before/label", "Original", position=(0, 0, 1.9))
    server.scene.add_label("/After/label", "Painted", position=(0, 0, 1.9))

    def add_pbr_mesh(path: str, points: np.ndarray, faces: np.ndarray,
                     color: tuple[float, ...], opacity: float,
                     metallic: float, roughness: float,
                     double_sided: bool = True) -> None:
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        rgba = [round(max(0.0, min(1.0, value)) * 255) for value in (*color, opacity)]
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba, metallicFactor=metallic, roughnessFactor=roughness,
            alphaMode="BLEND" if opacity < 1.0 else "OPAQUE", doubleSided=double_sided,
        ))
        server.scene.add_mesh_trimesh(path, mesh=mesh)

    for name, mesh_path, scale, original, paint, transform in visuals:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        points = np.asarray(mesh.vertices, dtype=np.float32) * scale
        points = points @ transform[:3, :3].T + transform[:3, 3]
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        add_pbr_mesh(f"/Before/{name}", points, faces, original[:3], original[3], 0.0, 0.5)
        if name not in ("Link_Neck_Pitch", "base_link"):
            color = paint["color"] if paint is not None else original[:3]
            opacity = paint.get("opacity", 1.0) if paint is not None else original[3]
            metallic = paint.get("metallic", 0.0) if paint is not None else 0.0
            roughness = paint.get("roughness", 0.5) if paint is not None else 0.5
            add_pbr_mesh(f"/After/{name}", points, faces, color, opacity, metallic, roughness)

    for link, selected in (
        ("Link_Neck_Pitch", face_visuals),
        ("base_link", torso_visuals),
    ):
        pose = poses[link]
        for name, points, faces, paint, double_sided in selected:
            points = points @ pose[:3, :3].T + pose[:3, 3]
            add_pbr_mesh(
                f"/After/{link}/{name}",
                points, faces, paint["color"], paint["opacity"],
                paint["metallic"], paint["roughness"], double_sided,
            )

    with server.gui.add_folder("Display"):
        show_before = server.gui.add_checkbox("Original", True)
        show_after = server.gui.add_checkbox("Painted", True)

    @show_before.on_update
    def _(_) -> None:
        before.visible = show_before.value

    @show_after.on_update
    def _(_) -> None:
        after.visible = show_after.value

    server.initial_camera.position = (2.5, -3.0, 1.6)
    server.initial_camera.look_at = (0, 0, 0.85)
    print(f"Open http://{args.host}:{args.port} to compare the two robots")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
