"""Build the standalone appearance layer for the public IGRIS-C USD."""

from __future__ import annotations

import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdShade


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ROOT = REPO_ROOT / "source/robros_lab/robros_lab/assets/robots/robros/igris_c"
APPEARANCE_DIR = ROBOT_ROOT / "appearance"
USD_DIR = ROBOT_ROOT / "from_urdf/usd/extras/wrist_hand_independent"
FACE_USD = USD_DIR / "igris_c_face_outer.usdc"
TORSO_USD = USD_DIR / "igris_c_torso_outer.usdc"
ROOT_PRIM = "/IGRIS_C_WRIST_HAND"


def read_materials(path: Path) -> dict[str, dict]:
    materials: dict[str, dict] = {}
    current: dict | None = None
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


def build(output: Path, base_usd: Path) -> None:
    for asset in (FACE_USD, TORSO_USD):
        if not asset.is_file():
            raise FileNotFoundError(asset)
    appearance = json.loads((APPEARANCE_DIR / "link_materials.json").read_text())
    mapping = appearance["links"]
    materials = read_materials(APPEARANCE_DIR / "materials.mtl")
    materials.update(appearance.get("materials", {}))
    layer = Sdf.Layer.CreateAnonymous("igris_c_colored.usda")
    layer.defaultPrim = "IGRIS_C_WRIST_HAND"
    layer.subLayerPaths = [str(FACE_USD), str(TORSO_USD), str(base_usd)]
    stage = Usd.Stage.Open(layer)
    stage.SetEditTarget(layer)
    stage.DefinePrim(ROOT_PRIM + "/Looks_Painted", "Scope")

    for name in sorted(set(mapping.values())):
        values = materials[name]
        material_path = f"{ROOT_PRIM}/Looks_Painted/{name}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*values["color"]))
        for field in ("metallic", "roughness", "opacity"):
            shader.CreateInput(field, Sdf.ValueTypeNames.Float).Set(values[field])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    torso_mat_25 = UsdShade.Shader.Get(stage, f"{ROOT_PRIM}/Looks_TorsoOuter/mat_25/PreviewSurface")
    if not torso_mat_25:
        raise ValueError("Missing torso mat_25 PreviewSurface")
    arm_material = materials["mat_0"]
    torso_mat_25.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*arm_material["color"])
    )
    for field in ("metallic", "roughness", "opacity"):
        torso_mat_25.CreateInput(field, Sdf.ValueTypeNames.Float).Set(arm_material[field])

    for link, material_name in mapping.items():
        visual = stage.GetPrimAtPath(f"{ROOT_PRIM}/{link}/visuals")
        if not visual or not visual.IsInstance():
            raise ValueError(f"No instanced USD visual for {link}")
        material = UsdShade.Material.Get(stage, f"{ROOT_PRIM}/Looks_Painted/{material_name}")
        UsdShade.MaterialBindingAPI.Apply(visual).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    authored = layer.ExportToString()
    for asset in (FACE_USD, TORSO_USD, base_usd):
        absolute_reference = f"@{asset}@"
        if authored.count(absolute_reference) != 1:
            raise RuntimeError(f"Could not locate the USD sublayer reference: {asset}")
        authored = authored.replace(absolute_reference, f"@{asset.name}@", 1)
    output.write_text(authored.rstrip() + "\n")


if __name__ == "__main__":
    build(USD_DIR / "igris_c_colored.usda", USD_DIR / "igris_c.usd")
