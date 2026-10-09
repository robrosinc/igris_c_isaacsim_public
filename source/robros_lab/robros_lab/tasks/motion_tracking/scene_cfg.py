"""Free-base IGRIS-C motion tracking scene."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .robot_cfg import IGRISCRobotSceneCfg


def _spawn_matte_ground(prim_path: str, cfg: sim_utils.GroundPlaneCfg, *args, **kwargs):
    """Keep the default collision plane and replace its checker material."""

    prim = sim_utils.spawn_ground_plane(prim_path, cfg, *args, **kwargs)
    material_path = f"{prim_path}/Looks/MatteFloor"
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.46, 0.49, 0.52),
        roughness=0.85,
        metallic=0.0,
    )
    material.func(material_path, material)
    sim_utils.bind_visual_material(f"{prim_path}/Environment/Geometry", material_path)
    return prim


@configclass
class IGRISCMotionTrackingSceneCfg(IGRISCRobotSceneCfg):
    """Standalone scene containing IGRIS-C, its sensors, ground, and lighting."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            func=_spawn_matte_ground,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
