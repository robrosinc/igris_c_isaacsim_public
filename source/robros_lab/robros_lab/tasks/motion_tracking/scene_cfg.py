"""Free-base IGRIS-C motion tracking scene."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .robot_cfg import IGRISCRobotSceneCfg


@configclass
class IGRISCMotionTrackingSceneCfg(IGRISCRobotSceneCfg):
    """Standalone scene containing IGRIS-C, its sensors, ground, and lighting."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
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
