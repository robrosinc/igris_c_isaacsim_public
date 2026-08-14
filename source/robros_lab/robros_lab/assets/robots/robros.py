"""Configuration for ROBROS robots.

The following configurations are available:

* :obj:`IGRIS_C_CFG`: IGRIS-C humanoid robot
* :obj:`IGRIS_C_WRIST_HAND_INDEPENDENT_CFG`: IGRIS-C with independently actuated wrists and hands

"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

# from robros_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab.assets.articulation import ArticulationCfg

from robros_lab.actuators import DelayedDCMotorWithSlackCfg, DelayedImplicitActuatorCfg

##
# Configuration - Actuators.
##


IGRISC_IMPLICIT_ACTUATOR = {
    "X12-150": DelayedImplicitActuatorCfg(
        joint_names_expr=[".*Hip_Pitch.*", ".*Knee.*"],
        # Peak speed is not listed in the datasheet; use rated speed as the hard cap for now.
        velocity_limit_sim=10.4720,
        effort_limit_sim=150.0,
        stiffness={
            ".*.*": 118.435253,
        },
        damping={
            ".*.*": 15.079645,
        },
        armature={".*Hip_Pitch.*": 0.2, ".*Knee.*": 0.15},
        friction={".*Hip_Pitch.*": 0.2, ".*Knee.*": 0.15},
        viscous_friction={".*Hip_Pitch.*": 1.5, ".*Knee.*": 1.5},
        max_delay=4,  # max delay in simulation steps
    ),
    "X8-120": DelayedImplicitActuatorCfg(
        joint_names_expr=[".*Hip_Roll.*"],
        velocity_limit_sim=16.5457,
        effort_limit_sim=120.0,
        stiffness=122.136354,
        damping=10.367256,
        armature=0.055,
        friction=0.5,
        viscous_friction=2.0,
        max_delay=4,  # max delay in simulation steps
    ),
    "X6-60": DelayedImplicitActuatorCfg(
        joint_names_expr=[".*Hip_Yaw.*", ".*Waist_Yaw.*", ".*_Shoulder_.*", ".*_Elbow_.*"],
        velocity_limit_sim=18.4307,
        effort_limit_sim=60.0,
        stiffness={
            ".*.*": 60.757285,
        },
        damping={
            ".*.*": 4.297699,
        },
        armature={
            ".*.*": 0.02,
        },
        friction={
            ".*.*": 1.01,
        },
        viscous_friction={".*.*": 1.5},
        max_delay=4,  # max delay in simulation steps
    ),
    "PLA": DelayedImplicitActuatorCfg(
        joint_names_expr=[".*Ankle.*", ".*Waist_Roll.*", ".*Waist_Pitch.*"],
        velocity_limit_sim={
            # Max Box
            ".*Waist_Roll.*": 10.1892,
            ".*Waist_Pitch.*": 18.43,
            ".*Ankle_Pitch.*": 18.1383,
            ".*Ankle_Roll.*": 19.8934,
            # # Conservative Box
            # ".*Waist_Roll.*": 6.5597,
            # ".*Waist_Pitch.*": 6.5597,
            # ".*Ankle_Pitch.*": 9.4877,
            # ".*Ankle_Roll.*": 9.4877,
        },
        effort_limit_sim={
            # Max Box
            ".*Waist_Roll.*": 217.0253,
            ".*Waist_Pitch.*": 120.0,
            ".*Ankle_Pitch.*": 163.1163,
            ".*Ankle_Roll.*": 148.7251,
            # # Conservative Box
            # ".*Waist_Roll.*": 77.28,
            # ".*Waist_Pitch.*": 77.28,
            # ".*Ankle_Pitch.*": 77.79,
            # ".*Ankle_Roll.*": 77.79,
        },
        stiffness={
            ".*Waist_Roll.*": 660.634418,
            ".*Waist_Pitch.*": 230.579719,
            ".*Ankle_Pitch.*": 93.373568,
            ".*Ankle_Roll.*": 77.634366,
        },  # P gain in Nm/rad
        damping={
            ".*Waist_Roll.*": 34.67,
            ".*Waist_Pitch.*": 12.10,
            ".*Ankle_Pitch.*": 14.863025,
            ".*Ankle_Roll.*": 12.355893,
        },
        armature={
            ".*Waist_Roll.*": 0.114,
            ".*Waist_Pitch.*": 0.047,
            ".*Ankle_Pitch.*": 0.14,
            ".*Ankle_Roll.*": 0.116,
        },
        friction={
            ".*Waist_Roll.*": 0.4,
            ".*Waist_Pitch.*": 0.2,
            ".*Ankle_Pitch.*": 0.2,
            ".*Ankle_Roll.*": 0.2,
        },
        viscous_friction={
            ".*Waist_Roll.*": 6.43,
            ".*Waist_Pitch.*": 2.5,
            ".*Ankle_Pitch.*": 0.01,
            ".*Ankle_Roll.*": 0.01,
        },
        max_delay=4,  # max delay in simulation steps
    ),
}
"""Configuration for MyActuator actuators with Implicit actuator model."""
IGRISC_EXPLICIT_ACTUATOR = {
    "X12-150": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Pitch.*", ".*Knee.*"],
        # Peak speed is not listed in the datasheet; use rated speed as the hard cap for now.
        velocity_limit=10.4720,  # 12.32,
        velocity_rated=10.4720,
        torque_rated=50.0,
        torque_limit=150.0,
        stiffness={
            ".*.*": 118.435253,
        },
        damping={
            ".*.*": 15.079645,
        },
        armature={".*Hip_Pitch.*": 0.3, ".*Knee.*": 0.18},
        friction={".*Hip_Pitch.*": 0.5, ".*Knee.*": 0.5},
        viscous_friction={".*Hip_Pitch.*": 2.4, ".*Knee.*": 2.2},
        slack_range={
            ".*Hip_Pitch.*": (0.02, 0.05),
            ".*Knee_Pitch.*": (0.02, 0.05),
        },
        max_delay=10,  # max delay in simulation steps
    ),
    "X8-120": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Roll.*"],
        velocity_limit=13.2994,  # 16.5457,
        velocity_rated=13.2994,
        torque_rated=43.0,
        torque_limit=120.0,
        stiffness=122.136354,
        damping=10.367256,
        armature=0.06,
        friction=0.1,
        viscous_friction=3.0,
        slack_range={".*Hip_Roll.*": (0.0, 0.02)},
        max_delay=10,  # max delay in simulation steps
    ),
    "X6-60": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Yaw.*", ".*Waist_Yaw.*", ".*_Shoulder_.*", ".*_Elbow_.*"],
        velocity_limit=16.0221,  # 18.4307,
        velocity_rated=16.0221,
        torque_rated=20.0,
        torque_limit=60.0,
        stiffness={
            ".*.*": 121.514570,
        },
        damping={
            ".*.*": 6.077864,
        },
        armature={
            ".*.*": 0.02,
        },
        friction={
            ".*.*": 1.01,
        },
        viscous_friction={".*.*": 1.8},
        slack_range={".*.*": (0.0, 0.02)},
        max_delay=10,  # max delay in simulation steps
    ),
    "PLA": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Ankle.*", ".*Waist_Roll.*", ".*Waist_Pitch.*"],
        velocity_limit={
            # Max Box
            ".*Waist_Roll.*": 8.6608,  # 10.1892,
            ".*Waist_Pitch.*": 15.6655,  # 18.43,
            ".*Ankle_Pitch.*": 18.1383,  # 21.3392,
            ".*Ankle_Roll.*": 19.8934,  # 23.4040,
            # # Conservative Box
            # ".*Waist_Roll.*": 6.5597,
            # ".*Waist_Pitch.*": 6.5597,
            # ".*Ankle_Pitch.*": 11.1620,
            # ".*Ankle_Roll.*": 11.1620,
        },
        velocity_rated={
            # Max Box
            ".*Waist_Roll.*": 8.6608,
            ".*Waist_Pitch.*": 15.6655,
            ".*Ankle_Pitch.*": 18.1383,
            ".*Ankle_Roll.*": 19.8934,
            # # Conservative Box
            # ".*Waist_Roll.*": 5.5757,
            # ".*Waist_Pitch.*": 5.5757,
            # ".*Ankle_Pitch.*": 9.4877,
            # ".*Ankle_Roll.*": 9.4877,
        },
        torque_rated={
            # Max Box
            ".*Waist_Roll.*": 72.3418,
            ".*Waist_Pitch.*": 40.0,
            ".*Ankle_Pitch.*": 54.3721,
            ".*Ankle_Roll.*": 49.5750,
            # # Conservative Box
            # ".*Waist_Roll.*": 25.76,
            # ".*Waist_Pitch.*": 25.76,
            # ".*Ankle_Pitch.*": 25.93,
            # ".*Ankle_Roll.*": 25.93,
        },
        torque_limit={
            # Max Box
            ".*Waist_Roll.*": 217.0253,
            ".*Waist_Pitch.*": 120.0,
            ".*Ankle_Pitch.*": 163.1163,
            ".*Ankle_Roll.*": 148.7251,
            # # Conservative Box
            # ".*Waist_Roll.*": 77.28,
            # ".*Waist_Pitch.*": 77.28,
            # ".*Ankle_Pitch.*": 77.79,
            # ".*Ankle_Roll.*": 77.79,
        },
        stiffness={
            ".*Waist_Roll.*": 660.634418,
            ".*Waist_Pitch.*": 230.579719,
            ".*Ankle_Pitch.*": 93.373568,
            ".*Ankle_Roll.*": 77.634366,
        },  # P gain in Nm/rad
        damping={
            ".*Waist_Roll.*": 34.67,
            ".*Waist_Pitch.*": 12.10,
            ".*Ankle_Pitch.*": 14.863025,
            ".*Ankle_Roll.*": 12.355893,
        },
        armature={
            ".*Waist_Roll.*": 0.114,
            ".*Waist_Pitch.*": 0.047,
            ".*Ankle_Pitch.*": 0.14,
            ".*Ankle_Roll.*": 0.12,
        },
        friction={
            ".*Waist_Roll.*": 3.6,
            ".*Waist_Pitch.*": 1.2,
            ".*Ankle_Pitch.*": 0.5,
            ".*Ankle_Roll.*": 0.5,
        },
        viscous_friction={
            ".*Waist_Roll.*": 7.5,
            ".*Waist_Pitch.*": 2.5,
            ".*Ankle_Pitch.*": 0.1,
            ".*Ankle_Roll.*": 0.1,
        },
        slack_range={
            ".*Waist_Roll.*": (0.0, 0.02),
            ".*Waist_Pitch.*": (0.0, 0.02),
            ".*Ankle_Pitch.*": (0.02, 0.05),
            ".*Ankle_Roll.*": (0.02, 0.04),
        },
        max_delay=10,  # max delay in simulation steps
    ),
}
"""Configuration for MyActuator actuators with Explicit actuator model."""
IGRISC_STIFF_EXPLICIT_ACTUATOR = {
    "X12-150": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Pitch.*", ".*Knee.*"],
        # Peak speed is not listed in the datasheet; use rated speed as the hard cap for now.
        velocity_limit=10.4720,  # 12.32,
        velocity_rated=10.4720,
        torque_rated=50.0,
        torque_limit=150.0,
        stiffness={
            ".*.*": 177.652880,
        },
        damping={
            ".*.*": 18.468718,
        },
        armature={".*Hip_Pitch.*": 0.3, ".*Knee.*": 0.18},
        friction={".*Hip_Pitch.*": 0.5, ".*Knee.*": 0.5},
        viscous_friction={".*Hip_Pitch.*": 2.4, ".*Knee.*": 2.2},
        slack_range={
            ".*Hip_Pitch.*": (0.02, 0.05),
            ".*Knee_Pitch.*": (0.02, 0.05),
        },
        max_delay=10,  # max delay in simulation steps
    ),
    "X8-120": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Roll.*"],
        velocity_limit=13.2994,  # 16.5457,
        velocity_rated=13.2994,
        torque_rated=43.0,
        torque_limit=120.0,
        stiffness=183.204531,
        damping=12.697244,
        armature=0.06,
        friction=0.1,
        viscous_friction=3.0,
        slack_range={".*Hip_Roll.*": (0.0, 0.02)},
        max_delay=10,  # max delay in simulation steps
    ),
    "X6-60": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Yaw.*", ".*Waist_Yaw.*", ".*_Shoulder_.*", ".*_Elbow_.*"],
        velocity_limit=16.0221,  # 18.4307,
        velocity_rated=16.0221,
        torque_rated=20.0,
        torque_limit=60.0,
        stiffness={
            ".*.*": 121.514570,
        },
        damping={
            ".*.*": 6.077864,
        },
        armature={
            ".*.*": 0.02,
        },
        friction={
            ".*.*": 1.01,
        },
        viscous_friction={".*.*": 1.8},
        slack_range={".*.*": (0.0, 0.02)},
        max_delay=10,  # max delay in simulation steps
    ),
    "PLA": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Ankle.*", ".*Waist_Roll.*", ".*Waist_Pitch.*"],
        velocity_limit={
            # Max Box
            ".*Waist_Roll.*": 8.6608,  # 10.1892,
            ".*Waist_Pitch.*": 15.6655,  # 18.43,
            ".*Ankle_Pitch.*": 18.1383,  # 21.3392,
            ".*Ankle_Roll.*": 19.8934,  # 23.4040,
            # # Conservative Box
            # ".*Waist_Roll.*": 6.5597,
            # ".*Waist_Pitch.*": 6.5597,
            # ".*Ankle_Pitch.*": 11.1620,
            # ".*Ankle_Roll.*": 11.1620,
        },
        velocity_rated={
            # Max Box
            ".*Waist_Roll.*": 8.6608,
            ".*Waist_Pitch.*": 15.6655,
            ".*Ankle_Pitch.*": 18.1383,
            ".*Ankle_Roll.*": 19.8934,
            # # Conservative Box
            # ".*Waist_Roll.*": 5.5757,
            # ".*Waist_Pitch.*": 5.5757,
            # ".*Ankle_Pitch.*": 9.4877,
            # ".*Ankle_Roll.*": 9.4877,
        },
        torque_rated={
            # Max Box
            ".*Waist_Roll.*": 72.3418,
            ".*Waist_Pitch.*": 40.0,
            ".*Ankle_Pitch.*": 54.3721,
            ".*Ankle_Roll.*": 49.5750,
            # # Conservative Box
            # ".*Waist_Roll.*": 25.76,
            # ".*Waist_Pitch.*": 25.76,
            # ".*Ankle_Pitch.*": 25.93,
            # ".*Ankle_Roll.*": 25.93,
        },
        torque_limit={
            # Max Box
            ".*Waist_Roll.*": 217.0253,
            ".*Waist_Pitch.*": 120.0,
            ".*Ankle_Pitch.*": 163.1163,
            ".*Ankle_Roll.*": 148.7251,
            # # Conservative Box
            # ".*Waist_Roll.*": 77.28,
            # ".*Waist_Pitch.*": 77.28,
            # ".*Ankle_Pitch.*": 77.79,
            # ".*Ankle_Roll.*": 77.79,
        },
        stiffness={
            ".*Waist_Roll.*": 660.634418,
            ".*Waist_Pitch.*": 230.579719,
            ".*Ankle_Pitch.*": 140.060352,
            ".*Ankle_Roll.*": 116.451549,
        },  # P gain in Nm/rad
        damping={
            ".*Waist_Roll.*": 34.67,
            ".*Waist_Pitch.*": 12.10,
            ".*Ankle_Pitch.*": 18.203414,
            ".*Ankle_Roll.*": 15.132817,
        },
        armature={
            ".*Waist_Roll.*": 0.114,
            ".*Waist_Pitch.*": 0.047,
            ".*Ankle_Pitch.*": 0.14,
            ".*Ankle_Roll.*": 0.12,
        },
        friction={
            ".*Waist_Roll.*": 3.6,
            ".*Waist_Pitch.*": 1.2,
            ".*Ankle_Pitch.*": 0.5,
            ".*Ankle_Roll.*": 0.5,
        },
        viscous_friction={
            ".*Waist_Roll.*": 7.5,
            ".*Waist_Pitch.*": 2.5,
            ".*Ankle_Pitch.*": 0.1,
            ".*Ankle_Roll.*": 0.1,
        },
        slack_range={
            ".*Waist_Roll.*": (0.02, 0.04),
            ".*Waist_Pitch.*": (0.02, 0.04),
            ".*Ankle_Pitch.*": (0.02, 0.05),
            ".*Ankle_Roll.*": (0.02, 0.04),
        },
        max_delay=10,  # max delay in simulation steps
    ),
}
"""Configuration for MyActuator actuators with Explicit actuator model with stiff gains."""

IGRISC_SIM_ACTUATOR = {
    "SIM_ACTUATOR": ImplicitActuatorCfg(
        joint_names_expr=[".*.*"],
        velocity_limit_sim=1000,
        effort_limit_sim=1000.0,
        stiffness={
            ".*.*": 800.0,
        },
        damping={
            ".*.*": 80.0,
        },
        armature={".*Hip_Pitch.*": 0.21, ".*Knee.*": 0.18},
        friction={".*Hip_Pitch.*": 2.5, ".*Knee.*": 2.5},
        viscous_friction={".*Hip_Pitch.*": 1.5, ".*Knee.*": 1.5},
    ),
}

IGRISC_WRIST_IMPLICIT_ACTUATOR = {
    "wrist": ImplicitActuatorCfg(
        joint_names_expr=[
            "Joint_Wrist_Yaw_Left",
            "Joint_Wrist_Roll_Left",
            "Joint_Wrist_Pitch_Left",
            "Joint_Wrist_Yaw_Right",
            "Joint_Wrist_Roll_Right",
            "Joint_Wrist_Pitch_Right",
        ],
        # Use the simulation-only drive gains and limits authored in the converted USD.
        stiffness=None,
        damping=None,
    ),
}

IGRISC_ACTIVE_HAND_IMPLICIT_ACTUATOR = {
    "active_hand": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_0_joint_thumb_proximal",
            "l_1_joint_thumb_middle",
            "l_3_joint_index_middle",
            "l_5_joint_middle_middle",
            "l_7_joint_ring_middle",
            "l_9_joint_little_middle",
            "r_0_joint_thumb_proximal",
            "r_1_joint_thumb_middle",
            "r_3_joint_index_middle",
            "r_5_joint_middle_middle",
            "r_7_joint_ring_middle",
            "r_9_joint_little_middle",
        ],
        # The ten distal follower joints are excluded from active position control.
        stiffness=None,
        damping=None,
    ),
}

IGRISC_DISTAL_HAND_IMPLICIT_ACTUATOR = {
    "distal_hand": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_2_joint_thumb_distal",
            "l_4_joint_index_distal",
            "l_6_joint_middle_distal",
            "l_8_joint_ring_distal",
            "l_10_joint_little_distal",
            "r_2_joint_thumb_distal",
            "r_4_joint_index_distal",
            "r_6_joint_middle_distal",
            "r_8_joint_ring_distal",
            "r_10_joint_little_distal",
        ],
        # Use the simulation position-drive gains authored during URDF-to-USD conversion.
        stiffness=None,
        damping=None,
    ),
}
##
# Configuration - Articulation.
##

_ROBROS_ASSET_DIR = Path(__file__).resolve().parent / "robros"
_USD_SUBDIR = Path("from_urdf") / "usd"


def _robot_usd_dir(robot_name: str, variant_dir: str | None = None) -> Path:
    usd_dir = _ROBROS_ASSET_DIR / robot_name / _USD_SUBDIR
    if variant_dir is not None:
        usd_dir = usd_dir / variant_dir
    return usd_dir


def _usd_path(robot_name: str, variant_dir: str | None = None) -> str:
    usd_path = _robot_usd_dir(robot_name, variant_dir) / f"{robot_name}.usd"
    if not usd_path.is_file():
        raise FileNotFoundError(f"Robot USD does not exist: {usd_path}")
    return str(usd_path)


IGRIS_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path("igris_c"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.961),
        joint_pos={
            # Legs (12)
            ".*_Hip_Pitch_.*": -0.2,
            ".*_Hip_Roll_.*": 0.0,
            ".*_Hip_Yaw_.*": 0.0,
            ".*_Knee_Pitch_.*": 0.3,
            ".*_Ankle_Pitch_.*": -0.1,
            ".*_Ankle_Roll_.*": 0.0,
            # Waist (3)
            ".*_Waist_.*": 0.0,
            # Left arm (4)
            ".*_Shoulder_Pitch_Left": 0.13,
            ".*_Shoulder_Roll_Left": 0.13,
            ".*_Shoulder_Yaw_Left": 0.0,
            ".*_Elbow_Pitch_Left": -0.3,
            # Right arm (4)
            ".*_Shoulder_Pitch_Right": 0.13,
            ".*_Shoulder_Roll_Right": -0.13,
            ".*_Shoulder_Yaw_Right": -0.0,
            ".*_Elbow_Pitch_Right": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=IGRISC_EXPLICIT_ACTUATOR,  # type: ignore
    soft_joint_pos_limit_factor=0.98,
)


IGRIS_C_WRIST_HAND_INDEPENDENT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path("igris_c", "extras/wrist_hand_independent"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.961),
        joint_pos={
            # Legs (12)
            ".*_Hip_Pitch_.*": -0.2,
            ".*_Hip_Roll_.*": 0.0,
            ".*_Hip_Yaw_.*": 0.0,
            ".*_Knee_Pitch_.*": 0.3,
            ".*_Ankle_Pitch_.*": -0.1,
            ".*_Ankle_Roll_.*": 0.0,
            # Waist (3)
            ".*_Waist_.*": 0.0,
            # Left arm (4)
            ".*_Shoulder_Pitch_Left": 0.13,
            ".*_Shoulder_Roll_Left": 0.13,
            ".*_Shoulder_Yaw_Left": 0.0,
            ".*_Elbow_Pitch_Left": -0.3,
            # Right arm (4)
            ".*_Shoulder_Pitch_Right": 0.13,
            ".*_Shoulder_Roll_Right": -0.13,
            ".*_Shoulder_Yaw_Right": -0.0,
            ".*_Elbow_Pitch_Right": -0.3,
            # Wrists (6)
            ".*_Wrist_Yaw_.*": 0.0,
            ".*_Wrist_Pitch_.*": 0.0,
            "Joint_Wrist_Roll_Left": 0.0,
            "Joint_Wrist_Roll_Right": 0.0,
            # Hands (22). Distal targets are subsequently generated by the hand controller.
            "[lr]_[0-9]+_joint_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        **IGRISC_EXPLICIT_ACTUATOR,
        **IGRISC_WRIST_IMPLICIT_ACTUATOR,
        **IGRISC_ACTIVE_HAND_IMPLICIT_ACTUATOR,
        **IGRISC_DISTAL_HAND_IMPLICIT_ACTUATOR,
    },  # type: ignore
    soft_joint_pos_limit_factor=0.98,
)

IGRISC_NECK_IMPLICIT_ACTUATOR = {
    "neck": ImplicitActuatorCfg(
        joint_names_expr=[".*Neck.*"],
        velocity_limit_sim=10.0,
        effort_limit_sim=20.0,
        # Placeholder PD gains for simulation-only neck control; real actuator values TBD.
        stiffness=30.0,
        damping=2.0,
        armature=0.01,
    ),
}

### Teleoperation variant: same robot with revolute neck yaw/pitch joints ###
IGRIS_C_TELEOP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path("igris_c", "extras/teleop"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            # Legs (12)
            ".*_Hip_Pitch_.*": -0.2,
            ".*_Hip_Roll_.*": 0.0,
            ".*_Hip_Yaw_.*": 0.0,
            ".*_Knee_Pitch_.*": 0.3,
            ".*_Ankle_Pitch_.*": -0.1,
            ".*_Ankle_Roll_.*": 0.0,
            # Waist (3)
            ".*_Waist_.*": 0.0,
            # Left arm (4)
            ".*_Shoulder_Pitch_Left": 0.13,
            ".*_Shoulder_Roll_Left": 0.13,
            ".*_Shoulder_Yaw_Left": 0.0,
            ".*_Elbow_Pitch_Left": -0.3,
            # Right arm (4)
            ".*_Shoulder_Pitch_Right": 0.13,
            ".*_Shoulder_Roll_Right": -0.13,
            ".*_Shoulder_Yaw_Right": -0.0,
            ".*_Elbow_Pitch_Right": -0.3,
            # Neck (2)
            ".*Neck.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={**IGRISC_EXPLICIT_ACTUATOR, **IGRISC_NECK_IMPLICIT_ACTUATOR},  # type: ignore
    soft_joint_pos_limit_factor=0.98,
)

IGRIS_C_LOCOMOTION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path("igris_c", "extras/locomotion"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            # Legs (12)
            ".*_Hip_Pitch_.*": -0.2,
            ".*_Hip_Roll_.*": 0.0,
            ".*_Hip_Yaw_.*": 0.0,
            ".*_Knee_Pitch_.*": 0.3,
            ".*_Ankle_Pitch_.*": -0.1,
            ".*_Ankle_Roll_.*": 0.0,
            # Waist (3)
            ".*_Waist_.*": 0.0,
            # Left arm (4)
            ".*_Shoulder_Pitch_Left": 0.13,
            ".*_Shoulder_Roll_Left": 0.13,
            ".*_Shoulder_Yaw_Left": 0.0,
            ".*_Elbow_Pitch_Left": -0.3,
            # Right arm (4)
            ".*_Shoulder_Pitch_Right": 0.13,
            ".*_Shoulder_Roll_Right": -0.13,
            ".*_Shoulder_Yaw_Right": -0.0,
            ".*_Elbow_Pitch_Right": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=IGRISC_EXPLICIT_ACTUATOR,  # type: ignore
    soft_joint_pos_limit_factor=0.98,
)

### Use for tracking task training ###
IGRIS_C_DUMMYHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path("igris_c", "extras/dummyhand"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            # Legs (12)
            ".*_Hip_Pitch_.*": -0.2,
            ".*_Hip_Roll_.*": 0.0,
            ".*_Hip_Yaw_.*": 0.0,
            ".*_Knee_Pitch_.*": 0.3,
            ".*_Ankle_Pitch_.*": -0.1,
            ".*_Ankle_Roll_.*": 0.0,
            # Waist (3)
            ".*_Waist_.*": 0.0,
            # Left arm (4)
            ".*_Shoulder_Pitch_Left": 0.13,
            ".*_Shoulder_Roll_Left": 0.13,
            ".*_Shoulder_Yaw_Left": 0.0,
            ".*_Elbow_Pitch_Left": -0.3,
            # Right arm (4)
            ".*_Shoulder_Pitch_Right": 0.13,
            ".*_Shoulder_Roll_Right": -0.13,
            ".*_Shoulder_Yaw_Right": -0.0,
            ".*_Elbow_Pitch_Right": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=IGRISC_STIFF_EXPLICIT_ACTUATOR,  # type: ignore
    soft_joint_pos_limit_factor=0.98,
)

### Used for motion refinement privileged training only! ###
IGRIS_C_SIM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path("igris_c", "extras/dummyhand"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators=IGRISC_SIM_ACTUATOR,  # type: ignore
)
"""Configuration for IGRIS-C, with only 23 joints allowed."""

##
# Configuration - Sensors.
##
