# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``isaacsim.asset.importer.urdf``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument("input", type=str, help="The path to the input URDF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

from pxr import Usd, UsdPhysics

MIMIC_DAMPING_RATIO = 1.0


def restore_urdf_mimic_joint_limits(urdf_path: str, usd_path: str) -> int:
    """Restore exact URDF limits and critically damp every PhysX mimic joint."""

    urdf_root = ET.parse(urdf_path).getroot()
    mimic_limits: dict[str, tuple[float, float]] = {}
    for joint_element in urdf_root.findall("joint"):
        if joint_element.find("mimic") is None:
            continue
        limit_element = joint_element.find("limit")
        if limit_element is None:
            raise ValueError(f"Mimic joint {joint_element.attrib['name']!r} has no URDF limit.")
        mimic_limits[joint_element.attrib["name"]] = (
            math.degrees(float(limit_element.attrib["lower"])),
            math.degrees(float(limit_element.attrib["upper"])),
        )

    if not mimic_limits:
        return 0

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD stage: {usd_path}")

    restored_joint_names: set[str] = set()
    for prim in stage.TraverseAll():
        joint_name = prim.GetName()
        if joint_name not in mimic_limits or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        lower_limit, upper_limit = mimic_limits[joint_name]
        mimic_schemas = [
            schema
            for schema in prim.GetAppliedSchemas()
            if schema.startswith("PhysxMimicJointAPI:")
        ]
        if len(mimic_schemas) != 1:
            raise RuntimeError(
                f"Mimic joint {joint_name!r} must have exactly one PhysX mimic axis, "
                f"got {mimic_schemas}."
            )
        mimic_axis = mimic_schemas[0].rsplit(":", maxsplit=1)[-1]
        damping_ratio_attr = prim.GetAttribute(f"physxMimicJoint:{mimic_axis}:dampingRatio")
        damping_ratio_attr.Set(MIMIC_DAMPING_RATIO)
        revolute_joint = UsdPhysics.RevoluteJoint(prim)
        revolute_joint.GetLowerLimitAttr().Set(lower_limit)
        revolute_joint.GetUpperLimitAttr().Set(upper_limit)
        restored_joint_names.add(joint_name)

    missing_joint_names = set(mimic_limits) - restored_joint_names
    if missing_joint_names:
        raise RuntimeError(
            "Generated USD is missing URDF mimic joints: "
            f"{sorted(missing_joint_names)}"
        )

    stage.GetRootLayer().Save()
    return len(restored_joint_names)


def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Create Urdf converter config
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        # In the installed converter this flag is passed directly to the URDF importer's `parse_mimic` option.
        # It must be enabled to preserve URDF <mimic> joints as PhysX mimic joints in the generated USD.
        convert_mimic_joints_to_normal_joints=True,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0)
        ),
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Urdf converter and import the file
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    restored_mimic_limit_count = restore_urdf_mimic_joint_limits(
        urdf_path, urdf_converter.usd_path
    )
    # print output
    print("URDF importer output:")
    print(f"Generated USD file: {urdf_converter.usd_path}")
    print(f"Restored exact URDF limits for {restored_mimic_limit_count} mimic joints")
    print("-" * 80)
    print(
        f"Set damping ratio to {MIMIC_DAMPING_RATIO} for {restored_mimic_limit_count} mimic joints"
    )
    print("-" * 80)

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(urdf_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
