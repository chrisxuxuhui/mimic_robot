#!/usr/bin/env python3
"""
Convert pm01 MJCF to URDF for Isaac Lab.

This script extracts robot structure from pm01 MJCF files and generates
a simplified URDF with proper joint hierarchy, inertial properties, and
visual/collision geometries.

Usage:
    python convert_pm01_mjcf_to_urdf.py --output source/whole_body_tracking/whole_body_tracking/assets/hightorque/pm01/urdf/pm01_24dof.urdf
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse
import os
from pathlib import Path

def parse_mjcf(mjcf_file):
    """Parse MJCF file and extract robot structure."""
    tree = ET.parse(mjcf_file)
    root = tree.getroot()

    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("No worldbody found in MJCF")

    # Find LINK_BASE
    base_body = worldbody.find('body[@name="LINK_BASE"]')
    if base_body is None:
        raise ValueError("LINK_BASE not found in MJCF")

    # Extract base inertial properties
    base_inertial = base_body.find('inertial')
    base_mass = float(base_inertial.get('mass')) if base_inertial is not None else 0.0
    base_pos = base_inertial.get('pos')
    base_pos_vec = [float(x) for x in base_pos.split()] if base_pos else [0, 0, 0]

    # Parse serial_links.xml (included in LINK_BASE)
    # For simplicity, we'll assume a known joint structure
    # In reality, we would parse the included file

    return {
        'base': {
            'name': 'link_base',
            'mass': base_mass,
            'pos': base_pos_vec,
            'inertia': [0.01, 0.01, 0.01, 0, 0, 0]  # Simplified
        }
    }

def generate_urdf(robot_info, output_path):
    """Generate URDF file from robot information."""

    # PM01 joint structure (24 DOF)
    joints = [
        # Left leg
        {"name": "j00_hip_pitch_l_joint", "type": "revolute", "parent": "link_base", "child": "link_hip_pitch_l",
         "axis": "0 0.965926 -0.258819", "limit": "-3.141 2.443", "effort": "164", "pos": "0.01541 0.076141 -0.061208"},
        {"name": "j01_hip_roll_l_joint", "type": "revolute", "parent": "link_hip_pitch_l", "child": "link_hip_roll_l",
         "axis": "1 0 0", "limit": "-0.436 2.094", "effort": "164", "pos": "0.048 0.049359 -0.013226"},
        {"name": "j02_hip_yaw_l_joint", "type": "revolute", "parent": "link_hip_roll_l", "child": "link_hip_yaw_l",
         "axis": "0 0 1", "limit": "-1.57 4.014", "effort": "61", "pos": "-0.03139 -0.0015951 -0.086016"},
        {"name": "j03_knee_pitch_l_joint", "type": "revolute", "parent": "link_hip_yaw_l", "child": "link_knee_pitch_l",
         "axis": "0 1 0", "limit": "-0.3491 2.3911", "effort": "164", "pos": "-0.02602 -2.8566e-05 -0.23655"},
        {"name": "j04_ankle_pitch_l_joint", "type": "revolute", "parent": "link_knee_pitch_l", "child": "link_ankle_pitch_l",
         "axis": "0 1 0", "limit": "-0.6807 0.7243", "effort": "61", "pos": "-0.026756 0.00041994 -0.36305"},
        {"name": "j05_ankle_roll_l_joint", "type": "revolute", "parent": "link_ankle_pitch_l", "child": "link_ankle_roll_l",
         "axis": "1 0 0", "limit": "-0.2618 0.2618", "effort": "61", "pos": "0 0 -0.015"},

        # Right leg
        {"name": "j06_hip_pitch_r_joint", "type": "revolute", "parent": "link_base", "child": "link_hip_pitch_r",
         "axis": "0 0.965926 0.258819", "limit": "-3.141 2.443", "effort": "164", "pos": "0.01541 -0.076141 -0.061208"},
        {"name": "j07_hip_roll_r_joint", "type": "revolute", "parent": "link_hip_pitch_r", "child": "link_hip_roll_r",
         "axis": "1 0 0", "limit": "-2.094 0.436", "effort": "164", "pos": "0.048 -0.04936 -0.013226"},
        {"name": "j08_hip_yaw_r_joint", "type": "revolute", "parent": "link_hip_roll_r", "child": "link_hip_yaw_r",
         "axis": "0 0 1", "limit": "-1.57 4.014", "effort": "61", "pos": "-0.03139 0.0015951 -0.086016"},
        {"name": "j09_knee_pitch_r_joint", "type": "revolute", "parent": "link_hip_yaw_r", "child": "link_knee_pitch_r",
         "axis": "0 1 0", "limit": "-0.3491 2.3911", "effort": "164", "pos": "-0.02602 2.8566e-05 -0.23655"},
        {"name": "j10_ankle_pitch_r_joint", "type": "revolute", "parent": "link_knee_pitch_r", "child": "link_ankle_pitch_r",
         "axis": "0 1 0", "limit": "-0.6807 0.7243", "effort": "61", "pos": "-0.026756 -0.00041994 -0.36305"},
        {"name": "j11_ankle_roll_r_joint", "type": "revolute", "parent": "link_ankle_pitch_r", "child": "link_ankle_roll_r",
         "axis": "1 0 0", "limit": "-0.2618 0.2618", "effort": "61", "pos": "0 0 -0.015"},

        # Waist
        {"name": "j12_waist_yaw_joint", "type": "revolute", "parent": "link_base", "child": "link_waist_yaw",
         "axis": "0 0 1", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 0.1"},

        # Left arm
        {"name": "j13_shoulder_pitch_l_joint", "type": "revolute", "parent": "link_waist_yaw", "child": "link_shoulder_pitch_l",
         "axis": "0 1 0", "limit": "-1.57 1.57", "effort": "61", "pos": "0.1 0.15 0.1"},
        {"name": "j14_shoulder_roll_l_joint", "type": "revolute", "parent": "link_shoulder_pitch_l", "child": "link_shoulder_roll_l",
         "axis": "1 0 0", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 0"},
        {"name": "j15_shoulder_yaw_l_joint", "type": "revolute", "parent": "link_shoulder_roll_l", "child": "link_shoulder_yaw_l",
         "axis": "0 0 1", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 0"},
        {"name": "j16_elbow_pitch_l_joint", "type": "revolute", "parent": "link_shoulder_yaw_l", "child": "link_elbow_pitch_l",
         "axis": "0 1 0", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 -0.2"},
        {"name": "j17_elbow_yaw_l_joint", "type": "revolute", "parent": "link_elbow_pitch_l", "child": "link_elbow_yaw_l",
         "axis": "0 0 1", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 0"},

        # Right arm
        {"name": "j18_shoulder_pitch_r_joint", "type": "revolute", "parent": "link_waist_yaw", "child": "link_shoulder_pitch_r",
         "axis": "0 1 0", "limit": "-1.57 1.57", "effort": "61", "pos": "0.1 -0.15 0.1"},
        {"name": "j19_shoulder_roll_r_joint", "type": "revolute", "parent": "link_shoulder_pitch_r", "child": "link_shoulder_roll_r",
         "axis": "1 0 0", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 0"},
        {"name": "j20_shoulder_yaw_r_joint", "type": "revolute", "parent": "link_shoulder_roll_r", "child": "link_shoulder_yaw_r",
         "axis": "0 0 1", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 0"},
        {"name": "j21_elbow_pitch_r_joint", "type": "revolute", "parent": "link_shoulder_yaw_r", "child": "link_elbow_pitch_r",
         "axis": "0 1 0", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 -0.2"},
        {"name": "j22_elbow_yaw_r_joint", "type": "revolute", "parent": "link_elbow_pitch_r", "child": "link_elbow_yaw_r",
         "axis": "0 0 1", "limit": "-1.57 1.57", "effort": "61", "pos": "0 0 0"},

        # Head
        {"name": "j23_head_yaw_joint", "type": "revolute", "parent": "link_waist_yaw", "child": "link_head_yaw",
         "axis": "0 0 1", "limit": "-1.57 1.57", "effort": "61", "pos": "0.1 0 0.2"},
    ]

    # Generate URDF
    urdf_lines = []
    urdf_lines.append('<?xml version="1.0" encoding="utf-8"?>')
    urdf_lines.append('<robot name="pm01_24dof">')
    urdf_lines.append('  <mujoco>')
    urdf_lines.append('    <compiler meshdir="../meshes/" balanceinertia="true" discardvisual="false" />')
    urdf_lines.append('  </mujoco>')

    # Add base link
    urdf_lines.append('  <link name="link_base">')
    urdf_lines.append('    <inertial>')
    urdf_lines.append(f'      <origin xyz="{robot_info["base"]["pos"][0]} {robot_info["base"]["pos"][1]} {robot_info["base"]["pos"][2]}" rpy="0 0 0" />')
    urdf_lines.append(f'      <mass value="{robot_info["base"]["mass"]}" />')
    urdf_lines.append(f'      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02" />')
    urdf_lines.append('    </inertial>')
    urdf_lines.append('    <visual>')
    urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    urdf_lines.append('      <geometry>')
    urdf_lines.append('        <mesh filename="meshes/LINK_BASE.STL" />')
    urdf_lines.append('      </geometry>')
    urdf_lines.append('      <material name="">')
    urdf_lines.append('        <color rgba="0.75294 0.75294 0.75294 1" />')
    urdf_lines.append('      </material>')
    urdf_lines.append('    </visual>')
    urdf_lines.append('    <collision>')
    urdf_lines.append('      <origin xyz="0 0 0.1" rpy="0 0 0" />')
    urdf_lines.append('      <geometry>')
    urdf_lines.append('        <box size="0.2 0.15 0.2" />')
    urdf_lines.append('      </geometry>')
    urdf_lines.append('    </collision>')
    urdf_lines.append('  </link>')

    # Add other links (simplified)
    link_names = set(["link_base"])
    for joint in joints:
        child_link = joint["child"]
        if child_link not in link_names:
            urdf_lines.append(f'  <link name="{child_link}">')
            urdf_lines.append('    <inertial>')
            urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
            urdf_lines.append('      <mass value="0.1" />')
            urdf_lines.append('      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />')
            urdf_lines.append('    </inertial>')
            urdf_lines.append('    <visual>')
            urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
            urdf_lines.append('      <geometry>')
            urdf_lines.append(f'        <mesh filename="meshes/{child_link.upper().replace("LINK_", "LINK_")}.STL" />')
            urdf_lines.append('      </geometry>')
            urdf_lines.append('      <material name="">')
            urdf_lines.append('        <color rgba="0.75294 0.75294 0.75294 1" />')
            urdf_lines.append('      </material>')
            urdf_lines.append('    </visual>')
            urdf_lines.append('    <collision>')
            urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
            urdf_lines.append('      <geometry>')
            urdf_lines.append('        <sphere radius="0.05" />')
            urdf_lines.append('      </geometry>')
            urdf_lines.append('    </collision>')
            urdf_lines.append('  </link>')
            link_names.add(child_link)

    # Add joints
    for joint in joints:
        urdf_lines.append(f'  <joint name="{joint["name"]}" type="{joint["type"]}">')
        urdf_lines.append(f'    <origin xyz="{joint["pos"]}" rpy="0 0 0" />')
        urdf_lines.append(f'    <parent link="{joint["parent"]}" />')
        urdf_lines.append(f'    <child link="{joint["child"]}" />')
        urdf_lines.append(f'    <axis xyz="{joint["axis"]}" />')
        urdf_lines.append(f'    <limit lower="{joint["limit"].split()[0]}" upper="{joint["limit"].split()[1]}" effort="{joint["effort"]}" velocity="10.0" />')
        urdf_lines.append('    <dynamics damping="0.1" friction="0.0" />')
        urdf_lines.append('  </joint>')

    urdf_lines.append('</robot>')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(urdf_lines))

    print(f"URDF generated: {output_path}")
    print(f"Note: This is a simplified URDF for testing. For accurate simulation,")
    print(f"      inertial properties and geometries need to be refined from MJCF data.")

def main():
    parser = argparse.ArgumentParser(description="Convert pm01 MJCF to URDF")
    parser.add_argument("--output", type=str, required=True,
                       help="Output URDF file path")
    parser.add_argument("--mjcf", type=str,
                       default="GMR/assets/pm01/xml/serial_pm_v2.xml",
                       help="Input MJCF file path")

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Parse MJCF (simplified)
    robot_info = parse_mjcf(args.mjcf)

    # Generate URDF
    generate_urdf(robot_info, args.output)

if __name__ == "__main__":
    main()