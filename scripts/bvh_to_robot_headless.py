#!/usr/bin/env python3
"""
Headless BVH to robot CSV conversion.
Skips visualization for use in headless environments.
"""

import argparse
import pathlib
import time
import csv
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from rich import print
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "pm01","pi_football","hightorque_hi"],
        default="unitree_g1",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
        required=True,
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (for testing).",
    )

    args = parser.parse_args()

    robot_for_gmr = args.robot

    # Robot-specific joint names and reordering mappings
    # For PM01: 24 DOF, joint names match GMR output order, no reordering needed
    # For pi_football: 22 DOF, needs reordering from GMR order to CSV order
    # For other robots: default to pi_football mapping (may need adjustment)
    robot_joint_configs = {
        "pm01": {
            "joint_names": [
                "j00_hip_pitch_l_joint",
                "j01_hip_roll_l_joint",
                "j02_hip_yaw_l_joint",
                "j03_knee_pitch_l_joint",
                "j04_ankle_pitch_l_joint",
                "j05_ankle_roll_l_joint",
                "j06_hip_pitch_r_joint",
                "j07_hip_roll_r_joint",
                "j08_hip_yaw_r_joint",
                "j09_knee_pitch_r_joint",
                "j10_ankle_pitch_r_joint",
                "j11_ankle_roll_r_joint",
                "j12_waist_yaw_joint",
                "j13_shoulder_pitch_l_joint",
                "j14_shoulder_roll_l_joint",
                "j15_shoulder_yaw_l_joint",
                "j16_elbow_pitch_l_joint",
                "j17_elbow_yaw_l_joint",
                "j18_shoulder_pitch_r_joint",
                "j19_shoulder_roll_r_joint",
                "j20_shoulder_yaw_r_joint",
                "j21_elbow_pitch_r_joint",
                "j22_elbow_yaw_r_joint",
                "j23_head_yaw_joint",
            ],
            "reorder_indices": list(range(24)),  # no reordering
        },
        "pi_football": {
            "joint_names": [
                # left leg
                'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
                # right leg
                'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
                # left arm
                'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
                # right arm
                'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
            ],
            "reorder_indices": [
                # left leg (keep original)
                0, 1, 2, 3, 4, 5,
                # right leg (move from 11-16 to 6-11)
                11, 12, 13, 14, 15, 16,
                # left arm (move from 6-10 to 12-16)
                6, 7, 8, 9, 10,
                # right arm (keep 17-21)
                17, 18, 19, 20, 21
            ],
        },
    }

    # Default configuration (pi_football) for other robots
    default_config = robot_joint_configs["pi_football"]
    robot_config = robot_joint_configs.get(args.robot, default_config)
    joint_names = robot_config["joint_names"]
    reorder_indices = robot_config["reorder_indices"]

    save_dir = os.path.dirname(args.save_path)
    if save_dir:  # Only create directory if it's not empty
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []

    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_lafan1_file(args.bvh_file)

    if args.max_frames is not None and args.max_frames < len(lafan1_data_frames):
        lafan1_data_frames = lafan1_data_frames[:args.max_frames]

    # Initialize the retargeting system
    retargeter = GMR(
        src_human="bvh",
        tgt_robot=robot_for_gmr,
        actual_human_height=actual_human_height,
    )

    motion_fps = 30

    print(f"Retargeting {len(lafan1_data_frames)} frames for robot {args.robot}...")

    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")

    # Process frames without visualization
    for i in range(len(lafan1_data_frames)):
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)

        qpos_list.append(qpos)

    pbar.close()

    # Save CSV
    print("Saving CSV...")
    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # save from wxyz to xyzw
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    # Reorder joint angles: from left leg → left arm → right leg → right arm to left leg → right leg → left arm → right arm
    # Current order: 0-5(left leg), 6-10(left arm), 11-16(right leg), 17-21(right arm)
    # Desired order: 0-5(left leg), 6-11(right leg), 12-16(left arm), 17-21(right arm)
    reorder_indices = [
        # left leg (keep original)
        0, 1, 2, 3, 4, 5,           # l_hip_pitch → l_ankle_roll
        # right leg (move from 11-16 to 6-11)
        11, 12, 13, 14, 15, 16,     # r_hip_pitch → r_ankle_roll
        # left arm (move from 6-10 to 12-16)
        6, 7, 8, 9, 10,             # l_shoulder_pitch → l_wrist
        # right arm (keep 17-21)
        17, 18, 19, 20, 21          # r_shoulder_pitch → r_wrist
    ]
    dof_pos = dof_pos[:, reorder_indices]

    # Save as CSV format
    with open(args.save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        header = ['frame']
        header.extend([f'root pos {i}' for i in ['x', 'y','z']])
        header.extend([f'root rot {i}' for i in ['x','y','z','w']])

        # joint names in reordered order
        joint_names = [
            # left leg
            'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll',
            # right leg
            'r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
            # left arm
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_upper_arm', 'l_elbow', 'l_wrist',
            # right arm
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_upper_arm', 'r_elbow', 'r_wrist'
        ]
        header.extend(joint_names)
        writer.writerow(header)

        # Write data per frame
        for frame_idx in range(len(qpos_list)):
            row = [frame_idx]
            row.extend(root_pos[frame_idx].tolist())
            row.extend(root_rot[frame_idx].tolist())
            row.extend(dof_pos[frame_idx].tolist())
            writer.writerow(row)
    print(f"Saved to {args.save_path}")

    print("Done.")