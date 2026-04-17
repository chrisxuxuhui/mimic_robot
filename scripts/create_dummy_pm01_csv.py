#!/usr/bin/env python3
"""
Create a dummy CSV file for pm01 robot with zero joint angles.
Used for testing the csv_to_npz.py pipeline.

CSV format matches pi_plus: root pos x,y,z, root rot x,y,z,w, then joint angles.
No frame column.
"""

import csv

def main():
    output_path = "source/motion/hightorque/pm01/csv/dummy_pm01.csv"
    num_frames = 10
    fps = 30

    # Joint names as expected by whole_body_tracking pm01 config
    joint_names = [
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
    ]

    # Create header (no frame column)
    header = []
    header.extend([f'root pos {i}' for i in ['x', 'y', 'z']])
    header.extend([f'root rot {i}' for i in ['x', 'y', 'z', 'w']])
    header.extend(joint_names)

    # Generate dummy data: root at (0,0,0.35), root rotation identity, joints at zero
    root_pos = [0.0, 0.0, 0.35]
    root_rot = [0.0, 0.0, 0.0, 1.0]  # xyzw
    joint_angles = [0.0] * len(joint_names)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame in range(num_frames):
            row = []
            row.extend(root_pos)
            row.extend(root_rot)
            row.extend(joint_angles)
            writer.writerow(row)

    print(f"Created dummy CSV at {output_path} with {num_frames} frames")
    print(f"Header length: {len(header)} columns")
    print(f"Expected columns: 3 root pos + 4 root rot + {len(joint_names)} joints = {len(header)}")

if __name__ == "__main__":
    main()