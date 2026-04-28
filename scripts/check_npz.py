#!/usr/bin/env python3
"""Check npz file structure."""

import numpy as np
import sys

npz_file = 'source/motion/hightorque/pm01/npz/fight1_subject.npz'

try:
    print(f"Checking {npz_file}...")
    data = np.load(npz_file)

    print(f"Keys: {list(data.keys())}")
    print("\nShapes:")
    for k in data.keys():
        print(f"  {k}: {data[k].shape}")

    print("\nSample values (first frame):")
    for k in data.keys():
        if len(data[k].shape) > 0:
            print(f"  {k}: {data[k][0] if data[k].shape[0] > 0 else 'empty'}")

    # Check specific important arrays
    if 'joint_pos' in data:
        print(f"\njoint_pos shape: {data['joint_pos'].shape}")
        print(f"Number of frames: {data['joint_pos'].shape[0]}")
        print(f"Number of joints: {data['joint_pos'].shape[1]}")

    if 'body_pos_w' in data:
        print(f"\nbody_pos_w shape: {data['body_pos_w'].shape}")
        print(f"Number of frames: {data['body_pos_w'].shape[0]}")
        print(f"Number of bodies: {data['body_pos_w'].shape[1]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()