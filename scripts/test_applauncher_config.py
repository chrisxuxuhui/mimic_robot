#!/usr/bin/env python3
"""Test AppLauncher with config dict for headless video recording."""

import os
import sys
import time

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("[TEST] Testing AppLauncher with config dict")
print(f"[TEST] Python version: {sys.version}")

try:
    # Import AppLauncher
    print("[TEST] Importing isaaclab.app...")
    from isaaclab.app import AppLauncher

    # Configuration for headless video recording
    config = {
        "headless": True,
        "physics_engine": "physx",  # Use PhysX instead of Warp
        "renderer": "PathTracing",  # Simpler renderer than RayTracedLighting
        "width": 1280,
        "height": 720,
        "enable_cameras": True,
    }

    print(f"[TEST] Config: {config}")

    # Launch AppLauncher with config dict
    print("[TEST] Launching AppLauncher...")
    app_launcher = AppLauncher(config)
    simulation_app = app_launcher.app

    print("[TEST] ✓ AppLauncher initialized successfully!")

    # Wait a bit to see if any errors occur
    time.sleep(2)

    print("[TEST] Closing simulation app...")
    simulation_app.close()
    print("[TEST] ✓ Simulation app closed!")

    print("[TEST] ✅ All tests passed!")

except Exception as e:
    print(f"[TEST] ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)