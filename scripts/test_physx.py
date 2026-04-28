#!/usr/bin/env python3
"""测试使用 PhysX 而不是 Warp 作为物理引擎"""

import os
import sys
import time

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 尝试禁用某些检查
os.environ['OMNI_USER_CONFIG'] = ''

print("[TEST] 开始测试 PhysX 物理引擎")
print(f"[TEST] Python 版本: {sys.version}")

try:
    # 导入 AppLauncher
    print("[TEST] 导入 isaaclab.app...")
    from isaaclab.app import AppLauncher

    # 尝试配置使用 PhysX
    # 根据 Isaac Sim 文档，可以通过配置选择物理引擎
    config = {
        "headless": True,
        "physics_engine": "physx",  # 尝试使用 PhysX 而不是默认的 Warp
        "renderer": "RayTracedLighting",
    }

    print(f"[TEST] 启动配置: {config}")

    # 启动 AppLauncher
    print("[TEST] 启动 AppLauncher...")
    app_launcher = AppLauncher(config)
    simulation_app = app_launcher.app

    print("[TEST] ✓ Isaac Sim 初始化成功 (使用 PhysX)!")

    # 等待一下，看看是否有错误
    time.sleep(2)

    print("[TEST] 关闭 SimulationApp...")
    simulation_app.close()
    print("[TEST] ✓ SimulationApp 关闭成功!")

    print("[TEST] ✅ 所有测试通过!")

except ImportError as e:
    print(f"[TEST] ❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[TEST] ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)