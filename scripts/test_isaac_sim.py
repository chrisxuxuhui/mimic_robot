#!/usr/bin/env python3
"""直接测试 Isaac Sim SimulationApp"""

import os
import sys
import time

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("[TEST] 直接测试 Isaac Sim SimulationApp")
print(f"[TEST] Python 版本: {sys.version}")

try:
    # 导入 Isaac Sim SimulationApp
    print("[TEST] 导入 isaacsim.SimulationApp...")
    from isaacsim import SimulationApp

    # 配置参数
    # 参考：https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_script.html
    config = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "physics_engine": "physx",  # 尝试 PhysX
        # "physics_device": "cpu",  # 尝试 CPU 物理
        "width": 1280,
        "height": 720,
    }

    print(f"[TEST] 配置: {config}")

    # 启动 SimulationApp
    print("[TEST] 启动 SimulationApp...")
    simulation_app = SimulationApp(config)

    print("[TEST] ✓ SimulationApp 初始化成功!")

    # 获取仿真上下文
    from isaacsim import SimulationContext
    sim = SimulationContext()
    print("[TEST] ✓ SimulationContext 创建成功!")

    # 等待一下
    time.sleep(2)

    print("[TEST] 关闭...")
    sim = None
    simulation_app.close()
    print("[TEST] ✓ 关闭成功!")

    print("[TEST] ✅ 所有测试通过!")

except ImportError as e:
    print(f"[TEST] ❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[TEST] ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)