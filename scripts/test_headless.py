#!/usr/bin/env python3
"""最简单的 Isaac Sim headless 模式测试"""

import os
import sys

# 设置 CUDA 环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("[TEST] 开始 Isaac Sim headless 模式测试")
print(f"[TEST] Python 版本: {sys.version}")
print(f"[TEST] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

try:
    # 导入 AppLauncher
    print("[TEST] 导入 isaaclab.app...")
    from isaaclab.app import AppLauncher

    # 启动 headless 模式
    print("[TEST] 启动 AppLauncher (headless=True)...")
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    print("[TEST] ✓ Isaac Sim headless 模式初始化成功!")

    # 立即关闭
    print("[TEST] 关闭 SimulationApp...")
    simulation_app.close()
    print("[TEST] ✓ SimulationApp 关闭成功!")

    print("[TEST] ✅ 所有测试通过!")

except Exception as e:
    print(f"[TEST] ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)