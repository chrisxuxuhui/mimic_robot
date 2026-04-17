# PM01 机器人集成进展报告

**日期：** 2026年4月16日  
**状态：** 主要配置已完成，待全面测试

## ✅ 已完成的任务

### 1. 修复 pm01 关节名称配置
**文件：** `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/pm01/flat_env_cfg.py`

已添加完整的 24 个关节名称配置，遵循 HI 机器人模式：
```python
self.actions.joint_pos.joint_names = [
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
self.commands.motion.joint_names = self.actions.joint_pos.joint_names
```

### 2. 更新所有 GMR 脚本支持 PM01
已为以下 GMR 脚本添加 `pm01` 选项：

| 脚本文件 | 状态 |
|----------|------|
| `GMR/scripts/smplx_to_robot_dataset.py` | ✅ 已添加 |
| `GMR/scripts/bvh_to_robot_dataset.py` | ✅ 已添加 |
| `GMR/scripts/optitrack_to_robot.py` | ✅ 已添加 |
| `GMR/scripts/vis_robot_motion.py` | ✅ 已添加 |

**注意：** 所有脚本已统一使用 `pm01` 命名约定

### 3. 验证 URDF 链接名称匹配
**文件：** `source/whole_body_tracking/whole_body_tracking/assets/hightorque/pm01/urdf/pm01_24dof.urdf`

已验证配置中的所有身体链接名称均存在于 URDF 文件中：
- ✅ `link_base` - 基础链接
- ✅ `link_hip_roll_l`, `link_knee_pitch_l`, `link_ankle_roll_l` - 左腿链接
- ✅ `link_hip_roll_r`, `link_knee_pitch_r`, `link_ankle_roll_r` - 右腿链接  
- ✅ `link_shoulder_roll_l`, `link_elbow_pitch_l`, `link_elbow_yaw_l` - 左臂链接
- ✅ `link_shoulder_roll_r`, `link_elbow_pitch_r`, `link_elbow_yaw_r` - 右臂链接

### 4. 脚本兼容性更新
以下核心脚本已添加 `pm01` 选项支持：

| 脚本文件 | 用途 | 状态 |
|----------|------|------|
| `scripts/csv_to_npz.py` | CSV 转 NPZ 转换 | ✅ 已支持 `pm01` |
| `scripts/replay_npz.py` | NPZ 运动回放 | ✅ 已支持 `pm01` |
| `scripts/sim2sim.py` | 仿真到仿真验证 | ✅ 已支持 `pm01` |
| `scripts/bvh_to_robot_headless.py` | BVH 重定向 | ✅ 已支持 `pm01` |

## 🔄 待测试的完整流水线

### 测试环境要求
1. **GMR 环境** (Python 3.10): 运动重定向相关脚本
2. **Isaac Lab 环境** (Python 3.11): 训练、回放、仿真相关功能

### 测试步骤

#### GMR 环境测试
```bash
# 1. BVH 到机器人 CSV 重定向
python scripts/bvh_to_robot.py --robot pm01 --bvh_file MotionData/lafan1/{xxx}.bvh --save_path RetargetData/lafan1/csv/pm01/{xxx}.csv

# 2. SMPL-X 数据重定向 (如果适用)
python GMR/scripts/smplx_to_robot_dataset.py --robot pm01 --src_folder {src} --tgt_folder {tgt}
```

#### Isaac Lab 环境测试
```bash
# 3. CSV 到 NPZ 转换
python scripts/csv_to_npz.py --robot pm01 --input_file source/motion/hightorque/pm01/csv/{xxx}.csv --input_fps 30 --output_name source/motion/hightorque/pm01/npz/{motion_name}

# 4. NPZ 运动回放
python scripts/replay_npz.py --robot pm01 --motion_file source/motion/hightorque/pm01/npz/{motion_name}.npz

# 5. 环境加载测试
python -c "import gymnasium as gym; gym.make('Tracking-Flat-PM01-v0')"

# 6. 仿真到仿真验证
python scripts/sim2sim.py --robot pm01 --motion_file source/motion/hightorque/pm01/npz/{motion_name}.npz --xml_path source/whole_body_tracking/whole_body_tracking/assets/hightorque/pm01/urdf/pm01_24dof.urdf --policy_path {policy}.onnx
```

### 训练流程测试
```bash
# 7. 策略训练
python scripts/rsl_rl/train.py --task=Tracking-Flat-PM01-Wo-v0 --motion_file source/motion/hightorque/pm01/npz/{motion_name}.npz --headless --log_project_name pm01_beyondmimic

# 8. 策略回放
python scripts/rsl_rl/play.py --task=Tracking-Flat-PM01-Wo-v0 --checkpoint {checkpoint}.pt --num_envs=1 --motion_file source/motion/hightorque/pm01/npz/{motion_name}.npz
```

## ⚠️ 已知问题

1. **环境依赖**: 测试需要切换不同的 conda 环境：
   - `gmr` 环境用于 GMR 相关脚本
   - `isaaclab` 环境用于训练和仿真脚本

## 📋 下一步建议

1. **优先测试** CSV 到 NPZ 转换流程，验证关节顺序匹配
2. **验证** `Tracking-Flat-PM01-v0` 和 `Tracking-Flat-PM01-Wo-v0` 环境正确加载
3. **检查** PM01 的 24 DOF 是否与动作空间维度匹配
4. **确认** 运动数据的根位置偏移处理正确（Z轴高度）

## 📁 关键文件位置

- **机器人配置**: `source/whole_body_tracking/whole_body_tracking/robots/pm01.py`
- **环境配置**: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/pm01/`
- **URDF 文件**: `source/whole_body_tracking/whole_body_tracking/assets/hightorque/pm01/urdf/pm01_24dof.urdf`
- **运动数据**: `source/motion/hightorque/pm01/` (需手动创建目录结构)

---

**最后更新**: 2026年4月16日  
**更新者**: Claude Code