# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

The project depends on Isaac Sim v5.0.0, Isaac Lab v2.2.0, and rsl_rl 2.3.1. Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Use Conda for environment management.

Install the core package:
```bash
python -m pip install -e source/whole_body_tracking
```

For motion retargeting (GMR), a separate environment is required:
```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e GMR
conda install -c conda-forge libstdcxx-ng -y
```

## Common Commands

### Motion Data Pipeline

1. **Retarget BVH to robot CSV** (GMR environment):
```bash
python scripts/bvh_to_robot.py --bvh_file MotionData/lafan1/{xxx}.bvh --robot pi_football --save_path RetargetData/lafan1/csv/pi_plus/{xxx}.csv --rate_limit
```

2. **Trim CSV** (Isaac Lab environment):
```bash
python scripts/csv_cut_pi_plus.py --input_csv RetargetData/lafan1/csv/pi_plus/pi_plus_dance1_subject2.csv --output_csv RetargetData/lafan1/csv/pi_plus/{xxx}.csv --start_frame {number} --end_frame {number} --remove_frame_column --z_offset 0.00 --decimal_places 6
```

3. **Convert CSV to NPZ** (Isaac Lab environment):
```bash
python scripts/csv_to_npz.py --robot pi_plus --input_file source/motion/hightorque/pi_plus/csv/xxx.csv --input_fps 30 --output_name source/motion/hightorque/pi_plus/npz/{motion_name} [--headless]
```

4. **Playback NPZ motion**:
```bash
python scripts/replay_npz.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```

### Training & Evaluation

5. **Train policy**:
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --headless --log_project_name pi_plus_beyondmimic
```
   - Omit `--headless` to launch Isaac Sim GUI.
   - Add `--logger wandb` to enable WandB logging (optional).
   - Use `--save_interval=10` to control checkpoint frequency.

6. **Resume training**:
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --resume {load_run_name} --headless
```

7. **Play trained policy**:
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-PI-Plus-Wo-v0 --checkpoint {logs_path_to}/model_xxx.pt --num_envs=1 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```

8. **Validate policy in MuJoCo** (sim2sim):
```bash
python scripts/sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --xml_path source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus/mjcf/pi_20dof.xml --policy_path {logs_path_to}/exported/{model_xxx}.onnx --save_json --loop
```

### Model Conversion

8. **Convert to RKNN** (for deployment):
```bash
python scripts/convert2rknn.py --input {model_path}.onnx --output {output_path}.rknn --platform {rk3568/rk3588}
```

## Architecture Overview

The codebase is organized around the **BeyondMimic** framework for humanoid robot motion tracking via reinforcement learning. Key components:

### Core MDP Modules (`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/`)
- `commands.py`: Computes pose/velocity errors, initial state randomization, adaptive sampling.
- `rewards.py`: DeepMimic reward functions and smoothing terms.
- `events.py`: Domain randomization.
- `observations.py`: Observation terms for motion tracking.
- `terminations.py`: Early termination and timeouts.

### Configuration
- `tracking_env_cfg.py`: Environment hyperparameters for tracking tasks.
- `config/pi_plus/agents/rsl_rl_ppo_cfg.py`: PPO hyperparameters for Pi Plus robot.
- `config/hi/`: Similar configurations for HI robot.
- Task configurations are loaded via Hydra (`@hydra_task_config`). The task name (e.g., `Tracking-Flat-PI-Plus-Wo-v0`) corresponds to entries in the Isaac Lab task registry.

### Robot Definitions (`source/whole_body_tracking/whole_body_tracking/robots/`)
- `pi_plus.py`, `hi.py`: Robot‑specific settings (skeleton parameters, joint stiffness/damping, action scaling).
- `actuator.py`: Actuator model utilities.

### Scripts (`scripts/`)
- **Preprocessing**: `bvh_to_robot.py`, `csv_cut_pi_plus.py`, `csv_to_npz.py`, `smplx_to_pi_plus.py`
- **Training & Playback**: `rsl_rl/train.py`, `rsl_rl/play.py`
- **Evaluation**: `sim2sim.py` (MuJoCo validation), `replay_npz.py` (motion playback)
- **Conversion**: `convert2rknn.py` (RKNN export), `upload_npz.py` (data upload)

### Data Pipeline
Human motion (BVH) → GMR retargeting → CSV → NPZ → RL training (Isaac Lab) → ONNX → MuJoCo validation → RKNN deployment.

## Environment Notes

- **Two separate conda environments** are used:
  - `gmr`: For motion retargeting (GMR). Requires Python 3.10.
  - `isaaclab`: For training and simulation (Isaac Lab). Typically Python 3.11.
- **Joint ordering mismatch**: CSV files store joints in limb‑grouped order; NPZ files store them in the robot's internal URDF order (function‑grouped). The mapping is handled by `robot.find_joints(joint_names, preserve_order=True)` in `csv_to_npz.py`.
- **Root Z offset**: CSV root Z (human height ~0.06m) may cause the robot to sink. Use `--z_offset 0.33` in `csv_cut_pi_plus.py` to raise the robot to proper height.
- **Frame count reduction**: Verify output frame count matches expected duration; interpolation in `MotionLoader._interpolate_motion` may halve duration incorrectly.
- **WandB logging**: Training logs can be sent to WandB (optional). Use `--logger wandb` in training commands; omit to disable.

## Troubleshooting

### GMR Installation Issues
When GMR is installed as an editable package (`pip install -e GMR`), Python may not properly reload module changes. If you encounter `KeyError` issues (e.g., `KeyError: 'pm01'`) or the code doesn't reflect recent changes:

1. **Reinstall GMR in the gmr environment**:
   ```bash
   conda run -n gmr pip install -e /home/ubuntu/chris/code/Mini-Pi-Plus_BeyondMimic/GMR
   ```

2. **Check that robot names are in the GMR params**:
   - Verify robot name exists in `GMR/general_motion_retargeting/params.py` dictionaries:
     - `ROBOT_XML_DICT`
     - `IK_CONFIG_DICT` (under appropriate source: `smplx`, `bvh`, `fbx`, etc.)
     - `ROBOT_BASE_DICT`
     - `VIEWER_CAM_DISTANCE_DICT`

3. **Test GMR script compatibility**:
   - Ensure robot name is listed in script argparse `choices` (e.g., in `bvh_to_robot.py`, `smplx_to_robot.py`).

Common symptoms:
- `KeyError: 'robot_name'` in `motion_retarget.py` line 25
- Robot not appearing in script help text (`--help`)
- Old code behavior persisting after modifications

## Development Guidelines

- **Code formatting**: Uses `black` and `isort` with configuration in `pyproject.toml`.
- **Type checking**: `pyright` configured with minimal strictness (`typeCheckingMode = "basic"`).
- **Imports**: Organized into sections (FUTURE, STDLIB, THIRDPARTY, ISAACLABPARTY, FIRSTPARTY, LOCALFOLDER). First‑party package is `whole_body_tracking`.
- **Pre‑commit**: Pre‑commit hooks are enabled (see badge in README).

## Important Paths

- Motion data: `source/motion/hightorque/pi_plus/` (CSV and NPZ)
- Robot assets: `source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus/`
- Logs: `logs/rsl_rl/{experiment_name}/` (default) or custom `--log_dir_path`
- Exported models: `logs/rsl_rl/{experiment_name}/.../exported/`

## Original Project

Based on [BeyondMimic (whole_body_tracking)](https://github.com/HybridRobotics/whole_body_tracking). GMR retargeting from [GMR](https://github.com/YanjieZe/GMR).