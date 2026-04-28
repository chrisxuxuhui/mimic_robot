"""Script to play a trained policy checkpoint and record video with custom camera control.

.. code-block:: bash

    # Basic usage with auto-naming:
    python scripts/rsl_rl/record_policy_video.py --task=Tracking-Flat-PI-Plus-Wo-v0 \\
        --motion_file source/motion/hightorque/pi_plus/npz/motion.npz \\
        --headless

    # Custom output path and resolution:
    python scripts/rsl_rl/record_policy_video.py --task=Tracking-Flat-PI-Plus-Wo-v0 \\
        --motion_file source/motion/hightorque/pi_plus/npz/motion.npz \\
        --output_video recordings/my_motion.mp4 --resolution 1920x1080 --fps 30 \\
        --headless

    # Resume from a specific run and record:
    python scripts/rsl_rl/record_policy_video.py --task=Tracking-Flat-PI-Plus-Wo-v0 \\
        --motion_file source/motion/hightorque/pi_plus/npz/motion.npz \\
        --resume my_run_name --output_video my_motion.mp4 \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Record video of a trained policy playback.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--output_video", type=str, default=None, help="Output video file path (e.g., output.mp4). Auto-generated if not set.")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for video recording.")
parser.add_argument("--resolution", type=str, default="1280x720", help="Video resolution WxH (e.g., 1280x720).")
parser.add_argument("--duration", type=float, default=None, help="Recording duration in seconds (default: video_length * dt).")
parser.add_argument("--camera_distance", type=float, default=2.5, help="Camera distance from robot.")
parser.add_argument("--camera_height", type=float, default=1.0, help="Camera height offset from robot root.")
parser.add_argument("--no_export_onnx", action="store_true", default=False, help="Skip ONNX export.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras for video recording (required for headless + rgb_array rendering)
args_cli.enable_cameras = True

# auto-enable video flag when output_video is specified
if args_cli.output_video:
    args_cli.video = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _build_record_video_kwargs(output_video, video_length, fps, task_name):
    """Build kwargs for gym.wrappers.RecordVideo from CLI args."""
    if output_video:
        video_dir = os.path.dirname(output_video) or "."
        video_name = os.path.splitext(os.path.basename(output_video))[0]
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = task_name.replace("Tracking-Flat-", "").replace("-", "_") if task_name else "unknown"
        video_name = f"policy_{safe_name}_{timestamp}"
        video_dir = "."

    return {
        "video_folder": os.path.abspath(video_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": video_length,
        "name_prefix": video_name,
        "disable_logger": True,
        "fps": fps,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent and record video."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # override viewer resolution from CLI (RecordVideo renders via env_cfg.viewer.resolution)
    w, h = map(int, args_cli.resolution.split("x"))
    env_cfg.viewer.resolution = (w, h)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = None
    if agent_cfg.load_checkpoint is not None:
        potential_path = agent_cfg.load_checkpoint
        if os.path.isabs(potential_path) or os.sep in potential_path:
            if os.path.isfile(potential_path):
                resume_path = os.path.abspath(potential_path)
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {potential_path}")

    if resume_path is None:
        run_dir_expr = agent_cfg.load_run if isinstance(agent_cfg.load_run, str) and len(agent_cfg.load_run) > 0 else ".*"
        checkpoint_expr = (
            agent_cfg.load_checkpoint if (isinstance(agent_cfg.load_checkpoint, str) and os.sep not in agent_cfg.load_checkpoint)
            else ".*"
        )
        resume_path = get_checkpoint_path(log_root_path, run_dir_expr, checkpoint_expr)

    # create isaac environment with rgb_array rendering for video recording
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # setup video recording via gymnasium RecordVideo wrapper
    video_kwargs = _build_record_video_kwargs(
        args_cli.output_video, args_cli.video_length, args_cli.fps, args_cli.task
    )
    print("[INFO] Recording video during playback.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    log_dir = os.path.dirname(resume_path)
    sim = env.unwrapped.sim

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if not args_cli.no_export_onnx:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        exported_policy_name = resume_path.split('/')[-1]
        exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')

        export_motion_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename=exported_onnx_name,
        )
        attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir, exported_onnx_name)

    # Determine recording steps
    if args_cli.duration is not None:
        sim_dt = sim.get_physics_dt()
        steps_to_record = int(args_cli.duration / sim_dt)
        print(f"[INFO]: Recording duration: {args_cli.duration}s -> {steps_to_record} steps")
    else:
        steps_to_record = args_cli.video_length

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        # Track camera to follow robot (set before env.step so RecordVideo captures the right view)
        try:
            root_pos = env.unwrapped.root_states[0, :3].cpu().numpy()
            cam_eye = root_pos + np.array([args_cli.camera_distance, args_cli.camera_distance, args_cli.camera_height])
            sim.set_camera_view(cam_eye, root_pos)
        except Exception:
            pass

        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if timestep >= steps_to_record:
            print(f"[INFO]: Recording complete ({timestep} steps).")
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
