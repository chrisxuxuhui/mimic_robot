"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims and record video.

.. code-block:: bash

    # Usage Examples:
    # For HI robot with local file and video recording:
    python scripts/replay_npz_video.py --robot hi --motion_file source/motion/hightorque/hi/npz/hi_dance1_subject2.npz --output_video hi_motion.mp4 --headless

    # For PI Plus robot with local file:
    python scripts/replay_npz_video.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/pi_plus_dance1_subject2.npz --output_video pi_motion.mp4 --headless

    # For PM01 robot with custom resolution and duration:
    python scripts/replay_npz_video.py --robot pm01 --motion_file source/motion/hightorque/pm01/npz/fight1_subject.npz --output_video pm01_fight.mp4 --fps 30 --resolution 1920x1080 --duration 10 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions and record video.")
parser.add_argument("--robot", type=str, choices=["hi", "pi_plus", "pm01"], required=True,
                   help="Robot type: hi (Hi), pi_plus (PI Plus), pm01 (PM01)")
parser.add_argument("--registry_name", type=str, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, help="Local motion NPZ file path")
parser.add_argument("--output_video", type=str, default=None, help="Output video file path (e.g., output.mp4)")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for video recording")
parser.add_argument("--duration", type=float, default=None, help="Duration to record in seconds (default: full motion)")
parser.add_argument("--resolution", type=str, default="1280x720", help="Video resolution WxH (e.g., 1280x720)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Ensure cameras are enabled in headless mode for video recording
if args_cli.headless and hasattr(args_cli, 'enable_cameras') and not args_cli.enable_cameras:
    args_cli.enable_cameras = True
    print("[INFO] Headless mode detected, enabling cameras for video recording.")

# Set physics engine to PhysX to avoid Warp CUDA errors
if not hasattr(args_cli, 'physics_engine'):
    args_cli.physics_engine = 'physx'
    print("[INFO] Setting physics engine to PhysX (Warp requires newer CUDA driver)")

# Set renderer to PathTracing for better headless compatibility
if not hasattr(args_cli, 'renderer'):
    args_cli.renderer = 'PathTracing'

# Ensure cameras are enabled for headless video recording
if args_cli.headless:
    if not hasattr(args_cli, 'enable_cameras'):
        args_cli.enable_cameras = True
    elif not args_cli.enable_cameras:
        args_cli.enable_cameras = True
        print("[INFO] Headless mode detected, enabling cameras for video recording.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# For video recording
import os
import numpy as np

# Import Replicator for headless video recording (Isaac Sim 5.0+)
try:
    import omni.replicator.core as rep
    REPLICATOR_AVAILABLE = True
except ImportError:
    REPLICATOR_AVAILABLE = False
    print("[WARNING] omni.replicator.core not available. Video recording in headless mode may not work.")

##
# Pre-defined configs
##
from whole_body_tracking.robots.hi import HI_CFG
from whole_body_tracking.robots.pi_plus import PI_PLUS_CFG
from whole_body_tracking.robots.pm01 import PM01_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

# Robot configurations
ROBOT_CONFIGS = {
    "hi": {
        "cfg": HI_CFG,
        "name": "Hi"
    },
    "pi_plus": {
        "cfg": PI_PLUS_CFG,
        "name": "PI Plus"
    },
    "pm01": {
        "cfg": PM01_CFG,
        "name": "PM01"
    }
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation (will be set dynamically based on robot type)
    robot: ArticulationCfg = None


class VideoRecorder:
    """Video recorder for Isaac Sim, supporting both headless (Replicator) and non-headless modes."""

    def __init__(self, sim, output_video, fps=30, duration=None, headless=False, resolution="1280x720"):
        self.sim = sim
        self.output_video = output_video
        self.fps = fps
        self.duration = duration
        self.headless = headless
        self.resolution = resolution

        # Parse resolution
        self.width, self.height = map(int, resolution.split('x'))

        # Recording state
        self.recording_enabled = False
        self.recording_completed = False
        self.frame_count = 0
        self.max_frames = int(duration * fps) if duration else None
        self.frames_to_record = []

        # Replicator setup for headless mode
        self.rep_writer = None
        self.rep_trigger = None

        # Initialize recording
        self._initialize_recording()

    def _initialize_recording(self):
        """Initialize video recording based on mode."""
        if not self.output_video:
            return

        self.recording_enabled = True
        print(f"[INFO]: Video recording enabled. Output: {self.output_video}, FPS: {self.fps}, Resolution: {self.resolution}")
        if self.max_frames:
            print(f"[INFO]: Will record up to {self.max_frames} frames ({self.duration}s)")

        # Headless mode: use Replicator if available
        if self.headless and REPLICATOR_AVAILABLE:
            print("[INFO]: Headless mode detected, setting up Replicator for video recording...")
            try:
                import omni.replicator.core as rep

                # Create or get camera
                camera_path = "/OmniverseKit_Persp"

                # Check if camera exists, if not create one
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                camera_prim = stage.GetPrimAtPath(camera_path)
                if not camera_prim.IsValid():
                    # Create camera
                    from pxr import UsdGeom
                    camera_prim = UsdGeom.Camera.Define(stage, camera_path)

                # Create render product with specified resolution
                camera = rep.get.prim_at_path(camera_path)
                render_product = rep.create.render_product(camera, (self.width, self.height))

                # Try to get VideoWriter if available, otherwise use BasicWriter
                try:
                    self.rep_writer = rep.WriterRegistry.get("VideoWriter")
                    writer_params = {
                        "output_dir": os.path.dirname(self.output_video) or ".",
                        "fps": self.fps,
                        "codec": "h264"
                    }
                    print(f"[INFO]: Using Replicator VideoWriter with params: {writer_params}")
                except Exception as e:
                    print(f"[INFO]: VideoWriter not available, using BasicWriter: {e}")
                    self.rep_writer = rep.WriterRegistry.get("BasicWriter")
                    output_dir = os.path.dirname(self.output_video) or "."
                    # Ensure absolute path
                    if not os.path.isabs(output_dir):
                        output_dir = os.path.abspath(output_dir)
                    # BasicWriter parameters
                    writer_params = {
                        "output_dir": output_dir,
                        "rgb": True
                    }
                    # Try to add file_prefix if supported
                    try:
                        writer_params["file_prefix"] = os.path.splitext(os.path.basename(self.output_video))[0]
                    except Exception:
                        pass  # file_prefix may not be supported

                self.rep_writer.initialize(**writer_params)
                self.rep_writer.attach([render_product])

                # Try to create trigger using Isaac Sim 5.0+ API
                try:
                    self.rep_trigger = rep.orchestrator.step_async()
                    print("[INFO]: Replicator video recording setup completed with orchestrator trigger.")
                except AttributeError:
                    # No trigger API available, rely on BasicWriter automatic capture
                    self.rep_trigger = None
                    print("[INFO]: Replicator video recording setup completed. No trigger API, relying on automatic capture.")

            except Exception as e:
                print(f"[WARNING]: Failed to setup Replicator recording: {e}")
                print("[INFO]: Falling back to manual frame capture.")
                self.rep_writer = None
                self.rep_trigger = None
        else:
            print("[INFO]: Using manual frame capture for video recording.")

    def capture_frame(self):
        """Capture current frame if recording is enabled."""
        if not self.recording_enabled:
            return

        # Check duration limit
        if self.max_frames and self.frame_count >= self.max_frames:
            print(f"[INFO]: Reached duration limit ({self.duration}s), stopping recording.")
            self.recording_enabled = False
            self.recording_completed = True
            return

        # Headless mode with Replicator
        if self.headless and self.rep_writer is not None:
            # Replicator BasicWriter will capture frames automatically
            self.frame_count += 1
            return

        # Manual frame capture for non-headless or fallback
        try:
            image = None

            # Method 1: sim.capture() - common in Isaac Sim
            if hasattr(self.sim, 'capture'):
                try:
                    image = self.sim.capture(format='rgb')
                except Exception:
                    pass

            # Method 2: sim.get_screenshot()
            if image is None and hasattr(self.sim, 'get_screenshot'):
                try:
                    image = self.sim.get_screenshot()
                except Exception:
                    pass

            # Method 3: sim.get_render_image()
            if image is None and hasattr(self.sim, 'get_render_image'):
                try:
                    image = self.sim.get_render_image()
                except Exception:
                    pass

            # Method 4: omni.isaac.synthetic_utils for headless mode
            if image is None and self.headless:
                try:
                    import omni.isaac.synthetic_utils as su
                    camera_path = "/OmniverseKit_Persp"
                    image = su.get_synthetic_data(camera_path, "rgb")
                except Exception:
                    pass

            if image is not None:
                # Convert to uint8 if needed
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)

                self.frames_to_record.append(image)
                self.frame_count += 1
            else:
                if self.frame_count == 0:
                    print("[WARNING]: Could not capture image. Video frames may be empty.")

        except Exception as e:
            print(f"[WARNING]: Failed to capture frame: {e}")

    def save_video(self):
        """Save recorded video to file."""
        # For Replicator in headless mode, it should have saved automatically
        if self.headless and self.rep_writer is not None:
            print("[INFO]: Replicator video should have been saved automatically.")
            return

        # Manual save for non-Replicator mode
        if not self.frames_to_record:
            print("[WARNING]: No frames captured. Video file not created.")
            return

        print(f"[INFO]: Saving {len(self.frames_to_record)} frames to {self.output_video}...")
        try:
            # Try using imageio
            import imageio
            writer = imageio.get_writer(self.output_video, fps=self.fps, macro_block_size=1)
            for frame in self.frames_to_record:
                writer.append_data(frame)
            writer.close()
            print(f"[INFO]: Video saved successfully to {self.output_video}")
        except ImportError:
            print("[WARNING]: imageio not installed. Install with: pip install imageio")
            # Try OpenCV as fallback
            try:
                import cv2
                if self.frames_to_record:
                    h, w = self.frames_to_record[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(self.output_video, fourcc, self.fps, (w, h))
                    for frame in self.frames_to_record:
                        # Convert RGB to BGR for OpenCV
                        if frame.shape[2] == 3:  # RGB
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            out.write(frame_bgr)
                        else:
                            out.write(frame)
                    out.release()
                    print(f"[INFO]: Video saved using OpenCV to {self.output_video}")
            except ImportError:
                print("[WARNING]: Neither imageio nor OpenCV available. Cannot save video.")
                print("[INFO]: Frames captured but not saved. Install: pip install imageio opencv-python")
        except Exception as e:
            print(f"[ERROR]: Failed to save video: {e}")

    def is_recording(self):
        """Check if recording is still enabled."""
        return self.recording_enabled


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]

    # Debug info
    print(f"[DEBUG] Robot articulation loaded: {robot}")
    print(f"[DEBUG] Number of joints: {robot.num_joints}")
    print(f"[DEBUG] Joint names: {robot.joint_names}")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    print(f"[DEBUG] Physics timestep: {sim_dt}")

    # Initialize video recorder
    video_recorder = VideoRecorder(
        sim=sim,
        output_video=args_cli.output_video,
        fps=args_cli.fps,
        duration=args_cli.duration,
        headless=args_cli.headless,
        resolution=args_cli.resolution
    )

    # Support both local file and WandB registry
    if args_cli.motion_file:
        motion_file = args_cli.motion_file
        print(f"[INFO]: Using local motion file: {motion_file}")
    else:
        if not args_cli.registry_name:
            raise ValueError("Either --motion_file or --registry_name must be provided")
        
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO]: Using WandB motion file: {registry_name}")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Debug motion info
    print(f"[DEBUG] Motion file: {motion_file}")
    print(f"[DEBUG] Motion frames: {motion.time_step_total}")
    print(f"[DEBUG] Motion joint positions shape: {motion.joint_pos.shape}")
    print(f"[DEBUG] Motion body positions shape: {motion.body_pos_w.shape}")
    print(f"[DEBUG] First frame waist joint (j12_waist_yaw_joint) position: {motion.joint_pos[0, 12].item() if motion.joint_pos.shape[1] > 12 else 'N/A'}")

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()

        # Set camera to follow robot before rendering
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        sim.render()  # We don't want physic (sim.step())

        # Capture frame for video recording
        video_recorder.capture_frame()

        # Stop simulation if recording completed (duration limit reached)
        if video_recorder.recording_completed:
            print("[INFO]: Recording completed, stopping simulation.")
            break

        scene.update(sim_dt)

    # Save recorded video
    video_recorder.save_video()


def main():
    # Get robot configuration
    robot_config = ROBOT_CONFIGS[args_cli.robot]
    print(f"[INFO]: Using robot configuration: {args_cli.robot} ({robot_config['name']})")
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    # Design scene with robot-specific configuration
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_config["cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print(f"[INFO]: Setup complete for {robot_config['name']} robot...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
