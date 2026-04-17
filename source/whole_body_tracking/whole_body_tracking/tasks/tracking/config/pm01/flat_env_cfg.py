from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.pm01 import PM01_CFG, PM01_ACTION_SCALE

from whole_body_tracking.tasks.tracking.config.pi_plus.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm


@configclass
class PM01FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = PM01_ACTION_SCALE

        # 指定参与运动跟踪的关节（24个关节全部参与）
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

        # motion command也需要知道参与跟踪的关节（与动作空间一致）
        self.commands.motion.joint_names = self.actions.joint_pos.joint_names

        self.commands.motion.anchor_body_name = "link_base"  # URDF根链接名

        # pm01跟踪的身体部位（基于URDF链接名）
        self.commands.motion.body_names = [
            "link_base",           # 基础链接
            "link_hip_roll_l",     # 左髋滚转
            "link_knee_pitch_l",   # 左小腿（膝盖）
            "link_ankle_roll_l",   # 左踝滚转
            "link_hip_roll_r",     # 右髋滚转
            "link_knee_pitch_r",   # 右小腿（膝盖）
            "link_ankle_roll_r",   # 右踝滚转

            "link_shoulder_roll_l",  # 左肩滚转
            "link_elbow_pitch_l",    # 左肘（俯仰）
            "link_elbow_yaw_l",      # 左肘末端（偏航，作为手腕）
            "link_shoulder_roll_r",  # 右肩滚转
            "link_elbow_pitch_r",    # 右肘（俯仰）
            "link_elbow_yaw_r",      # 右肘末端（偏航，作为手腕）
        ]

        # 相机设置：自由视角，不跟随机器人
        self.viewer.eye = (3.0, 3.0, 2.0)  # 相机位置
        self.viewer.lookat = (0.0, 0.0, 1.0)  # 看向位置
        self.viewer.origin_type = "world"  # 世界坐标系，不跟随机器人
        self.viewer.asset_name = None  # 不绑定到特定资产

        # 关闭调试可视化显示
        # self.commands.motion.debug_vis = False  # 关闭motion命令的调试可视化
        self.scene.contact_forces.debug_vis = False  # 关闭接触力可视化

        # 保持训练时的默认设置（自适应采样等）
        self.rewards.motion_body_pos.weight = 1.5  # 从1.0增加到1.5
        self.rewards.motion_body_pos.params["std"] = 0.15  # 从1.0增加到1.5

        # 终止条件：脚踝和手腕（肘末端）位置误差
        self.terminations.ee_body_pos.params["body_names"] = [
            "link_ankle_roll_l",  # 左脚踝
            "link_ankle_roll_r",
            "link_elbow_yaw_l",   # 左肘末端（偏航，作为手腕）
            "link_elbow_yaw_r",   # 右肘末端
        ]

        # 不希望接触的身体部位（除了脚踝和手腕/肘末端）
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!link_ankle_roll_l$)(?!link_ankle_roll_r$)(?!link_elbow_yaw_l$)(?!link_elbow_yaw_r$).+$"
            ],
        )

        # 修复base_com事件配置，使用link_base而不是torso_link
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="link_base"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )


@configclass
class PM01FlatWoEnvCfg(PM01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 3),
                "dynamic_friction_range": (0.5, 3),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        )