import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# pm01电机常数（来自MJCF armature值）
ARMATURE_HIGH = 0.045325  # 高扭矩关节（髋/膝）：±164 Nm
ARMATURE_LOW = 0.039175   # 低扭矩关节（踝/臂/腰/头）：±61 Nm

# 使用pi_plus的刚度和阻尼值（待调整）
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIGH = 80      # 高扭矩关节刚度（同STIFFNESS_5047）
STIFFNESS_LOW = 30       # 低扭矩关节刚度（同STIFFNESS_4438）

DAMPING_HIGH = 1.1       # 高扭矩关节阻尼（同DAMPING_5047）
DAMPING_LOW = 0.6        # 低扭矩关节阻尼（同DAMPING_4438）

# 关节命名映射：MJCF J00_HIP_PITCH_L → URDF j00_hip_pitch_l_joint
# 正则表达式模式
LEG_JOINTS_EXPR = [
    "j00_hip_pitch_l_joint",
    "j01_hip_roll_l_joint",
    "j02_hip_yaw_l_joint",
    "j03_knee_pitch_l_joint",
    "j06_hip_pitch_r_joint",
    "j07_hip_roll_r_joint",
    "j08_hip_yaw_r_joint",
    "j09_knee_pitch_r_joint",
]

FOOT_JOINTS_EXPR = [
    "j04_ankle_pitch_l_joint",
    "j05_ankle_roll_l_joint",
    "j10_ankle_pitch_r_joint",
    "j11_ankle_roll_r_joint",
]

WAIST_YAW_JOINT_EXPR = ["j12_waist_yaw_joint"]

ARM_JOINTS_EXPR = [
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
]

HEAD_YAW_JOINT_EXPR = ["j23_head_yaw_joint"]

# 扭矩限制（Nm）
EFFORT_HIGH = 164.0  # 高扭矩关节
EFFORT_LOW = 61.0    # 低扭矩关节

# 速度限制（rad/s） - 暂用pi_plus值，待调整
VELOCITY_HIGH = 8.0
VELOCITY_LOW = 17.0


PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/hightorque/pm01/urdf/pm01_24dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),  # 初始高度，待调整
        joint_pos={
            # 左腿
            "j00_hip_pitch_l_joint": -0.25,
            "j01_hip_roll_l_joint": 0.0,
            "j02_hip_yaw_l_joint": 0.0,
            "j03_knee_pitch_l_joint": 0.65,
            "j04_ankle_pitch_l_joint": -0.4,
            "j05_ankle_roll_l_joint": 0.0,
            # 右腿
            "j06_hip_pitch_r_joint": -0.25,
            "j07_hip_roll_r_joint": 0.0,
            "j08_hip_yaw_r_joint": 0.0,
            "j09_knee_pitch_r_joint": 0.65,
            "j10_ankle_pitch_r_joint": -0.4,
            "j11_ankle_roll_r_joint": 0.0,
            # 腰部
            "j12_waist_yaw_joint": 0.0,
            # 左臂
            "j13_shoulder_pitch_l_joint": 0.0,
            "j14_shoulder_roll_l_joint": 0.0,
            "j15_shoulder_yaw_l_joint": 0.0,
            "j16_elbow_pitch_l_joint": 0.0,
            "j17_elbow_yaw_l_joint": 0.0,
            # 右臂
            "j18_shoulder_pitch_r_joint": 0.0,
            "j19_shoulder_roll_r_joint": 0.0,
            "j20_shoulder_yaw_r_joint": 0.0,
            "j21_elbow_pitch_r_joint": 0.0,
            "j22_elbow_yaw_r_joint": 0.0,
            # 头部
            "j23_head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINTS_EXPR,
            effort_limit_sim=EFFORT_HIGH,
            velocity_limit_sim=VELOCITY_HIGH,
            stiffness=STIFFNESS_HIGH,
            damping=DAMPING_HIGH,
            armature=ARMATURE_HIGH,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=FOOT_JOINTS_EXPR,
            effort_limit_sim=EFFORT_LOW,
            velocity_limit_sim=VELOCITY_LOW,
            stiffness=STIFFNESS_HIGH,  # 脚踝使用高刚度（同腿）
            damping=DAMPING_HIGH,
            armature=ARMATURE_LOW,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=WAIST_YAW_JOINT_EXPR,
            effort_limit_sim=EFFORT_LOW,
            velocity_limit_sim=VELOCITY_LOW,
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW,
            armature=ARMATURE_LOW,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS_EXPR,
            effort_limit_sim=EFFORT_LOW,
            velocity_limit_sim=VELOCITY_LOW,
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW,
            armature=ARMATURE_LOW,
        ),
        "head_yaw": ImplicitActuatorCfg(
            joint_names_expr=HEAD_YAW_JOINT_EXPR,
            effort_limit_sim=EFFORT_LOW,
            velocity_limit_sim=VELOCITY_LOW,
            stiffness=STIFFNESS_LOW,
            damping=DAMPING_LOW,
            armature=ARMATURE_LOW,
        ),
    },
)

# 计算动作缩放（遵循pi_plus模式）
PM01_ACTION_SCALE = {}
for a in PM01_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PM01_ACTION_SCALE[n] = 0.25 * e[n] / s[n]