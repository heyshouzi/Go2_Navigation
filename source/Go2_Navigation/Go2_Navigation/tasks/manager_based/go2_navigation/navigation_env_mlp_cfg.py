"""
Navigation environment configuration with MLP-based obstacle encoding.

This configuration uses a learnable MLP encoder to process raw lidar data,
instead of hand-crafted sector features.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    HfDiscreteObstaclesTerrainCfg,
)
from isaaclab.sensors import RayCasterCfg, patterns
from . import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

LOW_LEVEL_ENV_CFG = UnitreeGo2FlatEnvCfg()
LOW_LEVEL_ENV_CFG.observations.policy.base_lin_vel = None


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (1.57, 1.57)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = (
        mdp.PreTrainedPolicyActionCfg(
            asset_name="robot",
            policy_path=f"/home/wu/IsaacLab/logs/rsl_rl/unitree_go2_flat/2025-10-02_14-33-48/exported/policy.pt",
            # policy_path=f"/home/wu/Go2_Navigation/unitree_lab_policy.pt",git
            low_level_decimation=4,
            # low_level_actions=LOW_LEVEL_ENV_CFG.actions.JointPositionAction,  # 使用新配置中的 action 名称
            low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
            low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        )
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP with MLP obstacle encoding."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for high-level navigation policy (Actor).

        🆕 Uses 360-degree LiDAR matching real Unitree Go2 hardware!

        Observation structure:
        - pose_command (4): [x, y, z, heading] target pose in base frame
        - base_ang_vel (3): [wx, wy, wz] angular velocity
        - last_action (3): [vx, vy, vyaw] 上次采取的动作
        - obstacle_features (359): 🆕 RAW 360° lidar ranges (NOT encoded here!)
          * 359 rays, one per degree (0° = forward, 90° = left, 180° = back, 270° = right)
          * Range: 0-30m (matching Unitree L1 LiDAR specs)
          * Encoded by ActorCriticWithLidarEncoder: 359 → 36 dims (inside policy network)
          * Ensures proper gradient flow for end-to-end training

        ⚠️ Total input to actor: 369 dims (4+3+3+359)
        ⚠️ After policy's internal encoding: 4 + 3 + 3 + 36 = 46 dims

        ✅ Sim2Real: Identical to real Go2 LiDAR! No resampling needed in deployment.
        Note: base_lin_vel is removed from actor to improve robustness.
        """

        # 1. Target pose command
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "pose_command"}
        )

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # 4. 🆕 RAW 360° LiDAR data (will be encoded by policy network)
        obstacle_features = ObsTerm(
            func=mdp.obstacle_mlp_encoding,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "encoder_output_dim": 36,  # Target encoding dimension: 359 rays → 36 features
            },
        )

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic network.

        Critic has access to additional information (base_lin_vel) for better value estimation.

        Observation structure:
        - pose_command (4): [x, y, z, heading] target pose in base frame
        - base_lin_vel (3): [vx, vy, vz] linear velocity (only for critic)
        - projected_gravity (3): [gx, gy, gz] projected gravity
        - obstacle_features (359): RAW 360° lidar ranges
        ⚠️ Total input to critic: 372 dims (4+3+3+3+359)
        ⚠️ After policy's internal encoding: 4 + 3 + 3 + 3 + 36 = 49 dims
        """

        # 1. Target pose command
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "pose_command"}
        )

        # 2. Current velocity state (including base_lin_vel for critic)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # 4. 🆕 RAW 360° LiDAR data
        obstacle_features = ObsTerm(
            func=mdp.obstacle_mlp_encoding,
            params={
                "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
                "encoder_output_dim": 36,
            },
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Collision penalty (terminal - 严重碰撞导致终止)
    collision_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-2.0,
        params={"term_keys": ["base_contact"]},
    )

    # 🆕 运动过程中的碰撞惩罚（持续监测）- 课程学习从-1开始
    contact_force_penalty = RewTerm(
        func=mdp.contact_force_penalty,
        weight=-20.0,  # 🎓 课程学习：从轻惩罚开始，逐步增加到-5.0
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "base",
                    "Head_upper",
                    "Head_lower",  # 机身+头部
                    "FL_thigh",
                    "FR_thigh",
                    "RL_thigh",
                    "RR_thigh",  # 四条大腿
                ],
            ),
            "threshold": 1.0,  # 1N 接触力阈值（低于终止条件的 5N）
        },
    )

    # 🆕 障碍物接近惩罚（预防性）
    obstacle_proximity_penalty = RewTerm(
        func=mdp.obstacle_proximity_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_scanner"),
            "danger_distance": 0.2,
            "warning_distance": 0.5,
        },
    )

    # Timeout penalty
    timeout_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1.0,
        params={"term_keys": ["time_out"]},
    )

    # Position tracking (coarse-grained)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )

    # Position tracking (fine-grained)
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.5,
        params={"std": 0.5, "command_name": "pose_command"},
    )

    # Velocity smoothness
    velocity_smoothness = RewTerm(
        func=mdp.velocity_smoothness_penalty,
        weight=-0.5,
    )

    # Goal reached bonus
    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus_time_aware,
        weight=10.0,
        params={
            "command_name": "pose_command",
            "distance_threshold": 0.3,
            "base_reward": 30.0,
            "time_bonus_weight": 20.0,
        },
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-7.0, 7.0),
            pos_y=(3.0, 4.0),  # 🎓 课程学习：初始从第一档开始（3-4m）
            heading=(1.57, 1.57),
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 5.0,
        },
    )

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={
            "command_name": "pose_command",
            "distance_threshold": 0.3,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # 🎓 渐进式距离课程学习：从近到远（基于RL迭代数）
    adaptive_distance = CurrTerm(
        func=mdp.modify_command_range_curriculum,
        params={
            "command_name": "pose_command",
            "attribute": "pos_y",
            "curriculum_levels": [
                (3.0, 4.0),  # 第一档：近距离（3-4m）- 最简单
                (5.0, 6.0),  # 第二档：中距离（5-6m）- 中等难度
                (7.0, 8.0),  # 第三档：远距离（7-8m）- 困难
                (9.0, 10.0),  # 第四档：超远距离（9-10m）- 最高难度
            ],
            "success_thresholds": [
                0.60,
                0.70,
                0.80,
            ],  # 升级阈值：60%→档2，70%→档3，80%→档4
            "eval_interval": 10,  # 每10个RL iteration评估一次
            "min_episodes_per_eval": 200,  # 至少收集200个episode再评估
            "warmup_iterations": 20,  # 前20个RL iteration不评估
        },
    )

    adaptive_speed = CurrTerm(
        func=mdp.adaptive_speed_requirement,
        params={
            "success_threshold": 0.3,
            "low_speed_weight": 0.1,
            "high_speed_weight": 1.5,
            "eval_interval": 50,
        },
    )
    # # 🎓 碰撞惩罚权重课程学习：前50个RL iteration保持-1，然后逐步增加到-5
    # adaptive_collision_penalty = CurrTerm(
    #     func=mdp.adaptive_collision_penalty_curriculum,
    #     params={
    #         "term_name": "contact_force_penalty",
    #         "weight_levels": [-1.0, -3.0, -5.0],              # 3个等级：从轻到重
    #         "success_thresholds": [0.6, 0.8],                 # 2个升级阈值
    #         "eval_interval": 20,                               # 每20个RL iteration评估
    #         "min_episodes_per_eval": 200,                      # 至少200个episode
    #         "warmup_iterations": 50,                           # 前50个RL iteration保持-1权重
    #     },
    # )


@configclass
class NavigationEnvMLPCfg(ManagerBasedRLEnvCfg):
    """
    Navigation environment with MLP-based obstacle encoding.
    🆕 New feature: Learnable obstacle representation
    - MLP encoder: 600 raw lidar points → 17 learned features
    - Total observation: 4 + 3 + 3 + 17 = 27 dims
    - Encoder is trained end-to-end with the policy
    """

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        # decimation决定了环境以多少仿真步（frame）为周期进行一次外部RLCycle（即RL步长），
        # 比如decimation为10时，每10个物理仿真步才采集一次RL观测并做一次动作决策、奖励计算等。
        # 这样可以减小外层RL决策的频率，提高仿真效率，模拟真实机器人控制周期远慢于仿真步；
        # 这里乘以10表示每10个低层决策周期才进行一次高层RL环境步。
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 2
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation
                * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # RayCaster sensor for obstacle detection
        self.scene.obstacle_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.1)
            ),  # LiDAR安装位置（机身顶部）
            ray_alignment="yaw",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,  # 单线LiDAR
                vertical_fov_range=(0.0, 0.0),  # 单线：垂直FOV为0
                horizontal_fov_range=(0.0, 360.0),  # 360度水平扫描
                horizontal_res=1.0,  # 每度1条射线
                # 注意：360度扫描会排除最后一个点（360°与0°重复）
                # 实际生成 359 条射线：[0°, 1°, 2°, ..., 358°]
            ),
            max_distance=8.0,
            drift_range=(-0.0, 0.0),
            debug_vis=False,  # 训练时关闭可视化以提高速度
            mesh_prim_paths=["/World/ground"],
        )

        self.scene.num_envs = 1024
        self.scene.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        self.scene.terrain = TerrainImporterCfg(
            num_envs=self.scene.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=42,
                size=(20.0, 20.0),
                border_width=20.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        size=(20.0, 20.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=70,
                        obstacle_height_mode="fixed",
                        obstacle_width_range=(0.4, 0.8),
                        obstacle_height_range=(0.3, 1.0),
                        platform_width=2.0,
                    ),
                },
            ),
        )


from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils


class NavigationEnvMLPCfg_PLAY(NavigationEnvMLPCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.scene.obstacle_scanner.debug_vis = True
        self.episode_length_s = 15.0
        # 🎮 Play模式：测试全距离范围（3-10m）
        self.commands.pose_command.resampling_time_range = (15.0, 15.0)
        self.commands.pose_command.ranges.pos_x = (-2.0, -2.0)
        self.commands.pose_command.ranges.pos_y = (7.0, 7.0)
        # self.events.reset_base.params["pose_range"]["y"] = (-6.0, -6.0)
        # self.events.reset_base.params["pose_range"]["x"] = (-2.0, 2.0)

        # ------------------------------------------------------
        # 💡 添加光源 (IsaacLab 2.3)
        # ------------------------------------------------------
        self.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
        )
