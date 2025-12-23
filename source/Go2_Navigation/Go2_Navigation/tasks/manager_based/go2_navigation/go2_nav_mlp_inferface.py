#!/usr/bin/env python3
"""
Real-world deployment script for MLP-based navigation policy on Unitree Go2.

🆕 Simplified Version Features:
- Subscribes to /lidar_projected topic (std_msgs::Float32MultiArray, 359 dims)
- Policy network contains integrated MLP encoder (359 → 36 dims)
- Observation: pose_command(4) + projected_gravity(3) + lidar(359) = 366 dims
- Matches training configuration exactly (navigation_env_mlp_cfg.py PolicyCfg)
- No point cloud processing logic, directly uses projected lidar data

This script:
1. Loads the trained high-level navigation policy with MLP encoder
2. Reads sensor data from Unitree SDK2 (IMU) + ROS /lidar_projected topic
3. Constructs 366-dim observation vector (matches training config)
4. Infers velocity commands (vx, vy, vyaw)
5. Sends commands to low-level controller

Observation structure (matches training):
- pose_command (4): [x, y, z, heading] target pose in base frame
- projected_gravity (3): [gx, gy, gz] gravity projection in base frame
- obstacle_features (359): Raw 360° LiDAR ranges from /lidar_projected topic

Requirements:
- ROS must be installed and running
- /lidar_projected topic must be publishing (std_msgs::Float32MultiArray, 359 dims)

Author: AI Assistant
Date: 2025-01-XX
"""

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Odometry
import tf.transformations as tft
import tf2_ros



import torch
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse

# ROS imports for /lidar_projected topic - REQUIRED
try:
    import rospy
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
    print("✅ ROS (rospy) imported successfully")
except Exception as _ros_import_error:
    ROS_AVAILABLE = False
    print(f"❌ Error: ROS not available. /lidar_projected subscription requires ROS.")
    print(f"   Import error: {_ros_import_error}")

# Unitree SDK2 imports
try:
    # Add the unitree_sdk2_python directory to Python path
    import sys
    import os
    sdk_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'unitree_sdk2_python')
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LidarState_
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    SDK_AVAILABLE = True
    print("✅ Unitree SDK2 imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: unitree_sdk2_python not found. Running in simulation mode.")
    print(f"   Import error: {e}")
    SDK_AVAILABLE = False


@dataclass
class NavigationGoal:
    """Navigation goal in world frame."""
    x: float  # meters
    y: float  # meters
    yaw: float  # radians


@dataclass
class RobotState:
    """Robot state from sensors."""
    # Position (odometry)
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    
    # Orientation (quaternion)
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0
    
    # Linear velocity (base frame)
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    
    # Angular velocity (base frame)
    omega_x: float = 0.0
    omega_y: float = 0.0
    omega_z: float = 0.0
    
    # Lidar data (359 dims from /lidar_projected topic)
    lidar_projected: Optional[np.ndarray] = None  # 359-dimensional array


class UnitreeGo2Interface:
    """Interface to Unitree Go2 robot via SDK2."""
    
    def __init__(self, use_lidar: bool = True, lidar_topic: str = "/lidar_projected"):
        """
        Initialize Unitree Go2 interface.
        
        Args:
            use_lidar: 是否使用LiDAR（必须为True时才能使用导航）
            lidar_topic: LiDAR投影数据ROS话题名称（默认: /lidar_projected）
        """
        self.use_lidar = use_lidar
        self.lidar_topic = lidar_topic
        self.state = RobotState()
        self.running = False
        self.sport_state_msg = None
        self.sport_client = None
        self.lidar_data_lock = threading.Lock()  # Lock for thread-safe lidar data access
        
        # Enhanced position tracking
        self.odometry_position = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        self.imu_position = np.array([0.0, 0.0, 0.0])
        self.fused_position = np.array([0.0, 0.0, 0.0])
        self.last_imu_time = time.time()
        self.initial_quat = None
        
        # ROS定位系统相关
        self.localization_position = np.array([0.0, 0.0, 0.0])  # 从AMCL/odometry获取的位置 [x, y, yaw]
        self.localization_lock = threading.Lock()  # 定位数据锁
        self.use_localization = True  # 是否使用ROS定位系统
        self.map_frame = "map"  # 地图坐标系名称
        self.base_frame = "base_link"  # 机器人本体坐标系名称
        
        # 初始化ROS节点（如果还没有初始化）
        if ROS_AVAILABLE:
            if not rospy.core.is_initialized():
                rospy.init_node("go2_nav_mlp_simplified", anonymous=True, disable_signals=True)
            
            # 初始化tf2监听器
            try:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
                print("✅ TF2 listener initialized")
            except Exception as e:
                print(f"⚠️  Warning: TF2 listener initialization failed: {e}")
                self.tf_buffer = None
                self.tf_listener = None
            
            # 订阅RViz 2D Nav Goal
            self.current_goal = None
            self.goal_lock = threading.Lock()
            self.goal_sub = rospy.Subscriber(
                "/move_base_simple/goal",
                PoseStamped,
                self._rviz_goal_callback,
                queue_size=1
            )
            print("✅ Subscribed to RViz 2D Nav Goal (/move_base_simple/goal)")
            
            # 订阅AMCL定位结果（优先使用）
            try:
                self.amcl_sub = rospy.Subscriber(
                    "/amcl_pose",
                    PoseWithCovarianceStamped,
                    self._amcl_pose_callback,
                    queue_size=1
                )
                print("✅ Subscribed to AMCL pose (/amcl_pose)")
            except Exception as e:
                print(f"⚠️  Warning: AMCL subscription failed: {e}")
                self.amcl_sub = None
            
            # 订阅odometry作为备选（如果AMCL不可用）
            try:
                self.odom_sub = rospy.Subscriber(
                    "/odom",
                    Odometry,
                    self._odom_callback,
                    queue_size=1
                )
                print("✅ Subscribed to odometry (/odom)")
            except Exception as e:
                print(f"⚠️  Warning: Odometry subscription failed: {e}")
                self.odom_sub = None

        
        # Initialize sensor data subscribers
        print("🔧 Setting up sensor data subscribers...")
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        
        # Low state subscriber
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        ret = self.low_state_sub.Init(self._low_state_handler, 10)
        if ret != 0:
            print(f"⚠️  LowState subscriber failed with code: {ret}")
        else:
            print("✅ LowState subscriber initialized")
        
        # Sport mode state subscriber
        self.sport_state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sport_state_sub.Init(self._sport_state_handler, 10)
        print("✅ SportModeState subscriber initialized")
        
        # Lidar subscriber - /lidar_projected topic
        if self.use_lidar:
            if not ROS_AVAILABLE:
                raise RuntimeError("错误: use_lidar=True，但 ROS 不可用（未安装 rospy）。请安装ROS和rospy以使用/lidar_projected话题。")
            
            try:
                if not rospy.core.is_initialized():
                    rospy.init_node("go2_nav_mlp_simplified", anonymous=True, disable_signals=True)
                self.ros_lidar_sub = rospy.Subscriber(self.lidar_topic, Float32MultiArray, self._ros_lidar_projected_handler, queue_size=1)
                print(f"✅ 已订阅 /lidar_projected 话题: {self.lidar_topic}")
                # Start a background spin thread so callbacks are serviced
                self._ros_spin_thread = threading.Thread(target=rospy.spin, daemon=True)
                self._ros_spin_thread.start()
            except Exception as e:
                raise RuntimeError(f"错误: 订阅 /lidar_projected 话题失败: {e}。请检查ROS节点是否运行，话题 {self.lidar_topic} 是否存在。")
        
        print("✅ SportClient and sensors initialized successfully")

    def _rviz_goal_callback(self, msg: PoseStamped):
        """Callback for RViz 2D Nav Goal."""
        try:
            q = msg.pose.orientation
            _, _, yaw = tft.euler_from_quaternion([q.w, q.x, q.y, q.z])
            
            goal = NavigationGoal(
                x=msg.pose.position.x,
                y=msg.pose.position.y,
                yaw=yaw
            )
            
            with self.goal_lock:
                self.current_goal = goal
            
            print("\n🎯 New RViz Goal Received")
            print(f"   Map frame: x={goal.x:.2f}, y={goal.y:.2f}, yaw={np.degrees(yaw):.1f}°")
        except Exception as e:
            print(f"⚠️  Error processing RViz goal: {e}")
    
    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        """Callback for AMCL pose (激光定位)."""
        try:
            pose = msg.pose.pose
            q = pose.orientation
            _, _, yaw = tft.euler_from_quaternion([q.w, q.x, q.y, q.z])
            
            with self.localization_lock:
                self.localization_position[0] = pose.position.x
                self.localization_position[1] = pose.position.y
                self.localization_position[2] = yaw
            
        except Exception as e:
            print(f"⚠️  Error processing AMCL pose: {e}")
    
    def _odom_callback(self, msg: Odometry):
        """Callback for odometry (作为AMCL的备选)."""
        try:
            # 只有在AMCL不可用时才使用odometry
            if hasattr(self, 'amcl_sub') and self.amcl_sub is not None:
                return  # AMCL可用，不使用odometry
            
            pose = msg.pose.pose
            q = pose.orientation
            _, _, yaw = tft.euler_from_quaternion([q.w, q.x, q.y, q.z])
            
            with self.localization_lock:
                self.localization_position[0] = pose.position.x
                self.localization_position[1] = pose.position.y
                self.localization_position[2] = yaw
            
        except Exception as e:
            print(f"⚠️  Error processing odometry: {e}")
    
    def _get_robot_pose_from_tf(self) -> Optional[np.ndarray]:
        """通过TF获取机器人在地图坐标系中的位置 [x, y, yaw]."""
        if not ROS_AVAILABLE or self.tf_buffer is None:
            return None
        
        try:
            # 尝试获取从map到base_link的变换
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rospy.Time(0),
                timeout=rospy.Duration(0.1)
            )
            
            # 提取位置
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # 提取姿态（四元数转欧拉角）
            q = transform.transform.rotation
            _, _, yaw = tft.euler_from_quaternion([q.w, q.x, q.y, q.z])
            
            return np.array([x, y, yaw])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # TF查找失败，返回None
            return None
        except Exception as e:
            print(f"⚠️  Error getting robot pose from TF: {e}")
            return None
    
    def _low_state_handler(self, msg):
        """Callback for low state messages."""
        self.low_state_msg = msg
        self._read_sensor_data_from_lowstate()
    
    def _sport_state_handler(self, msg: SportModeState_):
        """Callback for sport mode state messages."""
        self.sport_state_msg = msg
        self._read_sensor_data()
    
    def _ros_lidar_projected_handler(self, msg: Float32MultiArray):
        """Callback for ROS /lidar_projected topic (std_msgs::Float32MultiArray)."""
        try:
            if msg is None or msg.data is None:
                return
            
            # Extract 359-dimensional array from Float32MultiArray
            data = np.array(msg.data, dtype=np.float32)
            
            # Validate length
            if len(data) != 359:
                print(f"⚠️  Warning: /lidar_projected data length is {len(data)}, expected 359")
                return
            
            # Thread-safe update
            with self.lidar_data_lock:
                self.state.lidar_projected = data
                
        except Exception as e:
            print(f"⚠️  Error processing /lidar_projected data: {e}")
    
    def _read_sensor_data(self):
        """Read and process sensor data from SportModeState."""
        if not SDK_AVAILABLE or self.sport_state_msg is None:
            return
        
        try:
            sport_state = self.sport_state_msg
            
            # Update state
            self.state.pos_x = sport_state.position[0]
            self.state.pos_y = sport_state.position[1]
            self.state.pos_z = sport_state.position[2]
            
            self.state.vel_x = sport_state.velocity[0]
            self.state.vel_y = sport_state.velocity[1]
            self.state.vel_z = sport_state.velocity[2]
            
            # IMU data
            self.state.quat_w = sport_state.imu_state.quaternion[0]
            self.state.quat_x = sport_state.imu_state.quaternion[1]
            self.state.quat_y = sport_state.imu_state.quaternion[2]
            self.state.quat_z = sport_state.imu_state.quaternion[3]
            
            self.state.omega_x = sport_state.imu_state.gyroscope[0]
            self.state.omega_y = sport_state.imu_state.gyroscope[1]
            self.state.omega_z = sport_state.imu_state.gyroscope[2]
            
            # Update position estimates
            self._update_position_estimates_from_data()
            
        except Exception as e:
            print(f"⚠️  Sensor read error: {e}")
    
    def _read_sensor_data_from_lowstate(self):
        """Read sensor data from LowState."""
        if not SDK_AVAILABLE or not hasattr(self, 'low_state_msg') or self.low_state_msg is None:
            return
        
        try:
            low_state = self.low_state_msg
            
            # Update IMU data
            self.state.quat_w = low_state.imu_state.quaternion[0]
            self.state.quat_x = low_state.imu_state.quaternion[1]
            self.state.quat_y = low_state.imu_state.quaternion[2]
            self.state.quat_z = low_state.imu_state.quaternion[3]
            
            self.state.omega_x = low_state.imu_state.gyroscope[0]
            self.state.omega_y = low_state.imu_state.gyroscope[1]
            self.state.omega_z = low_state.imu_state.gyroscope[2]
            
        except Exception as e:
            print(f"⚠️  LowState sensor read error: {e}")
    
    def _update_position_estimates_from_data(self):
        """Update position estimates using current sensor data."""
        self.odometry_position[0] = self.state.pos_x
        self.odometry_position[1] = self.state.pos_y
        self.odometry_position[2] = self._get_yaw_from_quat(self.state)
        
        # Use odometry as fused position
        self.fused_position = self.odometry_position.copy()
    
    def _get_yaw_from_quat(self, state: RobotState) -> float:
        """Extract yaw angle from quaternion."""
        w, x, y, z = state.quat_w, state.quat_x, state.quat_y, state.quat_z
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_fused_position(self):
        """
        获取融合后的位置估计 [x, y, yaw]。
        优先级：ROS定位系统（AMCL/odometry/TF） > SDK odometry > 默认值
        """
        # 优先使用ROS定位系统（AMCL/odometry）
        if self.use_localization and ROS_AVAILABLE:
            # 首先尝试从TF获取（最可靠）
            tf_pose = self._get_robot_pose_from_tf()
            if tf_pose is not None:
                with self.localization_lock:
                    self.localization_position = tf_pose
                return tf_pose.copy()
            
            # 如果TF不可用，使用AMCL/odometry回调的数据
            with self.localization_lock:
                if np.any(self.localization_position != 0.0) or np.any(np.abs(self.localization_position) > 1e-6):
                    return self.localization_position.copy()
        
        # 备选：使用SDK的odometry数据
        if SDK_AVAILABLE:
            if hasattr(self, 'sport_state_msg') and self.sport_state_msg is not None:
                sport_state = self.sport_state_msg
                yaw = self._get_yaw_from_quat(self.state)
                return np.array([sport_state.position[0], sport_state.position[1], yaw])
            
            return np.array([self.state.pos_x, self.state.pos_y, self._get_yaw_from_quat(self.state)])
        
        # 默认值
        return np.array([0.0, 0.0, 0.0])
    
    def start(self):
        """Start the interface."""
        if not SDK_AVAILABLE or self.sport_client is None:
            print("⚠️  Running in simulation mode")
            return
        
        print("🔧 Setting robot to balanced stand mode...")
        self.sport_client.BalanceStand()
        print("✅ Robot in balanced stand mode")
        
        print("⏳ Waiting for SportModeState data...")
        timeout = 10.0
        start_time = time.time()
        
        while self.sport_state_msg is None:
            if time.time() - start_time > timeout:
                print("⚠️  No SportModeState data received")
                break
            time.sleep(0.1)
        
        if self.sport_state_msg is not None:
            print("✅ SportModeState data received")
            pos = self.get_fused_position()
            print(f"📍 Initial position: x={pos[0]:.2f}, y={pos[1]:.2f}, yaw={np.degrees(pos[2]):.1f}°")
        
        self.running = True
        print("🚀 Interface started")
    
    def stop(self):
        """Stop the interface."""
        if self.sport_client is not None:
            self.sport_client.StopMove()
            print("🛑 Movement stopped")
        
        self.running = False
        print("🛑 Interface stopped")
    
    def send_velocity_command(self, vx: float, vy: float, vyaw: float):
        """Send velocity command to robot."""
        if not SDK_AVAILABLE or self.sport_client is None:
            print(f"[SIM] Command: vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
            return
        
        try:
            # Clip velocities to SDK limits
            vx_clipped = np.clip(vx, -0.6, 0.6)
            vy_clipped = np.clip(vy, -0.4, 0.4)
            vyaw_clipped = np.clip(vyaw, -0.8, 0.8)
            
            self.sport_client.Move(vx_clipped, vy_clipped, vyaw_clipped)
            
            # Reduce output frequency
            if not hasattr(self, '_move_debug_counter'):
                self._move_debug_counter = 0
            self._move_debug_counter += 1
            
            if self._move_debug_counter % 10 == 0:
                print(f"✅ Move: vx={vx_clipped:.2f}, vy={vy_clipped:.2f}, vyaw={vyaw_clipped:.2f}")
                
        except Exception as e:
            print(f"❌ Command send error: {e}")
    
    def get_state(self) -> RobotState:
        """Get current robot state."""
        return self.state


class NavigationController:
    """
    High-level navigation controller with MLP-based obstacle encoder.
    
    🆕 Simplified MLP Version:
    - Observation: 366 dims (pose(4) + projected_gravity(3) + lidar(359))
    - Policy network contains integrated MLP encoder (359 → 36)
    - Directly uses /lidar_projected topic data (359 dims)
    - No point cloud processing needed
    - Matches training configuration exactly (navigation_env_mlp_cfg.py PolicyCfg)
    """
    
    def __init__(
        self,
        policy_path: str,
        device: str = "cpu",
        use_lidar: bool = True,
        lidar_topic: str = "/lidar_projected",
        max_lidar_distance: float = 8.0,  # 与训练配置一致 (navigation_env_mlp_cfg.py: max_distance=8.0)
        lidar_angle_offset_deg: int = 0,
        lidar_reverse: bool = False,
    ):
        self.device = torch.device(device)
        
        # Load policy
        print(f"📦 Loading MLP navigation policy from: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("✅ Policy loaded successfully (includes integrated MLP encoder)")
        
        # Initialize robot interface
        self.robot = UnitreeGo2Interface(
            use_lidar=use_lidar,
            lidar_topic=lidar_topic
        )
        
        # Navigation state
        self.current_goal: Optional[NavigationGoal] = None
        self.initial_position: Optional[Tuple[float, float, float]] = None
        self.last_position_update = time.time()
        self.max_lidar_distance = max_lidar_distance
        # LiDAR index semantics:
        # Training (RayCaster + LidarPatternCfg) uses 359 rays for angles [0..358] degrees.
        # In real deployment, /lidar_projected must match this indexing. If not, use the
        # optional offset/reverse to align (after you validate with a real obstacle).
        self.lidar_angle_offset_deg = int(lidar_angle_offset_deg)
        self.lidar_reverse = bool(lidar_reverse)
        
        print(f"📊 Observation structure: 366 dims (matches training config)")
        print(f"   - pose_command: 4 dims [x, y, z, heading]")
        print(f"   - projected_gravity: 3 dims [gx, gy, gz]")
        print(f"   - obstacle_features (from /lidar_projected): 359 dims")
        print(f"   (Policy will encode lidar: 359 → 36 dims internally)")
    
    def start(self):
        """Start the controller."""
        self.robot.start()
        time.sleep(1.0)
        
        if not SDK_AVAILABLE:
            self.initial_position = (0.0, 0.0, 0.0)
            print(f"📍 Initial position (simulation): x=0.0, y=0.0, yaw=0.0")
        else:
            self.initial_position = (0.0, 0.0, 0.0)
            print(f"📍 Initial position (real robot): x=0.0, y=0.0, yaw=0.0")
    
    def stop(self):
        """Stop the controller."""
        self.robot.send_velocity_command(0.0, 0.0, 0.0)
        self.robot.stop()
    
    def set_goal(self, x: float, y: float, yaw: float):
        """Set navigation goal in world frame."""
        self.current_goal = NavigationGoal(x=x, y=y, yaw=yaw)
        
        state = self.robot.get_state()
        relative_goal = self._compute_relative_goal(state)
        initial_distance = np.linalg.norm(relative_goal[:2])
        
        print(f"🎯 New goal: x={x:.2f}, y={y:.2f}, yaw={np.degrees(yaw):.1f}°")
        print(f"📏 Initial distance to goal: {initial_distance:.2f}m")
    
    def _get_yaw_from_quat(self, state: RobotState) -> float:
        """Extract yaw angle from quaternion."""
        w, x, y, z = state.quat_w, state.quat_x, state.quat_y, state.quat_z
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return yaw
    
    def _compute_relative_goal(self, state: RobotState) -> np.ndarray:
        """
        计算目标位置相对于机器人本体坐标系的pose_command。
        
        使用ROS定位系统（AMCL/odometry/TF）获取机器人在地图坐标系中的位置，
        然后计算目标点相对于机器人本体的位置和朝向。
        """
        # 优先从robot接口获取rviz goal，如果没有则使用self.current_goal
        goal = None
        if hasattr(self.robot, 'current_goal') and self.robot.current_goal is not None:
            with self.robot.goal_lock:
                goal = self.robot.current_goal
        elif self.current_goal is not None:
            goal = self.current_goal
        
        if goal is None:
            return np.zeros(3, dtype=np.float32)
        
        # Goal在地图坐标系中的位置（从rviz获取）
        goal_x_w = goal.x
        goal_y_w = goal.y
        goal_yaw_w = goal.yaw
        
        # 机器人在地图坐标系中的位置（从ROS定位系统获取：AMCL/odometry/TF）
        fused_pos = self.robot.get_fused_position()
        robot_x_w = fused_pos[0]
        robot_y_w = fused_pos[1]
        robot_yaw_w = fused_pos[2]
        
        # 在世界坐标系中的差值
        delta_x_w = goal_x_w - robot_x_w
        delta_y_w = goal_y_w - robot_y_w
        
        # 旋转到机器人本体坐标系
        # 旋转矩阵：从世界坐标系到本体坐标系
        cos_yaw = np.cos(robot_yaw_w)
        sin_yaw = np.sin(robot_yaw_w)
        
        # 将世界坐标系的差值转换到本体坐标系
        delta_x_b = cos_yaw * delta_x_w + sin_yaw * delta_y_w
        delta_y_b = -sin_yaw * delta_x_w + cos_yaw * delta_y_w
        
        # 目标朝向相对于机器人当前朝向的差值
        delta_yaw = self._normalize_angle(goal_yaw_w - robot_yaw_w)
        
        return np.array([delta_x_b, delta_y_b, delta_yaw], dtype=np.float32)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _compute_projected_gravity(self, state: RobotState) -> np.ndarray:
        """
        计算重力在机器人本体坐标系下的投影。
        
        重力在世界坐标系中为 [0, 0, -g]，需要转换到机器人本体坐标系。
        使用四元数旋转矩阵将世界坐标系向量转换到本体坐标系。
        
        Args:
            state: 机器人状态（包含四元数）
        
        Returns:
            重力投影向量 [gx, gy, gz]，形状 (3,)
        """
        # 训练侧常用的是“单位重力方向向量”（而不是 m/s² 的 9.81），即 [0, 0, -1]，
        # 这样 projected_gravity 的范围稳定在 [-1, 1]，更适合作为网络输入。
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        # 四元数 (w, x, y, z)
        qw, qx, qy, qz = state.quat_w, state.quat_x, state.quat_y, state.quat_z
        
        # 使用四元数旋转矩阵公式：R = q * v * q^(-1)
        # 对于从世界坐标系到本体坐标系的转换，使用逆旋转：v_b = q^(-1) * v_w * q
        # 等价于使用旋转矩阵：R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
        #                            [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
        #                            [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
        # 但这是从本体到世界的旋转，我们需要逆旋转（转置）
        
        # 更直接的方法：使用四元数旋转公式
        # v' = q * [0, vx, vy, vz] * q^(-1) 其中 q^(-1) = [w, -x, -y, -z]
        # 对于逆旋转（世界到本体）：v_b = q^(-1) * [0, v_w] * q
        
        # 构建旋转矩阵（从世界到本体坐标系）
        # R_w_to_b = R_b_to_w^T
        # 从本体到世界的旋转矩阵：
        R_b_to_w = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float32)
        
        # 转置得到从世界到本体的旋转矩阵
        R_w_to_b = R_b_to_w.T
        
        # 应用旋转：gravity_b = R_w_to_b * gravity_w
        gravity_b = R_w_to_b @ gravity_w
        
        return gravity_b.astype(np.float32)
    
    def _construct_observation(self, state: RobotState) -> torch.Tensor:
        """
        构建366维观测向量，与训练配置完全一致。
        
        观测结构（与navigation_env_mlp_cfg.py中的PolicyCfg一致）：
        - pose_command (4): [x, y, z, heading] 目标姿态（机器人本体坐标系）
        - projected_gravity (3): [gx, gy, gz] 重力投影（机器人本体坐标系）
        - obstacle_features (359): 原始360° LiDAR距离数据（从/lidar_projected话题）
        
        总计: 4 + 3 + 359 = 366 dims
        
        Returns:
            观测张量 (1, 366)
        """
        obs_list = []
        
        # 1. Pose command (4 dims): [x, y, z, heading] in base frame
        relative_goal = self._compute_relative_goal(state)
        z_coord = 0.0  # 2D navigation
        pose_command = np.array([relative_goal[0], relative_goal[1], z_coord, relative_goal[2]], dtype=np.float32)
        obs_list.append(pose_command)
        
        # 2. Projected gravity (3 dims): [gx, gy, gz] in base frame
        projected_gravity = self._compute_projected_gravity(state)
        obs_list.append(projected_gravity)
        
        # 3. Obstacle features (359 dims): Raw LiDAR ranges from /lidar_projected topic
        lidar_data = self.get_lidar_projected(state)
        obs_list.append(lidar_data)
        
        # Concatenate all observations
        obs = np.concatenate(obs_list)
        
        # Validate observation dimension (must match training: 366 dims)
        assert obs.shape[0] == 366, f"Expected 366 dims (4+3+359), got {obs.shape[0]}"
        
        # Convert to torch tensor
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        return obs_tensor
    
    def get_lidar_projected(self, state: RobotState) -> np.ndarray:
        """
        Get lidar projected data (359 dims) from /lidar_projected topic.
        
        Returns:
            Array of distances, shape (359,)
        """
        if self.robot.use_lidar and state.lidar_projected is not None:
            # Thread-safe read
            with self.robot.lidar_data_lock:
                lidar_data = state.lidar_projected.copy()
            
            # Validate length
            if len(lidar_data) != 359:
                print(f"⚠️  Warning: lidar_projected length is {len(lidar_data)}, expected 359. Using max distance.")
                return np.full(359, self.max_lidar_distance, dtype=np.float32)
            
            # Clip to valid range and ensure finite values
            lidar_data = np.clip(lidar_data, 0.0, self.max_lidar_distance)
            lidar_data = np.nan_to_num(lidar_data, nan=self.max_lidar_distance, posinf=self.max_lidar_distance, neginf=0.0)

            # Optional alignment (only if you have validated /lidar_projected indexing)
            if self.lidar_reverse:
                lidar_data = lidar_data[::-1].copy()
            if self.lidar_angle_offset_deg != 0:
                # Positive offset rotates indices so that "new[0]" comes from "old[offset]".
                # This is useful when the publisher uses a different 0-degree reference.
                lidar_data = np.roll(lidar_data, -self.lidar_angle_offset_deg)
            
            return lidar_data.astype(np.float32)
        else:
            # No lidar data - return max distance for all rays
            return np.full(359, self.max_lidar_distance, dtype=np.float32)
    
    def print_lidar_8_directions(self, lidar_ranges_359: np.ndarray):
        """Print 8-direction lidar summary."""
        # Show 8 directions (every 45°)
        directions = [
            (0, "Front"),
            (45, "Front-Left"),
            (90, "Left"),
            (135, "Rear-Left"),
            (180, "Rear"),
            (225, "Rear-Right"),
            (270, "Right"),
            (315, "Front-Right"),
        ]
        
        print("8-Direction Summary:")
        for angle, name in directions:
            idx = angle % 359
            distance = lidar_ranges_359[idx]
            status = "🟢" if distance > 2.0 else "🟡" if distance > 1.0 else "🔴"
            print(f"      {status} {name:12s} ({angle:3d}°): {distance:.2f}m")
    
    def print_lidar_data(self, lidar_ranges_359: np.ndarray):
        """Print 359-dimensional lidar data summary."""
        print(f"📡 Lidar Projected (359 rays):")
        
        # Summary statistics
        print(f"   Min: {np.min(lidar_ranges_359):.2f}m, Max: {np.max(lidar_ranges_359):.2f}m, Mean: {np.mean(lidar_ranges_359):.2f}m")
        
        # Show 8 directions (every 45°)
        directions = [
            (0, "Front"),
            (45, "Front-Left"),
            (90, "Left"),
            (135, "Rear-Left"),
            (180, "Rear"),
            (225, "Rear-Right"),
            (270, "Right"),
            (315, "Front-Right"),
        ]
        
        print("   8-Direction Summary:")
        for angle, name in directions:
            idx = angle % 359
            distance = lidar_ranges_359[idx]
            status = "🟢" if distance > 2.0 else "🟡" if distance > 1.0 else "🔴"
            print(f"      {status} {name:12s} ({angle:3d}°): {distance:.2f}m")
    
    def step(self) -> Tuple[float, float, float]:
        """
        Execute one control step.
        
        Returns:
            (vx, vy, vyaw) velocity commands
        """
        # Get current state
        state = self.robot.get_state()
        
        # Construct observation (366 dims, matches training config)
        obs = self._construct_observation(state)
        
        # Inference (policy will encode lidar internally)
        with torch.no_grad():
            action = self.policy(obs)
        
        # Extract velocity commands
        vx = action[0, 0].item()
        vy = action[0, 1].item()
        vyaw = action[0, 2].item()
        
        # Clip to safe ranges
        vx = np.clip(vx, -0.3, 0.3)
        vy = np.clip(vy, -0.15, 0.15)
        vyaw = np.clip(vyaw, -0.1, 0.1)
        
        # Send command
        self.robot.send_velocity_command(vx, vy, vyaw)
        
        return vx, vy, vyaw
    
    def check_goal_reached(self, threshold: float = 0.3) -> bool:
        """Check if goal is reached."""
        # 检查是否有goal（从rviz或手动设置）
        goal = None
        if hasattr(self.robot, 'current_goal') and self.robot.current_goal is not None:
            with self.robot.goal_lock:
                goal = self.robot.current_goal
        elif self.current_goal is not None:
            goal = self.current_goal
        
        if goal is None:
            return False
        
        state = self.robot.get_state()
        relative_goal = self._compute_relative_goal(state)
        
        distance = np.linalg.norm(relative_goal[:2])
        return bool(distance < threshold)
    
    def print_status(self):
        """Print current status."""
        state = self.robot.get_state()
        
        # Get position estimates
        fused_pos = self.robot.get_fused_position()
        odom_pos = self.robot.odometry_position
        
        # 获取定位系统位置
        localization_pos = None
        if hasattr(self.robot, 'localization_position'):
            with self.robot.localization_lock:
                localization_pos = self.robot.localization_position.copy()
        
        print(f"\n{'='*60}")
        print(f"📍 Fused Position (ROS定位): x={fused_pos[0]:.2f}, y={fused_pos[1]:.2f}, yaw={np.degrees(fused_pos[2]):.1f}°")
        if localization_pos is not None:
            print(f"📍 Localization (AMCL/Odom): x={localization_pos[0]:.2f}, y={localization_pos[1]:.2f}, yaw={np.degrees(localization_pos[2]):.1f}°")
        print(f"📊 Odometry (SDK): x={odom_pos[0]:.2f}, y={odom_pos[1]:.2f}, yaw={np.degrees(odom_pos[2]):.1f}°")
        print(f"🏃 Velocity: vx={state.vel_x:.2f}, vy={state.vel_y:.2f}, vz={state.vel_z:.2f}")
        print(f"🔄 Ang Vel: wx={state.omega_x:.2f}, wy={state.omega_y:.2f}, wz={state.omega_z:.2f}")
        
        # 检查goal（优先从rviz获取）
        goal = None
        if hasattr(self.robot, 'current_goal') and self.robot.current_goal is not None:
            with self.robot.goal_lock:
                goal = self.robot.current_goal
        elif self.current_goal is not None:
            goal = self.current_goal
        
        if goal is not None:
            relative_goal = self._compute_relative_goal(state)
            distance = np.linalg.norm(relative_goal[:2])
            print(f"🎯 Goal (Map frame): x={goal.x:.2f}, y={goal.y:.2f}, yaw={np.degrees(goal.yaw):.1f}°")
            print(f"📏 Distance to goal: {distance:.2f}m")
            print(f"🔺 Relative (Base frame): dx={relative_goal[0]:.2f}, dy={relative_goal[1]:.2f}, dyaw={np.degrees(relative_goal[2]):.1f}°")
        else:
            print(f"🎯 Goal: None (等待RViz 2D Nav Goal或手动设置)")
        
        print(f"📡 Lidar: {'Enabled (/lidar_projected, 359 dims)' if self.robot.use_lidar else 'Disabled'}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Deploy MLP navigation policy on Unitree Go2 (Simplified Version)")
    parser.add_argument(
        "--policy",
        type=str,
        default="policy_with_encoder.pt",
        help="Path to trained MLP policy (.pt file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run policy inference",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Control loop rate (Hz)",
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=0.3,
        help="Goal reached threshold (meters)",
    )
    parser.add_argument(
        "--no-lidar",
        action="store_true",
        help="Disable lidar sensor",
    )
    parser.add_argument(
        "--lidar-topic",
        type=str,
        default="/lidar_projected",
        help="ROS topic for lidar projected data (std_msgs::Float32MultiArray, 359 dims)",
    )
    parser.add_argument(
        "--max-lidar-distance",
        type=float,
        default=8.0,  # 与训练配置一致 (navigation_env_mlp_cfg.py: max_distance=8.0)
        help="Maximum lidar distance (meters)",
    )
    parser.add_argument(
        "--lidar-angle-offset-deg",
        type=int,
        default=0,
        help="Optional LiDAR index offset (degrees). Positive means new[0]=old[offset].",
    )
    parser.add_argument(
        "--lidar-reverse",
        action="store_true",
        help="Optional LiDAR index reversal (use only after validating /lidar_projected semantics).",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 Unitree Go2 Navigation Deployment (MLP Simplified Version)")
    print("="*60)
    print(f"📦 Policy: {args.policy}")
    print(f"💻 Device: {args.device}")
    print(f"⚡ Control Rate: {args.rate} Hz")
    print(f"🎯 Goal Threshold: {args.goal_threshold} m")
    print(f"📡 Lidar: {'Disabled' if args.no_lidar else f'Enabled ({args.lidar_topic}, 359 dims, max {args.max_lidar_distance}m)'}")
    print("="*60)
    
    # Initialize controller
    controller = NavigationController(
        policy_path=args.policy,
        device=args.device,
        use_lidar=not args.no_lidar,
        lidar_topic=args.lidar_topic,
        max_lidar_distance=args.max_lidar_distance,
        lidar_angle_offset_deg=args.lidar_angle_offset_deg,
        lidar_reverse=args.lidar_reverse,
    )
    
    try:
        # Start controller
        controller.start()
        
        print("\n📝 Goal Input Mode")
        print("方式1: 在RViz中使用'2D Nav Goal'工具设置目标点（推荐）")
        print("方式2: 手动输入目标坐标 (格式: x y yaw)")
        print("命令: 'status' 查看状态, 'q' 退出\n")
        
        dt = 1.0 / args.rate
        last_goal_check_time = time.time()
        navigating = False
        
        while True:
            # 检查是否有goal（从rviz或手动设置）
            goal = None
            if hasattr(controller.robot, 'current_goal') and controller.robot.current_goal is not None:
                with controller.robot.goal_lock:
                    goal = controller.robot.current_goal
            elif controller.current_goal is not None:
                goal = controller.current_goal
            
            # 如果有goal且不在导航中，开始导航
            if goal is not None and not navigating:
                print("\n🚀 开始导航到目标点...")
                print("在RViz中设置新的2D Nav Goal可以更新目标")
                print("按 Ctrl+C 停止导航\n")
                navigating = True
                last_goal_check_time = time.time()
            
            # 如果正在导航，执行控制循环
            if navigating and goal is not None:
                try:
                    # 检查是否到达目标
                    if controller.check_goal_reached(args.goal_threshold):
                        print("\n✅ 目标已到达!")
                        controller.robot.send_velocity_command(0.0, 0.0, 0.0)
                        # 清除goal
                        if hasattr(controller.robot, 'current_goal'):
                            with controller.robot.goal_lock:
                                controller.robot.current_goal = None
                        controller.current_goal = None
                        navigating = False
                        controller.print_status()
                        continue
                    
                    # 执行控制步骤
                    vx, vy, vyaw = controller.step()
                    
                    # 定期打印状态
                    current_time = time.time()
                    if current_time - last_goal_check_time >= 2.0:  # 每2秒打印一次
                        state = controller.robot.get_state()
                        relative_goal = controller._compute_relative_goal(state)
                        distance_to_goal = np.linalg.norm(relative_goal[:2])
                        fused_pos = controller.robot.get_fused_position()
                        
                        print(f"⚡ 速度命令: vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}")
                        print(f"📍 当前位置 (地图坐标系): x={fused_pos[0]:.2f}, y={fused_pos[1]:.2f}, yaw={np.degrees(fused_pos[2]):.1f}°")
                        print(f"📏 到目标距离: {distance_to_goal:.2f}m")
                        print(f"🔺 相对目标 (本体坐标系): dx={relative_goal[0]:.2f}, dy={relative_goal[1]:.2f}, dyaw={np.degrees(relative_goal[2]):.1f}°")
                        
                        # 进度条
                        max_distance = 10.0
                        progress = max(0, min(1.0, 1.0 - distance_to_goal / max_distance))
                        bar_length = 20
                        filled_length = int(bar_length * progress)
                        bar = "█" * filled_length + "░" * (bar_length - filled_length)
                        print(f"📊 进度: [{bar}] {progress*100:.1f}%\n")
                        
                        last_goal_check_time = current_time
                    
                    time.sleep(dt)
                    
                except KeyboardInterrupt:
                    print("\n⏸️  导航已中断")
                    controller.robot.send_velocity_command(0.0, 0.0, 0.0)
                    navigating = False
                    # 不清除goal，允许继续导航
                    continue
            else:
                # 没有goal，等待用户输入或rviz goal
                # 使用非阻塞方式检查用户输入
                import select
                import sys
                
                if sys.stdin.isatty():  # 只在终端模式下检查输入
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        user_input = input().strip()
                        if user_input.lower() == 'q':
                            break
                        elif user_input.lower() == 'status':
                            controller.print_status()
                            continue
                        elif user_input.lower().startswith('goal') or len(user_input.split()) == 3:
                            try:
                                parts = user_input.split()
                                if len(parts) == 3:
                                    x, y, yaw = map(float, parts)
                                    controller.set_goal(x, y, yaw)
                                    navigating = True
                                    print(f"✅ 手动设置目标: x={x:.2f}, y={y:.2f}, yaw={np.degrees(yaw):.1f}°")
                                else:
                                    print("❌ 无效输入。格式: x y yaw")
                            except ValueError:
                                print("❌ 无效数字")
                        else:
                            print("❌ 未知命令。使用 'status' 查看状态, 'q' 退出, 或输入 'x y yaw' 设置目标")
                else:
                    # 非终端模式，只等待rviz goal
                    time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    
    finally:
        # Clean shutdown
        controller.stop()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()

