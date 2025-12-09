# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def heading_command_error_squared(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """使用平方惩罚，对大偏差更敏感"""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b ** 2  # 平方惩罚

def obstacle_safety_distance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float = 1.0,
    danger_distance: float = 0.5,
) -> torch.Tensor:
    """静态障碍物安全距离奖励。
    
    返回违规程度（正值），配合负weight使用以实现惩罚。
    - 距离 >= safe_distance: 返回 0 (安全，无违规)
    - danger_distance < 距离 < safe_distance: 线性违规 (0-0.5)
    - 距离 <= danger_distance: 强违规（指数增长，0.5-1.0）
    
    使用示例：
        RewTerm(func=mdp.obstacle_safety_distance, weight=-50.0, ...)
        → 接近障碍物时获得负奖励（惩罚）
    
    Args:
        env: The environment.
        sensor_cfg: The raycaster sensor configuration (obstacle_scanner).
        safe_distance: 安全距离阈值（米）。超过此距离无违规。Defaults to 1.0.
        danger_distance: 危险距离阈值（米）。低于此距离强违规。Defaults to 0.5.
        
    Returns:
        Violation score for each environment. Shape: (num_envs,).
        返回值为正（违规程度），距离越近值越大，配合负weight实现惩罚。
    """
    # 获取RayCaster传感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 获取所有射线的距离
    ray_hits_w = sensor.data.ray_hits_w  # (num_envs, num_rays, 3)
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    
    # 计算到障碍物的距离
    distances = torch.norm(ray_hits_w - sensor_pos_w, dim=-1)  # (num_envs, num_rays)
    
    # 处理inf（未命中的射线视为安全距离）
    distances = torch.where(
        torch.isinf(distances),
        torch.full_like(distances, safe_distance * 2),  # 远大于安全距离
        distances
    )
    
    # 取最小距离（最危险的障碍物）
    min_distances = torch.min(distances, dim=1)[0]  # (num_envs,)
    
    # 计算违规程度（分段函数，返回正值）
    violation = torch.zeros_like(min_distances)
    
    # 危险区域（d < danger_distance）：指数违规 (0.5-1.0)
    danger_mask = min_distances < danger_distance
    if danger_mask.any():
        # 指数函数：距离越近，违规越大
        ratio = min_distances[danger_mask] / danger_distance
        # 将原来的[-1, ~0]映射到[1.0, 0.5]
        violation[danger_mask] = torch.exp(-2.0 * ratio) - math.exp(-2.0) + 1.0  # 范围: [0.5, 1.0]
    
    # 警告区域（danger_distance <= d < safe_distance）：线性违规 (0-0.5)
    warning_mask = (min_distances >= danger_distance) & (min_distances < safe_distance)
    if warning_mask.any():
        # 线性插值：距离越近违规越大
        ratio = (min_distances[warning_mask] - danger_distance) / (safe_distance - danger_distance)
        violation[warning_mask] = 0.5 * (1.0 - ratio)  # 范围: [0, 0.5]
    
    # 安全区域（d >= safe_distance）：无违规 (violation = 0)
    
    return violation


def obstacle_proximity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.8,
    kernel: str = "exp",
) -> torch.Tensor:
    """基于所有扇区的障碍物接近度惩罚（更平滑的版本）。
    
    考虑所有方向的障碍物，使用平均距离而不是最小距离。
    
    Args:
        env: The environment.
        sensor_cfg: The raycaster sensor configuration.
        threshold: 距离阈值（米），低于此值开始惩罚。Defaults to 0.8.
        kernel: 惩罚核函数类型 ("exp" 或 "linear")。Defaults to "exp".
        
    Returns:
        Proximity penalty. Shape: (num_envs,).
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 计算距离
    ray_hits_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(ray_hits_w - sensor_pos_w, dim=-1)
    
    # 处理inf
    distances = torch.where(
        torch.isinf(distances),
        torch.full_like(distances, threshold * 2),
        distances
    )
    
    # 计算平均前方距离（权重平均）
    # 可以考虑只计算前半球的射线
    mean_distance = torch.mean(distances, dim=1)
    
    # 计算惩罚
    if kernel == "exp":
        # 指数核：距离越近，惩罚增长越快
        violation = torch.clamp(threshold - mean_distance, min=0.0)
        penalty = -violation * torch.exp(violation / threshold)
    else:  # linear
        # 线性核：简单线性惩罚
        violation = torch.clamp(threshold - mean_distance, min=0.0)
        penalty = -violation / threshold
    
    return penalty

def time_efficiency_bonus(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    奖励快速接近目标的逻辑设计如下：

    - 奖励由两个部分组成：靠近目标的程度（距离）和接近目标的速度（速度方向）。
    - 距离目标越近，奖励越大（用距离的负指数函数做加权，越近权重越大）。
    - 只有当机器人朝向目标移动时，速度越大奖励越大（用速度在目标方向上的投影）。
    - 这样鼓励机器人既要靠近目标，也要以高效的速度朝目标前进。

    Args:
        env: 环境对象。
        command_name: 目标命令的名称。

    Returns:
        奖励值，shape: (num_envs,)
    """
    # 目标在base坐标系下的位置
    command = env.command_manager.get_command(command_name)  # (num_envs, 3/4)
    des_pos_b = command[:, :2]  # 只取x, y平面
    distance = torch.norm(des_pos_b, dim=1)  # (num_envs,)

    # 机器人在世界坐标系下的线速度（x, y）
    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]  # (num_envs, 2)
    # 机器人朝向目标的单位向量（在base系，需转到世界系）
    # 这里假设目标向量已在base系，需转到世界系。简化：假设base朝向与世界对齐。
    # 更严谨做法：需用base的朝向旋转des_pos_b到世界系。
    # 但如果目标命令本身就是在base系下的相对向量，则直接用即可。
    direction_to_goal = torch.nn.functional.normalize(des_pos_b, dim=1)  # (num_envs, 2)
    # 速度在目标方向上的投影
    velocity_towards_goal = (vel_w * direction_to_goal).sum(dim=1)  # (num_envs,)

    # 距离加权（距离越近，权重越大）
    proximity_weight = torch.exp(-distance / 2.0)  # (num_envs,)

    # 奖励 = 速度投影 * 距离加权 * 系数
    reward = velocity_towards_goal * proximity_weight * 0.1

    return reward


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.3,
) -> torch.Tensor:
    """成功到达目标点的奖励。
    
    当机器人进入目标区域（距离<threshold）时给予一次性大额奖励。
    这个奖励会在终止时触发，鼓励策略尽快到达目标。
    
    Args:
        env: 环境实例
        command_name: 命令名称
        distance_threshold: 认为到达目标的距离阈值（米）
        
    Returns:
        奖励张量：到达目标区域内的环境返回1.0，否则返回0.0
    """
    # 获取目标位置命令
    command = env.command_manager.get_command(command_name)
    target_pos_b = command[:, :3]  # 目标位置（机器人坐标系）
    
    # 计算到目标的距离（只考虑xy平面）
    distance = torch.norm(target_pos_b[:, :2], dim=1)
    
    # 如果距离小于阈值，返回1.0（成功到达），否则返回0.0
    return (distance < distance_threshold).float()


def goal_reached_bonus_time_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.3,
    base_reward: float = 100.0,
    time_bonus_weight: float = 100.0,
) -> torch.Tensor:
    """时间感知的成功到达奖励。
    
    鼓励机器人尽快到达目标：
    - 快速到达（剩余时间多）→ 高奖励
    - 慢速到达（剩余时间少）→ 低奖励
    
    Args:
        env: 环境实例
        command_name: 命令名称
        distance_threshold: 认为到达目标的距离阈值（米）
        base_reward: 基础奖励（总是给予）
        time_bonus_weight: 时间奖励权重（根据剩余时间计算）
        
    Returns:
        奖励张量：到达时 = base_reward + time_bonus，未到达 = 0
        
    示例：
        episode_length = 12秒 = 60步
        
        6秒到达（30步）：
          time_ratio = 30/60 = 0.5
          reward = 100 + (1-0.5)*100 = 150 ✅ 快速奖励
          
        11秒到达（55步）：
          time_ratio = 55/60 = 0.92
          reward = 100 + (1-0.92)*100 = 108 🟡 慢速奖励
    """
    # 获取目标位置命令
    command = env.command_manager.get_command(command_name)
    target_pos_b = command[:, :3]
    
    # 计算到目标的距离（只考虑xy平面）
    distance = torch.norm(target_pos_b[:, :2], dim=1)
    reached = (distance < distance_threshold).float()
    
    # 计算已用时间比例 (0-1)
    time_ratio = env.episode_length_buf.float() / env.max_episode_length
    
    # 计算时间奖励：剩余时间越多，奖励越高
    time_bonus = (1.0 - time_ratio) * time_bonus_weight
    
    # 总奖励 = 基础奖励 + 时间奖励（仅在到达时给予）
    return reached * (base_reward + time_bonus)


def velocity_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """速度平滑性惩罚。
    
    返回违规程度（正值），配合负weight使用以实现惩罚。
    - 计算当前速度与上一帧速度的差值
    - 差值越大，违规程度越高
    
    使用示例：
        RewTerm(func=mdp.velocity_smoothness_penalty, weight=-0.1, ...)
        → 速度突变时获得负奖励（惩罚）
    
    Args:
        env: 环境实例
        
    Returns:
        违规程度张量：shape (num_envs,)，值为正（加速度大小）
        
    注意：
        这个函数需要环境存储上一帧的速度。
        如果是第一步（没有历史），返回0（无违规）。
    """
    robot = env.scene["robot"]
    current_vel = robot.data.root_lin_vel_b[:, :2]  # 当前线速度 (x, y)
    
    # 检查是否有历史速度记录
    if not hasattr(env, '_last_lin_vel'):
        # 第一步，初始化历史速度
        env._last_lin_vel = current_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算速度变化（加速度的近似）
    vel_change = current_vel - env._last_lin_vel
    acceleration = torch.norm(vel_change, dim=1)  # L2范数
    
    # 更新历史速度
    env._last_lin_vel = current_vel.clone()
    
    # 返回正值（违规程度），配合负weight实现惩罚
    return acceleration


def time_efficiency_bonus_fixed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """修正版的时间效率奖励（正确的坐标系转换）。
    
    奖励朝向目标快速移动的行为：
    - 速度在目标方向的投影越大，奖励越高
    - 距离目标越近，权重越大（鼓励接近时保持速度）
    
    Args:
        env: 环境对象
        command_name: 目标命令的名称
        
    Returns:
        奖励值，shape: (num_envs,)
        
    修正内容：
        正确处理坐标系转换，将目标方向从机器人坐标系转换到世界坐标系
    """
    # 获取目标位置（机器人坐标系）
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]  # (num_envs, 2) - 机器人坐标系
    distance = torch.norm(des_pos_b, dim=1)
    
    # 获取机器人状态
    robot = env.scene["robot"]
    yaw = robot.data.heading_w  # (num_envs,) - 世界坐标系中的朝向角
    
    # 将目标方向从机器人坐标系转换到世界坐标系
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    # 旋转矩阵：R = [[cos, -sin], [sin, cos]]
    direction_w_x = des_pos_b[:, 0] * cos_yaw - des_pos_b[:, 1] * sin_yaw
    direction_w_y = des_pos_b[:, 0] * sin_yaw + des_pos_b[:, 1] * cos_yaw
    direction_w = torch.stack([direction_w_x, direction_w_y], dim=1)
    direction_w = torch.nn.functional.normalize(direction_w, dim=1)
    
    # 速度在目标方向的投影（现在都在世界坐标系）
    vel_w = robot.data.root_lin_vel_w[:, :2]  # (num_envs, 2) - 世界坐标系
    velocity_towards_goal = (vel_w * direction_w).sum(dim=1)  # 标量积
    
    # 距离加权（距离越近，权重越大）
    proximity_weight = torch.exp(-distance / 2.0)
    
    # 奖励 = 速度投影 * 距离加权 * 系数
    return velocity_towards_goal * proximity_weight * 0.1


def goal_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """奖励向目标前进的进度（距离减少量）。
    
    核心思想：每一步如果缩短了与目标的距离，就给正奖励。
    - 距离减少 → 正奖励（鼓励）
    - 距离增加 → 负奖励（惩罚）
    - 距离不变 → 零奖励
    
    这是一个密集奖励，直接反映策略的效果。
    
    Args:
        env: 环境
        command_name: 目标命令名称
        
    Returns:
        奖励：距离减少量（米），正值=靠近，负值=远离
    """
    # 获取当前到目标的距离
    command = env.command_manager.get_command(command_name)
    current_distance = torch.norm(command[:, :2], dim=1)  # 只看xy平面
    
    # 存储键
    storage_key = f"previous_distance_{command_name}"
    
    # 🔧 检查episode重置：如果reset标志存在，清除历史
    if hasattr(env, 'episode_length_buf'):
        # episode_length_buf=0 表示刚刚重置
        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            if storage_key in env.extras:
                # 重置对应环境的历史距离
                env.extras[storage_key][reset_mask] = current_distance[reset_mask]
    
    # 初始化：如果是第一次调用，存储当前距离
    if storage_key not in env.extras:
        env.extras[storage_key] = current_distance.clone()
        return torch.zeros_like(current_distance)  # 第一步没有历史，返回0
    
    # 获取上一步的距离
    previous_distance = env.extras[storage_key]
    
    # 计算进度：距离减少量
    # positive = 靠近目标（好）
    # negative = 远离目标（差）
    progress = previous_distance - current_distance
    
    # 更新存储（为下一步准备）
    env.extras[storage_key] = current_distance.clone()
    
    # 放大系数：将米转换为合适的奖励尺度
    # 例如：缩短0.1米 → 奖励1.0
    return progress * 10.0


def heading_alignment(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """奖励机器人朝向与目标方向的对齐程度。
    
    使用余弦相似度：朝向完全对齐=1.0，完全相反=-1.0
    相比heading_command_error系列（基于角度误差），这个使用余弦更平滑。
    
    Args:
        env: 环境
        command_name: 目标命令名称
        
    Returns:
        奖励：朝向对齐度（0到1之间，1表示完全对齐）
    """
    # 获取目标位置（机器人坐标系）
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]  # (num_envs, 2)
    
    # 计算到目标的角度（机器人坐标系中）
    # atan2(y, x)：x轴=前方，y轴=左侧
    angle_to_goal = torch.atan2(des_pos_b[:, 1], des_pos_b[:, 0])
    
    # 计算朝向对齐度（使用余弦）
    # cos(0) = 1.0 (完全对齐)
    # cos(π) = -1.0 (完全相反)
    alignment = torch.cos(angle_to_goal)
    
    # 映射到 [0, 1]：完全对齐=1，垂直=0.5，相反=0
    alignment_normalized = (alignment + 1.0) / 2.0
    
    return alignment_normalized


def safe_velocity_near_obstacles(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    safe_distance: float = 2.0,
    danger_distance: float = 1.0,
    safe_speed: float = 1.5,
) -> torch.Tensor:
    """奖励机器人在接近障碍物时减速。
    
    返回违规程度（正值），配合负weight使用以实现惩罚。
    
    核心思想：
    - 离障碍物远时（>safe_distance），可以全速前进 → 返回0
    - 接近障碍物时（<safe_distance），速度应该降低 → 超速返回正值
    - 很近时（<danger_distance），应该几乎停止 → 超速惩罚更大
    
    使用示例：
        RewTerm(func=mdp.safe_velocity_near_obstacles, weight=-1.0, ...)
        → 接近障碍物时超速会获得负奖励（惩罚）
    
    Args:
        env: 环境
        sensor_cfg: 传感器配置
        asset_cfg: 机器人asset配置
        safe_distance: 安全距离（米），超过此距离可全速
        danger_distance: 危险距离（米），小于此距离应减速
        safe_speed: 安全速度上限（m/s）
        
    Returns:
        违规程度：速度超限时的违规程度（正值）
    """
    # 获取障碍物距离数据
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_distances = sensor.data.ray_hits_w[..., 0]  # (num_envs, num_rays)
    
    # 找到最近的障碍物距离
    min_distance = torch.min(ray_distances, dim=1)[0]  # (num_envs,)
    
    # 获取当前速度
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.root_lin_vel_w
    speed = torch.norm(velocity[:, :2], dim=1)  # xy平面速度
    
    # 计算危险度（0-1，1=非常危险）
    # danger_level = 1.0 when distance < danger_distance
    # danger_level = 0.0 when distance > safe_distance
    danger_level = torch.clamp(
        (safe_distance - min_distance) / (safe_distance - danger_distance),
        0.0,
        1.0,
    )
    
    # 计算期望的安全速度
    # 当danger_level=1时，期望速度=0.3m/s（几乎停止）
    # 当danger_level=0时，期望速度=safe_speed（正常速度）
    desired_speed = safe_speed * (1.0 - danger_level * 0.8)  # 最低降到20%速度
    
    # 如果实际速度超过期望速度，计算违规程度
    speed_violation = torch.clamp(speed - desired_speed, min=0.0)
    
    # 违规程度 = 速度违规量 * 危险度（返回正值）
    violation = speed_violation * danger_level * 2.0
    
    return violation


def heading_towards_velocity(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = 0.2
) -> torch.Tensor:
    """奖励机器人朝向其速度方向（而不是目标方向）。
    
    让机器人的头部朝向运动方向，这样可以：
    1. 更自然的运动（四足动物通常朝向运动方向）
    2. 更高效的避障（可以侧向/后退移动）
    3. 更灵活的导航策略
    
    Args:
        env: 环境
        asset_cfg: 机器人asset配置
        velocity_threshold: 速度阈值（m/s），低于此速度不计算奖励（避免原地打转时的噪声）
        
    Returns:
        奖励：朝向与速度方向的对齐度（0到1，1表示完全对齐）
    """
    # 获取asset
    asset = env.scene[asset_cfg.name]
    
    # 获取机器人在世界坐标系中的线速度 (num_envs, 3)
    velocity_w = asset.data.root_lin_vel_w
    velocity_xy = velocity_w[:, :2]  # 只看xy平面的速度
    
    # 计算速度大小
    speed = torch.norm(velocity_xy, dim=1)
    
    # 获取机器人朝向（yaw角）
    robot_heading = asset.data.heading_w  # (num_envs,)
    
    # 计算速度方向角度
    velocity_angle = torch.atan2(velocity_xy[:, 1], velocity_xy[:, 0])  # (num_envs,)
    
    # 计算朝向与速度方向的夹角
    angle_diff = velocity_angle - robot_heading
    
    # 归一化角度到[-π, π]
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    
    # 计算对齐度：cos(angle_diff)
    # cos(0) = 1.0 (完全对齐)
    # cos(π) = -1.0 (完全相反)
    alignment = torch.cos(angle_diff)
    
    # 映射到 [0, 1]
    alignment_normalized = (alignment + 1.0) / 2.0
    
    # 只在速度足够大时才给奖励（避免原地打转时的噪声）
    mask = speed > velocity_threshold
    reward = torch.where(mask, alignment_normalized, torch.zeros_like(alignment_normalized))
    
    return reward


def backward_motion_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚后退运动（倒着走）。
    
    计算机器人速度在其朝向方向上的投影：
    - 如果投影为负（后退），返回正的惩罚值
    - 如果投影为正（前进），返回0
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
        
    Returns:
        后退惩罚（正值表示后退程度，配合负weight使用）
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取机器人的线速度（世界坐标系）
    velocity_w = asset.data.root_lin_vel_w  # (num_envs, 3)
    
    # 获取机器人的朝向（世界坐标系，yaw角对应的方向向量）
    robot_heading = asset.data.heading_w  # (num_envs, 3)，已归一化
    
    # 计算速度在朝向上的投影（正值=前进，负值=后退）
    forward_speed = torch.sum(velocity_w * robot_heading, dim=1)  # (num_envs,)
    
    # 如果速度为负（后退），返回后退程度；否则返回0
    # 使用ReLU确保只惩罚后退，不奖励前进
    backward_amount = torch.clamp(-forward_speed, min=0.0)  # 后退速度的绝对值
    
    return backward_amount


def contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """运动过程中的碰撞惩罚（基于接触力）。
    
    实时监测机器人与环境的接触力，当接触力超过阈值时给予惩罚。
    这与终止条件不同，可以在运动过程中持续监测并惩罚轻微碰撞。
    
    Args:
        env: 环境实例
        sensor_cfg: 接触力传感器配置（如 "contact_forces"）
        threshold: 接触力阈值（牛顿），超过此值则认为发生碰撞
            默认 1.0N，低于终止条件的阈值（5.0N）
    
    Returns:
        碰撞惩罚（正值），配合负weight使用
    """
    # 获取接触力传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取指定部位的接触力 (num_envs, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    
    # 计算接触力的模（大小）
    force_magnitudes = torch.norm(contact_forces, dim=-1)  # (num_envs, num_bodies)
    
    # 取所有监测部位的最大接触力
    max_contact_force = torch.max(force_magnitudes, dim=-1)[0]  # (num_envs,)
    
    # 超过阈值的部分作为惩罚
    # penalty = max(0, force - threshold)
    penalty = torch.clamp(max_contact_force - threshold, min=0.0)
    
    # 归一化到 [0, 1] 范围，避免惩罚过大
    # 假设最大接触力不超过 20N
    max_expected_force = 20.0
    penalty_normalized = torch.clamp(penalty / max_expected_force, max=1.0)
    
    return penalty_normalized


def obstacle_proximity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    danger_distance: float = 0.3,
    warning_distance: float = 0.8,
) -> torch.Tensor:
    """基于LiDAR的障碍物接近惩罚。
    
    当机器人距离障碍物过近时给予惩罚，鼓励保持安全距离。
    这是一种预防性惩罚，在实际碰撞之前就开始起作用。
    
    Args:
        env: 环境实例
        sensor_cfg: LiDAR传感器配置（如 "obstacle_scanner"）
        danger_distance: 危险距离（米），低于此距离惩罚最大
        warning_distance: 警告距离（米），高于此距离无惩罚
    
    Returns:
        接近惩罚（正值），配合负weight使用
    """
    # 获取LiDAR传感器
    ray_caster: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 获取LiDAR距离数据 (num_envs, num_rays)
    distances = ray_caster.data.ray_hits_w[..., -1]
    
    # 找到每个环境的最小距离（最近障碍物）
    min_distance = torch.min(distances, dim=-1)[0]  # (num_envs,)
    
    # 计算惩罚
    # - 距离 >= warning_distance: penalty = 0
    # - danger_distance < 距离 < warning_distance: 线性插值
    # - 距离 <= danger_distance: penalty = 1.0
    penalty = torch.zeros_like(min_distance)
    
    # 在警告区间内，线性惩罚
    in_warning_zone = (min_distance >= danger_distance) & (min_distance < warning_distance)
    penalty[in_warning_zone] = (warning_distance - min_distance[in_warning_zone]) / (
        warning_distance - danger_distance
    )
    
    # 在危险区间内，最大惩罚
    in_danger_zone = min_distance < danger_distance
    penalty[in_danger_zone] = 1.0
    
    return penalty
