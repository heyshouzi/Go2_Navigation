# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for navigation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_mlp_encoding(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    encoder_output_dim: int = 36,
) -> torch.Tensor:
    """
    返回每条激光的距离值，在机器人本体坐标系下的平面距离。
    未命中的射线 distance = sensor.cfg.max_distance

    Args:
        env: 环境实例
        sensor_cfg: 传感器配置
        encoder_output_dim: 编码器输出维度

    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 世界坐标下相对向量
    ray_hits_w = sensor.data.ray_hits_w  # (env, rays, 3)
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)  # (env, 1, 3)
    relative_vec_w = ray_hits_w - sensor_pos_w  # (env, rays, 3)

    # 识别未命中的射线（ray_hits_w 中包含 inf 或 NaN）
    # 检查每个射线的任意坐标是否为 inf 或 NaN
    is_missed = torch.isinf(ray_hits_w).any(dim=-1) | torch.isnan(ray_hits_w).any(
        dim=-1
    )  # (env, rays)

    # 对于未命中的射线，将相对向量设置为零，避免在坐标变换时产生 NaN
    # 后续会用 max_distance 替换这些值
    relative_vec_w = torch.where(
        is_missed.unsqueeze(-1), torch.zeros_like(relative_vec_w), relative_vec_w
    )

    # 使用 quat_apply_inverse 将世界坐标系下的向量转换到传感器本体坐标系
    quat_w = sensor.data.quat_w  # (env, 4) - (w, x, y, z)

    # 将相对向量从世界坐标系转换到传感器本体坐标系
    num_envs, num_rays, _ = relative_vec_w.shape
    relative_vec_w_flat = relative_vec_w.reshape(-1, 3)  # (env * rays, 3)
    quat_w_expanded = (
        quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4)
    )  # (env * rays, 4)

    # 应用逆旋转：从世界坐标系转换到传感器坐标系
    relative_vec_b_flat = math_utils.quat_apply_inverse(
        quat_w_expanded, relative_vec_w_flat
    )
    relative_vec_b = relative_vec_b_flat.reshape(
        num_envs, num_rays, 3
    )  # (env, rays, 3)

    # 计算平面距离（仅使用 x-y 平面，忽略 z 轴）
    distance = torch.norm(relative_vec_b[..., :2], dim=-1)  # (env, rays)

    # 处理未命中的射线和超出最大距离的情况
    max_d = sensor.cfg.max_distance
    distance = torch.where(
        is_missed | torch.isnan(distance) | torch.isinf(distance) | (distance > max_d),
        torch.full_like(distance, max_d),
        distance,
    )

    # 最终安全检查：确保距离在有效范围内 [0, max_d]
    distance = torch.clamp(distance, 0.0, max_d)

    # 保存配置（仅首次调用时）
    if not hasattr(env, "_lidar_config"):
        env._lidar_config = {
            "num_rays": distance.shape[1],
            "encoder_output_dim": encoder_output_dim,
            "max_distance": max_d,
        }
        print(f"[LiDAR] {distance.shape[1]} rays → distance only, maxD={max_d}")

    return distance
