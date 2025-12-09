# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom curriculum functions for navigation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def adaptive_speed_requirement(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    success_threshold: float = 0.4,
    low_speed_weight: float = 0.3,
    high_speed_weight: float = 2.0,
    eval_interval: int = 50,  # 每50个iteration评估一次
) -> dict[str, float]:
    """根据成功率自适应调整速度要求。
    
    监控成功率，动态调整time_efficiency的权重：
    - 成功率 < 40%：降低速度要求（weight = 0.3）
    - 成功率 ≥ 40%：提高速度要求（weight = 2.0）
    
    Args:
        env: 环境实例
        env_ids: 环境ID（curriculum函数必须接受但可能不用）
        success_threshold: 成功率阈值（默认0.4 = 40%）
        low_speed_weight: 低速度要求的权重
        high_speed_weight: 高速度要求的权重
        eval_interval: 评估间隔（每N个iteration评估一次）
        
    Returns:
        包含当前状态的字典，用于日志记录
    """
    # 初始化状态变量
    if not hasattr(env, '_speed_curriculum_state'):
        env._speed_curriculum_state = {
            'current_weight': low_speed_weight,
            'iteration_count': 0,
            'success_count': 0,
            'episode_count': 0,
        }
    
    state = env._speed_curriculum_state
    state['iteration_count'] += 1
    
    # 统计本iteration的成功和总episode数
    # 通过检查哪些环境刚reset来统计
    if hasattr(env, 'reset_buf') and env.reset_buf is not None:
        # reset_buf为True表示这个环境刚终止
        terminated_envs = env.reset_buf.sum().item()
        state['episode_count'] += terminated_envs
        
        # 统计其中有多少是成功的
        if terminated_envs > 0 and hasattr(env.termination_manager, 'get_term'):
            try:
                goal_reached = env.termination_manager.get_term('goal_reached')
                success_envs = (goal_reached & env.reset_buf).sum().item()
                state['success_count'] += success_envs
            except:
                pass  # 如果获取失败，跳过
    
    # 每N个iterations评估一次
    if state['iteration_count'] % eval_interval == 0:
        # 计算成功率
        if state['episode_count'] > 0:
            success_rate = state['success_count'] / state['episode_count']
        else:
            success_rate = 0.0
        
        # 根据成功率决定目标权重
        if success_rate >= success_threshold:
            target_weight = high_speed_weight
            status = "high_speed"
        else:
            target_weight = low_speed_weight
            status = "low_speed"
        
        # 平滑过渡（避免突然跳变）
        current = state['current_weight']
        if current != target_weight:
            # 每次调整20%的差距
            state['current_weight'] = current + (target_weight - current) * 0.2
        
        # 重置计数器（为下一个周期准备）
        state['success_count'] = 0
        state['episode_count'] = 0
        
        # 更新reward manager中的权重
        if hasattr(env, 'reward_manager'):
            for i, term_name in enumerate(env.reward_manager._term_names):
                if 'time_efficiency' in term_name:
                    env.reward_manager._term_cfgs[i].weight = state['current_weight']
                    break
        
        # 返回状态（用于日志）
        return {
            "success_rate": success_rate,
            "speed_weight": state['current_weight'],
            "status": 1.0 if status == "high_speed" else 0.0,
        }
    
    # 非评估iteration，返回当前状态
    return {
        "speed_weight": state['current_weight'],
    }


def adaptive_terrain_difficulty(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    easy_threshold: float = 0.3,
    hard_threshold: float = 0.6,
    obstacle_range: tuple[int, int, int] = (20, 50, 80),  # 简单、中等、困难
    height_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.5, 1.5),  # 简单：低障碍物
        (1.0, 3.0),  # 中等：中等障碍物
        (1.5, 4.5),  # 困难：高障碍物
    ),
    eval_interval: int = 50,
) -> dict[str, float]:
    """根据成功率自适应调整地形难度。
    
    动态调整障碍物数量和高度：
    - 成功率 < 30%：简单地形（20个障碍物，0.5-1.5m高）
    - 成功率 30-60%：中等地形（50个障碍物，1.0-3.0m高）
    - 成功率 > 60%：困难地形（80个障碍物，1.5-4.5m高）
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        easy_threshold: 简单/中等的分界线（默认0.3 = 30%）
        hard_threshold: 中等/困难的分界线（默认0.6 = 60%）
        obstacle_range: (简单数量, 中等数量, 困难数量)
        height_range: ((简单min, 简单max), (中等min, 中等max), (困难min, 困难max))
        eval_interval: 评估间隔
        
    Returns:
        状态字典
    """
    # 初始化状态
    if not hasattr(env, '_terrain_curriculum_state'):
        env._terrain_curriculum_state = {
            'iteration_count': 0,
            'success_count': 0,
            'episode_count': 0,
            'current_difficulty': 0,  # 0=简单, 1=中等, 2=困难
            'current_obstacle_count': obstacle_range[0],
            'current_height_min': height_range[0][0],
            'current_height_max': height_range[0][1],
        }
    
    state = env._terrain_curriculum_state
    state['iteration_count'] += 1
    
    # 统计成功率（与速度课程共享逻辑）
    if hasattr(env, 'reset_buf') and env.reset_buf is not None:
        terminated_envs = env.reset_buf.sum().item()
        state['episode_count'] += terminated_envs
        
        if terminated_envs > 0 and hasattr(env.termination_manager, 'get_term'):
            try:
                goal_reached = env.termination_manager.get_term('goal_reached')
                success_envs = (goal_reached & env.reset_buf).sum().item()
                state['success_count'] += success_envs
            except:
                pass
    
    # 每N个iterations评估一次
    if state['iteration_count'] % eval_interval == 0:
        # 计算成功率
        if state['episode_count'] > 0:
            success_rate = state['success_count'] / state['episode_count']
        else:
            success_rate = 0.0
        
        # 根据成功率决定目标难度
        if success_rate < easy_threshold:
            target_difficulty = 0  # 简单
            target_obstacles = obstacle_range[0]
            target_height = height_range[0]
            status = "easy"
        elif success_rate < hard_threshold:
            target_difficulty = 1  # 中等
            target_obstacles = obstacle_range[1]
            target_height = height_range[1]
            status = "medium"
        else:
            target_difficulty = 2  # 困难
            target_obstacles = obstacle_range[2]
            target_height = height_range[2]
            status = "hard"
        
        # 只在难度变化时更新
        if target_difficulty != state['current_difficulty']:
            # 平滑过渡障碍物数量
            current_obs = state['current_obstacle_count']
            diff_obs = target_obstacles - current_obs
            state['current_obstacle_count'] = int(current_obs + diff_obs * 0.3)
            
            # 平滑过渡高度范围
            current_h_min = state['current_height_min']
            current_h_max = state['current_height_max']
            diff_h_min = target_height[0] - current_h_min
            diff_h_max = target_height[1] - current_h_max
            state['current_height_min'] = current_h_min + diff_h_min * 0.3
            state['current_height_max'] = current_h_max + diff_h_max * 0.3
            
            state['current_difficulty'] = target_difficulty
        
        # 重置计数器
        state['success_count'] = 0
        state['episode_count'] = 0
        
        # 返回状态
        return {
            "success_rate": success_rate,
            "difficulty": float(state['current_difficulty']),
            "obstacle_count": float(state['current_obstacle_count']),
            "height_min": state['current_height_min'],
            "height_max": state['current_height_max'],
            "status": float(target_difficulty),
        }
    
    # 非评估iteration
    return {
        "difficulty": float(state['current_difficulty']),
        "obstacle_count": float(state['current_obstacle_count']),
    }


def adaptive_collision_penalty_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_name: str = "contact_force_penalty",
    weight_levels: list[float] = [-1.0,  -3.0,  -5.0],  # 从轻到重的惩罚权重
    success_thresholds: list[float] = [0.6, 0.8, 0.9],  # 升级阈值
    eval_interval: int = 100,
    min_episodes_per_eval: int = 200,
    warmup_iterations: int = 400,  # 前400个回合保持-1权重
) -> dict[str, float]:
    """根据成功率自适应调整碰撞惩罚权重。
    
    实现碰撞惩罚权重的课程学习：
    - Level 0: 轻惩罚 (-1.0) - 让机器人先学会基本导航
    - Level 1: 较重惩罚 (-3.0) - 开始关注碰撞
    - Level 2: 重惩罚 (-5.0) - 进一步减少碰撞
    
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        term_name: 奖励项名称（默认 "contact_force_penalty"）
        weight_levels: 权重等级列表，从轻到重
        success_thresholds: 升级阈值列表，长度为 len(weight_levels) - 1
        eval_interval: 评估间隔（每N个iteration评估一次）
        min_episodes_per_eval: 最少样本量
        warmup_iterations: 预热期
        
    Returns:
        状态字典，用于日志记录
    """
    # 初始化状态
    if not hasattr(env, '_collision_curriculum_state'):
        env._collision_curriculum_state = {
            'last_rl_iteration': 0,
            'success_count': 0,
            'episode_count': 0,
            'current_level': 0,  # 从最轻惩罚开始
            'current_weight': weight_levels[0],
            'initialized_at_iteration': 0,
        }
        print(f"🎓 [Collision Curriculum] 初始化完成，初始权重: {weight_levels[0]}")
    
    state = env._collision_curriculum_state
    
    # 统计成功率（每次环境重置时累积）
    if hasattr(env, 'reset_buf') and env.reset_buf is not None:
        terminated_envs = env.reset_buf.sum().item()
        state['episode_count'] += terminated_envs
        
        if terminated_envs > 0 and hasattr(env.termination_manager, 'get_term'):
            try:
                goal_reached = env.termination_manager.get_term('goal_reached')
                success_envs = (goal_reached & env.reset_buf).sum().item()
                state['success_count'] += success_envs
            except:
                pass
    
    # 🔧 使用RL训练迭代数
    current_rl_iteration = 0
    if hasattr(env, 'rl_iteration'):
        current_rl_iteration = env.rl_iteration
    elif hasattr(env, 'learning_iteration'):
        current_rl_iteration = env.learning_iteration
    elif hasattr(env, 'train_iteration'):
        current_rl_iteration = env.train_iteration
    else:
        decimation = getattr(env, 'decimation', 16)
        current_rl_iteration = env.common_step_counter // decimation
    
    iterations_since_last_eval = current_rl_iteration - state['last_rl_iteration']
    
    # 预热阶段：跳过评估，先累积数据，确保权重保持为-1
    if current_rl_iteration - state['initialized_at_iteration'] < warmup_iterations:
        # 在预热期内，确保权重保持为初始值（-1.0）
        if state['current_weight'] != weight_levels[0]:
            state['current_weight'] = weight_levels[0]
            # 更新reward manager中的权重
            if hasattr(env, 'reward_manager'):
                try:
                    term_cfg = env.reward_manager.get_term_cfg(term_name)
                    term_cfg.weight = state['current_weight']
                    env.reward_manager.set_term_cfg(term_name, term_cfg)
                    print(f"🔧 [Collision Curriculum] 预热期强制保持权重: {state['current_weight']}")
                except Exception as e:
                    print(f"⚠️ 预热期更新权重失败: {e}")
        
        return {
            "current_level": float(state['current_level']),
            "current_weight": state['current_weight'],
            "warmup": float(warmup_iterations - (current_rl_iteration - state['initialized_at_iteration'])),
        }

    # 每N个RL iterations评估一次，且保证样本量足够
    if iterations_since_last_eval >= eval_interval and state['episode_count'] >= min_episodes_per_eval:
        # 计算成功率
        if state['episode_count'] > 0:
            success_rate = state['success_count'] / state['episode_count']
        else:
            success_rate = 0.0
        
        # 判断是否升级
        current_level = state['current_level']
        max_level = len(weight_levels) - 1
        
        # 如果当前不是最高难度，且达到了升级阈值
        if current_level < max_level and success_rate >= success_thresholds[current_level]:
            # 升级到下一档
            new_level = current_level + 1
            state['current_level'] = new_level
            state['current_weight'] = weight_levels[new_level]
            
            # 更新reward manager中的权重
            if hasattr(env, 'reward_manager'):
                try:
                    term_cfg = env.reward_manager.get_term_cfg(term_name)
                    old_weight = term_cfg.weight
                    term_cfg.weight = state['current_weight']
                    env.reward_manager.set_term_cfg(term_name, term_cfg)
                    
                    print(f"\n🎓 [Collision Curriculum] 惩罚权重升级！")
                    print(f"   当前RL iteration: {current_rl_iteration}")
                    print(f"   Level {current_level} → Level {new_level}")
                    print(f"   成功率: {success_rate:.2%} (阈值: {success_thresholds[current_level]:.2%})")
                    print(f"   权重变化: {old_weight} → {state['current_weight']}")
                except Exception as e:
                    print(f"⚠️ 更新碰撞惩罚权重失败: {e}")
        
        # 重置计数器
        state['success_count'] = 0
        state['episode_count'] = 0
        state['last_rl_iteration'] = current_rl_iteration
        
        # 返回状态
        return {
            "success_rate": success_rate,
            "current_level": float(state['current_level']),
            "current_weight": state['current_weight'],
            "iterations_since_eval": iterations_since_last_eval,
        }
    
    # 非评估iteration 或 样本不足
    return {
        "current_level": float(state['current_level']),
        "current_weight": state['current_weight'],
        "episodes_collected": float(state['episode_count']),
    }


def modify_command_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    attribute: str,
    curriculum_levels: list[tuple[float, float]],
    success_thresholds: list[float],
    eval_interval: int = 100,
    min_episodes_per_eval: int = 200,  # 最少样本量：避免过早升级
    warmup_iterations: int = 10,       # 预热：前若干iteration不评估
) -> dict[str, float]:
    """根据成功率渐进式调整命令采样范围。
    
    实现三档距离课程学习：
    - Level 0 (最简单): 达到 threshold[0] 的成功率后进入 Level 1
    - Level 1 (中等): 达到 threshold[1] 的成功率后进入 Level 2
    - Level 2 (困难): 最终目标
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        command_name: 命令名称（如 "pose_command"）
        attribute: 要修改的属性名（如 "pos_y"）
        curriculum_levels: 课程等级列表，每个元素为 (min, max) 元组
            例如: [(3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        success_thresholds: 成功率阈值列表，长度为 len(curriculum_levels) - 1
            例如: [0.70, 0.75] 表示70%进入Level1，75%进入Level2
        eval_interval: 评估间隔（每N个iteration评估一次）
        
    Returns:
        状态字典，用于日志记录
    """
    # 初始化状态
    if not hasattr(env, '_distance_curriculum_state'):
        env._distance_curriculum_state = {
            'last_rl_iteration': 0,  # 上次评估时的RL迭代数
            'success_count': 0,
            'episode_count': 0,
            'current_level': 0,  # 从第一档开始
            'current_range': curriculum_levels[0],
            'initialized_at_iteration': 0,  # 初始化的RL迭代数
        }
        print(f"🎓 [Distance Curriculum] 初始化完成，初始范围: {curriculum_levels[0]}")
    
    state = env._distance_curriculum_state
    
    # 统计成功率（每次环境重置时累积）
    if hasattr(env, 'reset_buf') and env.reset_buf is not None:
        terminated_envs = env.reset_buf.sum().item()
        state['episode_count'] += terminated_envs
        
        if terminated_envs > 0 and hasattr(env.termination_manager, 'get_term'):
            try:
                goal_reached = env.termination_manager.get_term('goal_reached')
                success_envs = (goal_reached & env.reset_buf).sum().item()
                state['success_count'] += success_envs
            except:
                pass
    
    # 🔧 使用RL训练迭代数（从环境属性获取）
    # 尝试从不同可能的属性获取RL迭代数
    current_rl_iteration = 0
    if hasattr(env, 'rl_iteration'):
        current_rl_iteration = env.rl_iteration
    elif hasattr(env, 'learning_iteration'):
        current_rl_iteration = env.learning_iteration
    elif hasattr(env, 'train_iteration'):
        current_rl_iteration = env.train_iteration
    else:
        # 如果找不到RL迭代数，回退到环境步数（除以decimation估算）
        decimation = getattr(env, 'decimation', 16)
        current_rl_iteration = env.common_step_counter // decimation
    
    iterations_since_last_eval = current_rl_iteration - state['last_rl_iteration']
    
    # 预热阶段：跳过评估，先累积数据
    if current_rl_iteration - state['initialized_at_iteration'] < warmup_iterations:
        return {
            "current_level": float(state['current_level']),
            "range_min": state['current_range'][0],
            "range_max": state['current_range'][1],
            "warmup": float(warmup_iterations - (current_rl_iteration - state['initialized_at_iteration'])),
        }

    # 每N个RL iterations评估一次，且保证样本量足够
    if iterations_since_last_eval >= eval_interval and state['episode_count'] >= min_episodes_per_eval:
        # 计算成功率
        if state['episode_count'] > 0:
            success_rate = state['success_count'] / state['episode_count']
        else:
            success_rate = 0.0
        
        # 判断是否升级
        current_level = state['current_level']
        max_level = len(curriculum_levels) - 1
        
        # 如果当前不是最高难度，且达到了升级阈值
        if current_level < max_level and success_rate >= success_thresholds[current_level]:
            # 升级到下一档
            new_level = current_level + 1
            state['current_level'] = new_level
            state['current_range'] = curriculum_levels[new_level]
            
            # 更新命令管理器中的范围
            if hasattr(env, 'command_manager'):
                cmd_term = env.command_manager.get_term(command_name)
                if cmd_term is not None and hasattr(cmd_term, 'cfg'):
                    # 动态修改配置中的范围
                    if hasattr(cmd_term.cfg.ranges, attribute):
                        setattr(cmd_term.cfg.ranges, attribute, state['current_range'])
                        print(f"\n🎓 [Distance Curriculum] 距离升级！")
                        print(f"   当前RL iteration: {current_rl_iteration}")
                        print(f"   Level {current_level} → Level {new_level}")
                        print(f"   成功率: {success_rate:.2%} (阈值: {success_thresholds[current_level]:.2%})")
                        print(f"   收集的episode数: {state['episode_count']}")
                        print(f"   新的 {attribute} 范围: {state['current_range']}")
        
        # 重置计数器
        state['success_count'] = 0
        state['episode_count'] = 0
        state['last_rl_iteration'] = current_rl_iteration  # 🔧 更新上次评估的RL迭代数
        
        # 返回状态
        return {
            "success_rate": success_rate,
            "current_level": float(state['current_level']),
            "range_min": state['current_range'][0],
            "range_max": state['current_range'][1],
            "iterations_since_eval": iterations_since_last_eval,  # 添加到日志
        }
    
    # 非评估iteration 或 样本不足
    return {
        "current_level": float(state['current_level']),
        "range_min": state['current_range'][0],
        "range_max": state['current_range'][1],
        "episodes_collected": float(state['episode_count']),
    }

