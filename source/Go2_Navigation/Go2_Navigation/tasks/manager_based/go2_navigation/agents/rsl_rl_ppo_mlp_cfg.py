# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO configuration for navigation with MLP-integrated obstacle encoder.

This configuration uses a custom ActorCritic network that integrates
the ObstacleMLP encoder for end-to-end training with proper gradient flow.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class NavigationEnvPPOMLPRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    PPO runner configuration for navigation with MLP obstacle encoder.

    Key features:
    - Custom ActorCriticWithLidarEncoder network
    - End-to-end training: LiDAR encoder + policy
    - Proper gradient flow through all components
    """

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "unitree_go2_navigation_mlp"
    empirical_normalization = False

    # Custom policy network with integrated LiDAR encoder
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ActorCriticWithLidarEncoder",
        init_noise_std=1.0,
        # noise_std_type="log",
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    # LiDAR encoder configuration (passed as kwargs to ActorCritic __init__)
    # 🆕 360-degree LiDAR matching real Unitree Go2 hardware
    lidar_input_dim: int = -1  # Auto-detect from environment (359 rays for 360° scan)
    lidar_output_dim: int = 36  # Encoded feature dimension (359 → 36, more expressive)
    lidar_hidden_dims: list = [256, 128, 64]  # Hidden layers for LiDAR encoder
    lidar_max_distance: float = 8.0

    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
