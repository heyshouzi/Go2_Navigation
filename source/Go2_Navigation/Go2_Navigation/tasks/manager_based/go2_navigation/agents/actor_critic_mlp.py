# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom Actor-Critic network with integrated MLP obstacle encoder.

This module defines a custom ActorCritic class that integrates an MLP encoder
for processing raw LiDAR data. The encoder is part of the policy network,
ensuring proper gradient flow during end-to-end training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ObstacleMLP(nn.Module):
    """
    MLP encoder for obstacle perception.
    
    This encoder processes raw lidar ranges and outputs compact features.
    It's integrated into the ActorCritic network for end-to-end training.
    """
    
    def __init__(
        self,
        input_dim: int = 359,
        hidden_dims: list[int] = [256, 128, 64],
        output_dim: int = 36,
        max_distance: float = 8.0,
        dropout: float = 0.0,  # No dropout during RL training by default
    ):
        """
        Initialize the MLP encoder.
        
        Args:
            input_dim: Number of input lidar points (359 for 360° scan, excluding overlap at 0°=360°)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            max_distance: Maximum lidar range for normalization
            dropout: Dropout probability (usually 0 for RL)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_distance = max_distance
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 and i < len(hidden_dims) - 1 else nn.Identity(),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, lidar_ranges: torch.Tensor) -> torch.Tensor:
        """
        Encode lidar ranges to compact features.
        
        Args:
            lidar_ranges: Raw lidar ranges, shape (batch, num_rays)
        
        Returns:
            Encoded features, shape (batch, output_dim)
        """
        # Handle inf values (no hit)
        ranges = torch.where(
            torch.isinf(lidar_ranges),
            torch.full_like(lidar_ranges, self.max_distance),
            lidar_ranges
        )
        
        # Clip and normalize to [0, 1]
        ranges = torch.clamp(ranges, 0.0, self.max_distance)
        normalized = ranges / self.max_distance
        
        # Encode through MLP
        features = self.encoder(normalized)
        
        return features


class ActorCriticWithLidarEncoder(nn.Module):
    """
    Actor-Critic network with integrated 360° LiDAR MLP encoder.
    
    🆕 Designed for Unitree Go2 with 360-degree LiDAR (Unitree L1)
    
    This network expects observations with the following structure:
    - pose_command: (batch, 4)
    - base_lin_vel: (batch, 3)
    - base_ang_vel: (batch, 3)
    - obstacle_features: (batch, 359)  ← Raw 360° LiDAR data (359 rays)!
    
    The raw LiDAR data is encoded by an integrated ObstacleMLP before
    being concatenated with other observations.
    
    Architecture:
        Input → [Split] → Basic features (10)
                       → LiDAR raw (359) → ObstacleMLP → Encoded (36)
              → [Concat] → Combined (46)
              → Actor MLP → Actions (3)
              → Critic MLP → Value (1)
    
    ✅ Sim2Real: Identical sensor in simulation and real robot!
    Note: 359 rays cover 360° (0° to 358°), excluding 359° which equals 0°.
    """
    
    is_recurrent = False
    
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # LiDAR encoder config
        lidar_input_dim: int = 359,
        lidar_output_dim: int = 36,
        lidar_hidden_dims: list[int] = [256, 128, 64],
        lidar_max_distance: float = 8.0,
        # Standard ActorCritic config
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        """
        Initialize the ActorCritic with integrated LiDAR encoder.
        
        Args:
            obs: Observation samples from environment
            obs_groups: Observation group configuration
            num_actions: Number of action dimensions
            lidar_input_dim: Number of raw lidar rays
            lidar_output_dim: Encoded lidar feature dimension
            lidar_hidden_dims: Hidden layers for lidar encoder
            lidar_max_distance: Max lidar distance for normalization
            actor_obs_normalization: Whether to normalize actor observations
            critic_obs_normalization: Whether to normalize critic observations
            actor_hidden_dims: Hidden layers for actor
            critic_hidden_dims: Hidden layers for critic
            activation: Activation function
            init_noise_std: Initial action noise std
            noise_std_type: Type of noise std ('scalar' or 'log')
            **kwargs: Additional arguments (ignored)
        """
        if kwargs:
            print(
                "ActorCriticWithLidarEncoder.__init__ got unexpected arguments: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.obs_groups = obs_groups
        
        # Check if there's a separate critic observation group
        has_critic_group = "critic" in obs_groups and len(obs_groups["critic"]) > 0
        
        # Calculate Actor observation dimensions
        actor_obs_dim = 0
        for obs_group in obs_groups["policy"]:
            obs_shape = obs[obs_group].shape
            assert len(obs_shape) == 2, "Only 1D observations supported."
            actor_obs_dim += obs_shape[-1]
        
        # Calculate Critic observation dimensions
        if has_critic_group:
            critic_obs_dim = 0
            for obs_group in obs_groups["critic"]:
                obs_shape = obs[obs_group].shape
                assert len(obs_shape) == 2, "Only 1D observations supported."
                critic_obs_dim += obs_shape[-1]
        else:
            critic_obs_dim = actor_obs_dim  # Fallback to actor observations if no critic group
        
        # Parse observation structure
        # Actor: pose(4) + ang_vel(3) + lidar(N) = 7 + lidar
        # Critic: pose(4) + lin_vel(3) + ang_vel(3) + lidar(N) = 10 + lidar
        actor_num_basic_obs = 7  # 4 (pose) + 3 (ang_vel)
        actor_num_lidar_obs = actor_obs_dim - actor_num_basic_obs
        
        if has_critic_group:
            critic_num_basic_obs = 10  # 4 (pose) + 3 (lin_vel) + 3 (ang_vel)
            critic_num_lidar_obs = critic_obs_dim - critic_num_basic_obs
        else:
            critic_num_basic_obs = actor_num_basic_obs
            critic_num_lidar_obs = actor_num_lidar_obs
        
        print(f"📊 Actor observation structure: {actor_obs_dim} total dims")
        print(f"   - Basic observations: {actor_num_basic_obs} dims (pose + ang_vel)")
        print(f"   - LiDAR rays: {actor_num_lidar_obs} dims")
        
        if has_critic_group:
            print(f"📊 Critic observation structure: {critic_obs_dim} total dims")
            print(f"   - Basic observations: {critic_num_basic_obs} dims (pose + lin_vel + ang_vel)")
            print(f"   - LiDAR rays: {critic_num_lidar_obs} dims")
        
        # Validate LiDAR detection
        if actor_num_lidar_obs == 0:
            raise ValueError(
                "No LiDAR observation detected in actor observations! "
                f"Got observations: {obs_groups['policy']}"
            )
        
        if actor_num_lidar_obs != lidar_input_dim:
            print(f"⚠️  Warning: Expected {lidar_input_dim} LiDAR rays, got {actor_num_lidar_obs}")
            lidar_input_dim = actor_num_lidar_obs
        
        self.has_critic_group = has_critic_group
        self.actor_num_basic_obs = actor_num_basic_obs
        self.actor_num_lidar_obs = actor_num_lidar_obs
        self.critic_num_basic_obs = critic_num_basic_obs
        self.critic_num_lidar_obs = critic_num_lidar_obs
        
        # === LiDAR Encoder ===
        self.lidar_encoder = ObstacleMLP(
            input_dim=lidar_input_dim,
            hidden_dims=lidar_hidden_dims,
            output_dim=lidar_output_dim,
            max_distance=lidar_max_distance,
            dropout=0.0,
        )
        
        print(f"✅ LiDAR Encoder: {lidar_input_dim} → {lidar_output_dim}")
        print(f"   Parameters: {sum(p.numel() for p in self.lidar_encoder.parameters())}")
        
        # === Actor Network ===
        # Actor input: basic observations + encoded LiDAR features
        num_actor_obs = actor_num_basic_obs + lidar_output_dim
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        
        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        
        print(f"✅ Actor MLP: {num_actor_obs} → {num_actions}")
        print(f"   (Basic: {actor_num_basic_obs} + LiDAR: {lidar_output_dim} = {num_actor_obs})")
        
        # === Critic Network ===
        # Critic uses different observations if critic group is available
        num_critic_obs = critic_num_basic_obs + lidar_output_dim
        
        print(f"🔧 DEBUG: Creating Critic with input dim = {num_critic_obs}")
        print(f"🔧 DEBUG: Critic hidden dims = {critic_hidden_dims}")
        
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        
        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        
        print(f"✅ Critic MLP: {num_critic_obs} → 1")
        if has_critic_group:
            print(f"   (Basic: {critic_num_basic_obs} + LiDAR: {lidar_output_dim} = {num_critic_obs})")
        
        # === Action Noise ===
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
        
        print(f"🎯 Total trainable parameters: {sum(p.numel() for p in self.parameters())}")
    
    def reset(self, dones=None):
        """Reset recurrent states (not used in this non-recurrent network)."""
        pass
    
    def forward(self):
        """Forward pass (not implemented, use act() or evaluate())."""
        raise NotImplementedError
    
    def update_normalization(self, observations):
        """Update observation normalization statistics."""
        # Process observations (encode LiDAR)
        processed_obs_actor = self._process_observations(observations, is_actor=True)
        processed_obs_critic = self._process_observations(observations, is_actor=False)
        
        # Update normalizers if enabled
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(processed_obs_actor)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(processed_obs_critic)
    
    def _process_observations(self, observations, is_actor: bool = True):
        """
        Process observations: encode LiDAR and concatenate with basic obs.
        
        Args:
            observations: Dict of observations (with 'policy' and optionally 'critic' keys)
            is_actor: Whether processing for actor (vs critic)
        
        Returns:
            Concatenated observation tensor with LiDAR encoded
        """
        if is_actor:
            # Actor uses 'policy' group
            obs_tensor = observations["policy"]
            basic_obs = obs_tensor[:, :self.actor_num_basic_obs]  # First 7 dims (pose + ang_vel)
            lidar_obs = obs_tensor[:, self.actor_num_basic_obs:]  # Remaining dims (LiDAR)
        else:
            # Critic uses 'critic' group if available, otherwise 'policy'
            if self.has_critic_group and "critic" in observations:
                obs_tensor = observations["critic"]
                basic_obs = obs_tensor[:, :self.critic_num_basic_obs]  # First 10 dims (pose + lin_vel + ang_vel)
                lidar_obs = obs_tensor[:, self.critic_num_basic_obs:]  # Remaining dims (LiDAR)
            else:
                # Fallback to policy observations
                obs_tensor = observations["policy"]
                basic_obs = obs_tensor[:, :self.actor_num_basic_obs]
                lidar_obs = obs_tensor[:, self.actor_num_basic_obs:]
        
        # Encode LiDAR
        encoded_lidar = self.lidar_encoder(lidar_obs)
        
        # Concatenate: basic obs + encoded LiDAR
        combined_obs = torch.cat([basic_obs, encoded_lidar], dim=-1)
        
        return combined_obs
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    @property
    def std(self):
        if self.noise_std_type == "log":
            # Use exp to get std from log_std, and clamp to ensure positive values
            # Also handle potential inf/nan values
            std = torch.exp(self.log_std)
            return torch.clamp(std, min=1e-6, max=1e6)
        else:
            # Ensure std is always non-negative (clamp to prevent negative values)
            return torch.clamp(self._std, min=1e-6)
    
    @std.setter
    def std(self, std):
        if self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(std))
        else:
            self._std = nn.Parameter(std)
    
    def update_distribution(self, observations):
        """Update the action distribution based on current observations."""
        # Process observations (encode LiDAR)
        processed_obs = self._process_observations(observations, is_actor=True)
        
        # Normalize if needed
        processed_obs = self.actor_obs_normalizer(processed_obs)
        
        # Get action mean from actor
        mean = self.actor(processed_obs)
        
        # Create distribution
        self.distribution = Normal(mean, mean * 0.0 + self.std)
    
    def act(self, observations, **kwargs):
        """Sample actions from the policy."""
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Deterministic action for inference (no noise)."""
        # Process observations
        processed_obs = self._process_observations(observations, is_actor=True)
        processed_obs = self.actor_obs_normalizer(processed_obs)
        
        # Return mean action (no noise)
        actions_mean = self.actor(processed_obs)
        return actions_mean
    
    def evaluate(self, observations, **kwargs):
        """Evaluate state value using critic."""
        # Process observations (encode LiDAR)
        processed_obs = self._process_observations(observations, is_actor=False)
        
        # Normalize if needed
        processed_obs = self.critic_obs_normalizer(processed_obs)
        
        # Get value estimate
        value = self.critic(processed_obs)
        
        return value

