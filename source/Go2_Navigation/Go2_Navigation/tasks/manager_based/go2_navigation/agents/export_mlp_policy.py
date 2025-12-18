#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom exporter for ActorCriticWithLidarEncoder.

This exporter preserves the complete pipeline: lidar_encoder + actor,
ensuring proper sim2real transfer.
"""

import torch
import torch.nn as nn
import copy
import os


class PolicyWithLidarEncoder(nn.Module):
    """
    Standalone policy module with integrated lidar encoder.

    This module wraps the ActorCriticWithLidarEncoder for deployment,
    ensuring the lidar encoding is part of the exported model.
    """

    def __init__(self, policy, normalizer=None):
        """
        Initialize deployment policy.

        Args:
            policy: ActorCriticWithLidarEncoder instance
            normalizer: Optional normalizer (usually None for this architecture)
        """
        super().__init__()

        # Copy lidar encoder
        if hasattr(policy, "lidar_encoder"):
            self.lidar_encoder = copy.deepcopy(policy.lidar_encoder)
            print(
                f"✅ Copied LiDAR encoder: {policy.num_lidar_obs} → {self.lidar_encoder.output_dim}"
            )
        else:
            raise ValueError("Policy does not have lidar_encoder!")

        # Copy actor
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            print(f"✅ Copied Actor MLP")
        else:
            raise ValueError("Policy does not have actor!")

        # Store dimensions
        self.num_basic_obs = policy.num_basic_obs  # 10 (pose + vel)
        self.num_lidar_obs = policy.num_lidar_obs  # 359

        # Copy normalizer (usually Identity for this architecture)
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = nn.Identity()

        print(f"📊 Policy structure:")
        print(f"   Input: {self.num_basic_obs + self.num_lidar_obs} dims")
        print(f"   Basic obs: {self.num_basic_obs} dims")
        print(f"   LiDAR raw: {self.num_lidar_obs} dims")
        print(f"   LiDAR encoded: {self.lidar_encoder.output_dim} dims")
        print(
            f"   Actor input: {self.num_basic_obs + self.lidar_encoder.output_dim} dims"
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with lidar encoding.

        Args:
            obs: Observation tensor (batch, 370)
                 Structure: [pose(4), lin_vel(3), ang_vel(3), lidar(359)]

        Returns:
            Action tensor (batch, 3): [vx, vy, vyaw]
        """
        # Split observations
        basic_obs = obs[:, : self.num_basic_obs]  # First 10 dims
        lidar_obs = obs[:, self.num_basic_obs :]  # Remaining 359 dims

        # Encode lidar
        encoded_lidar = self.lidar_encoder(lidar_obs)

        # Concatenate
        processed_obs = torch.cat([basic_obs, encoded_lidar], dim=-1)

        # Normalize (usually Identity)
        processed_obs = self.normalizer(processed_obs)

        # Get action from actor
        action = self.actor(processed_obs)

        return action


def export_mlp_policy_as_jit(
    policy, normalizer, path: str, filename: str = "policy.pt"
):
    """
    Export ActorCriticWithLidarEncoder as JIT with complete preprocessing.

    Args:
        policy: ActorCriticWithLidarEncoder instance
        normalizer: Optional normalizer
        path: Export directory
        filename: Export filename
    """
    print("=" * 60)
    print("🚀 Exporting MLP Navigation Policy")
    print("=" * 60)

    # Create deployment wrapper
    deployment_policy = PolicyWithLidarEncoder(policy, normalizer)
    deployment_policy.to("cpu")
    deployment_policy.eval()

    # Script the model
    print("\n📦 Converting to TorchScript...")
    scripted_policy = torch.jit.script(deployment_policy)

    # Save
    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)
    scripted_policy.save(export_path)

    print(f"✅ Policy exported to: {export_path}")

    # Test the exported model
    print("\n🧪 Testing exported model...")
    test_obs = torch.randn(
        1, deployment_policy.num_basic_obs + deployment_policy.num_lidar_obs
    )

    # Test original
    with torch.no_grad():
        test_action_original = deployment_policy(test_obs)

    # Test loaded
    loaded_policy = torch.jit.load(export_path)
    with torch.no_grad():
        test_action_loaded = loaded_policy(test_obs)

    # Compare
    diff = torch.abs(test_action_original - test_action_loaded).max().item()
    print(f"   Max difference: {diff:.2e}")

    if diff < 1e-6:
        print("✅ Export test passed!")
    else:
        print(f"⚠️  Warning: Export test failed (diff={diff})")

    print("=" * 60)

    return export_path


def export_policy_from_checkpoint(
    checkpoint_path: str, export_dir: str, filename: str = "policy.pt"
):
    """
    Load a checkpoint and export the policy.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        export_dir: Export directory
        filename: Export filename

    Example:
        export_policy_from_checkpoint(
            "/home/wu/IsaacLab/logs/rsl_rl/unitree_go2_navigation_mlp/2025-10-15_16-57-32/model_3000.pt",
            "/home/wu/IsaacLab/logs/rsl_rl/unitree_go2_navigation_mlp/2025-10-15_16-57-32/exported"
        )
    """
    print(f"📦 Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract policy state dict
    if "model_state_dict" in checkpoint:
        policy_state_dict = checkpoint["model_state_dict"]
    elif "ac_state_dict" in checkpoint:
        policy_state_dict = checkpoint["ac_state_dict"]
    else:
        raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")

    # Need to reconstruct the policy architecture
    print("⚠️  Note: This function requires the policy architecture to be imported.")
    print("    Please use the RSL-RL runner's export function instead:")
    print(
        "    runner.save(os.path.join(runner.log_dir, f'model_{runner.current_learning_iteration}.pt'))"
    )
    print("    runner.export_policy_as_jit(...)")

    raise NotImplementedError(
        "Direct checkpoint export not implemented. "
        "Use the training script's export function instead."
    )


if __name__ == "__main__":
    """
    Standalone export script.

    Usage from training script:
        from isaaclab_tasks.manager_based.navigation.config.go2.export_mlp_policy import export_mlp_policy_as_jit

        # After training
        export_mlp_policy_as_jit(
            policy=agent.policy,  # ActorCriticWithLidarEncoder
            normalizer=agent.actor_critic.actor_obs_normalizer if hasattr(agent.actor_critic, 'actor_obs_normalizer') else None,
            path=os.path.join(log_dir, "exported"),
            filename="policy.pt"
        )
    """
    print(__doc__)
