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
        else:
            raise ValueError("Policy does not have lidar_encoder!")

        # Copy actor
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            print(f"✅ Copied Actor MLP")
        else:
            raise ValueError("Policy does not have actor!")

        # Store dimensions - 使用新的属性名称
        # 先打印所有可用属性以便调试
        try:
            policy_attrs = [attr for attr in dir(policy) if not attr.startswith('_')]
            print(f"🔍 Debug: Policy type: {type(policy)}")
            print(f"🔍 Debug: Policy attributes (first 20): {policy_attrs[:20]}")
        except Exception as e:
            print(f"⚠️  Warning: Could not list policy attributes: {e}")
        
        # 检查并获取基本观察维度
        try:
            if hasattr(policy, "actor_num_basic_obs"):
                self.num_basic_obs = getattr(policy, "actor_num_basic_obs")
                print(f"✅ Found actor_num_basic_obs: {self.num_basic_obs}")
            elif hasattr(policy, "num_basic_obs"):
                # 向后兼容旧版本
                self.num_basic_obs = getattr(policy, "num_basic_obs")
                print(f"✅ Found num_basic_obs (legacy): {self.num_basic_obs}")
            else:
                available_attrs = [attr for attr in dir(policy) if 'basic' in attr.lower() or 'obs' in attr.lower()]
                raise ValueError(
                    f"Policy does not have actor_num_basic_obs or num_basic_obs attribute!\n"
                    f"Available observation-related attributes: {available_attrs}"
                )
        except AttributeError as e:
            raise AttributeError(
                f"Failed to get basic observation dimension from policy: {e}\n"
                f"Policy type: {type(policy)}\n"
                f"Policy attributes: {[attr for attr in dir(policy) if not attr.startswith('_')]}"
            ) from e

        # 检查并获取 LiDAR 观察维度
        try:
            if hasattr(policy, "actor_num_lidar_obs"):
                self.num_lidar_obs = getattr(policy, "actor_num_lidar_obs")
                print(f"✅ Found actor_num_lidar_obs: {self.num_lidar_obs}")
            elif hasattr(policy, "num_lidar_obs"):
                # 向后兼容旧版本
                self.num_lidar_obs = getattr(policy, "num_lidar_obs")
                print(f"✅ Found num_lidar_obs (legacy): {self.num_lidar_obs}")
            else:
                available_attrs = [attr for attr in dir(policy) if 'lidar' in attr.lower() or 'obs' in attr.lower()]
                all_attrs = [attr for attr in dir(policy) if not attr.startswith('_')]
                raise AttributeError(
                    f"Policy does not have actor_num_lidar_obs or num_lidar_obs attribute!\n"
                    f"Available lidar/observation-related attributes: {available_attrs}\n"
                    f"All policy attributes: {all_attrs}"
                )
        except AttributeError as e:
            raise AttributeError(
                f"Failed to get LiDAR observation dimension from policy: {e}\n"
                f"Policy type: {type(policy)}\n"
                f"Policy has actor_num_basic_obs: {hasattr(policy, 'actor_num_basic_obs')}\n"
                f"Policy has actor_num_lidar_obs: {hasattr(policy, 'actor_num_lidar_obs')}\n"
                f"Policy attributes: {[attr for attr in dir(policy) if not attr.startswith('_')]}"
            ) from e

        print(
            f"✅ Copied LiDAR encoder: {self.num_lidar_obs} → {self.lidar_encoder.output_dim}"
        )

        # Copy normalizer (usually Identity for this architecture)
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            # 使用 policy 的 normalizer 如果存在
            if hasattr(policy, "actor_obs_normalizer"):
                self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
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
            obs: Observation tensor (batch, total_obs_dim)
                 Structure: [basic_obs(num_basic_obs), lidar_obs(num_lidar_obs)]
                 Note: obstacle_features (LiDAR) 位于最后 num_lidar_obs 维

        Returns:
            Action tensor (batch, 3): [vx, vy, vyaw]
        """
        # Split observations
        # obstacle_features 位于最后 num_lidar_obs 维（与 ActorCriticWithLidarEncoder 一致）
        basic_obs = obs[:, : -self.num_lidar_obs]  # 除了最后 num_lidar_obs 维之外的所有观察
        lidar_obs = obs[:, -self.num_lidar_obs:]  # 最后 num_lidar_obs 维（LiDAR）

        # Encode lidar using shared encoder
        encoded_lidar = self.lidar_encoder(lidar_obs)

        # Concatenate: basic obs + encoded LiDAR
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
        filename: Export filename (can be .pt or .pth)
    """
    print("=" * 60)
    print("🚀 Exporting MLP Navigation Policy (with LiDAR encoder)")
    print("=" * 60)

    # Create deployment wrapper
    deployment_policy = PolicyWithLidarEncoder(policy, normalizer)
    deployment_policy.to("cpu")
    deployment_policy.eval()

    # Script the model
    print("\n📦 Converting to TorchScript...")
    try:
        scripted_policy = torch.jit.script(deployment_policy)
        print("✅ Successfully scripted policy")
    except Exception as e:
        print(f"⚠️  Failed to script policy: {e}")
        print("   Trying trace method instead...")
        # Fallback: use trace method with example input
        example_obs = torch.randn(1, deployment_policy.num_basic_obs + deployment_policy.num_lidar_obs)
        scripted_policy = torch.jit.trace(deployment_policy, example_obs)
        print("✅ Successfully traced policy")

    # Save
    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)
    scripted_policy.save(export_path)

    print(f"✅ Policy exported to: {export_path}")
    
    # Also save as .pth format (full model, not just state_dict)
    base_name = os.path.splitext(filename)[0]
    pth_path = os.path.join(path, f"{base_name}.pt")
    torch.save(deployment_policy, pth_path)
    print(f"✅ Policy also saved as .pth to: {pth_path}")






    # Test the exported model (optional, wrapped in try-except)
    print("\n🧪 Testing exported model...")
    try:
        # Check if file exists and is readable
        if not os.path.exists(export_path):
            raise FileNotFoundError(f"Exported file not found: {export_path}")
        
        # Wait a bit to ensure file is fully written (for file system sync)
        import time
        time.sleep(0.1)
        
        # Check file size
        file_size = os.path.getsize(export_path)
        if file_size == 0:
            raise ValueError(f"Exported file is empty: {export_path}")
        print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
        
        test_obs = torch.randn(
            1, deployment_policy.num_basic_obs + deployment_policy.num_lidar_obs
        )

        # Test original
        with torch.no_grad():
            test_action_original = deployment_policy(test_obs)

        # Test loaded TorchScript model
        try:
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
        except Exception as load_error:
            print(f"⚠️  Warning: Could not load TorchScript model for testing: {load_error}")
            print("   This may be due to TorchScript compatibility issues, but the model file was saved successfully.")
            print("   You can try loading it later or use the .pth format instead.")
            
    except Exception as test_error:
        print(f"⚠️  Warning: Export test failed: {test_error}")
        print("   However, the model file was saved successfully.")
        print("   You can verify the export by loading it manually.")

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
