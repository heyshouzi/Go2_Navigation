# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for navigation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.3,
) -> torch.Tensor:
    """Terminate when the robot reaches the goal position.

    This is a SUCCESS termination - robot successfully completed the task.

    Args:
        env: The environment.
        command_name: Name of the command term (e.g., "pose_command").
        distance_threshold: Distance threshold to consider goal reached (in meters).
            Default is 0.3m.

    Returns:
        Boolean tensor indicating which environments should terminate (True = goal reached).
    """
    # Get the command (target position)
    command = env.command_manager.get_command(command_name)

    # Extract position command in body frame (first 3 elements: x, y, z)
    target_pos_b = command[:, :3]

    # Calculate distance to goal
    distance = torch.norm(target_pos_b[:, :2], dim=1)  # Only use x, y (2D distance)

    # Return True if distance is less than threshold
    return distance < distance_threshold
