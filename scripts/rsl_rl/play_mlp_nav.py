#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Play script for navigation with MLP-integrated obstacle encoder.

This script registers the custom ActorCriticWithLidarEncoder class
with rsl_rl before loading the checkpoint, ensuring proper model loading.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_mlp_nav.py \
        --task Isaac-Navigation-Flat-Unitree-Go2-MLP-Play-v0 \
        --num_envs 16
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play navigation with MLP encoder (RSL-RL)")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows (after AppLauncher initialization)"""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 🔥 CRITICAL: Register custom ActorCritic class with rsl_rl
from Go2_Navigation.tasks.manager_based.go2_navigation.agents.actor_critic_mlp import ActorCriticWithLidarEncoder
import rsl_rl.modules
# Register the custom class
setattr(rsl_rl.modules, 'ActorCriticWithLidarEncoder', ActorCriticWithLidarEncoder)
print("✅ Registered ActorCriticWithLidarEncoder with rsl_rl.modules")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using custom MLP encoder."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # 🆕 Auto-load the latest model if no specific checkpoint is provided
        print(f"[INFO] Looking for latest model in: {log_root_path}")
        import glob
        model_files = glob.glob(os.path.join(log_root_path, "**/model_*.pt"), recursive=True)
        if model_files:
            # Sort by modification time and get the latest
            latest_model = max(model_files, key=os.path.getmtime)
            resume_path = latest_model
            print(f"[INFO] Auto-selected latest model: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        try:
            policy_nn = runner.alg.actor_critic
        except AttributeError:
            # Fallback: try to get from algorithm directly
            policy_nn = runner.alg

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit with MLP encoder
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    
    print(f"[INFO] Exporting models to: {export_model_dir}")
    
    # 🆕 Use custom export function for MLP encoder (with LiDAR perception)
    from isaaclab_tasks.manager_based.navigation.config.go2.export_mlp_policy import export_mlp_policy_as_jit
    
    print("[INFO] Exporting policy with integrated LiDAR encoder...")
    try:
        export_mlp_policy_as_jit(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy_with_encoder.pt"
        )
        print("✅ Successfully exported policy with LiDAR encoder!")
    except Exception as e:
        print(f"⚠️  Failed to export with encoder: {e}")
        print("   Falling back to standard export...")
    
    # Also export standard versions for compatibility
    print("[INFO] Exporting standard policy versions...")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    
    print(f"[INFO] All models exported to: {export_model_dir}")
    print("   - policy_with_encoder.pt (with LiDAR perception)")
    print("   - policy.pt (standard)")
    print("   - policy.onnx (ONNX format)")

    dt = env.unwrapped.step_dt

    # 🆕 路径可视化：记录和可视化go2走过的路线
    from collections import defaultdict
    from pxr import UsdGeom, Gf, Usd
    import omni.usd
    
    # 存储每个环境的路径点
    path_points = defaultdict(list)  # {env_idx: [(x, y, z), ...]}
    
    # 获取USD stage
    stage = omni.usd.get_context().get_stage()
    
    def is_prim_path_valid(prim_path: str) -> bool:
        """检查prim路径是否有效"""
        prim = stage.GetPrimAtPath(prim_path)
        return prim.IsValid()
    
    def delete_prim(prim_path: str):
        """删除USD prim"""
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            stage.RemovePrim(prim_path)
    
    def update_path_visualization(env_idx: int, points: list):
        """更新路径可视化 - 使用线条段绘制（更可靠）"""
        if len(points) < 2:
            return
        
        # 使用线条段（Line segments）绘制路径，每两个点之间一条线
        parent_path = f"/World/PathVisualization/env_{env_idx}"
        
        # 如果父路径不存在，创建它
        if not is_prim_path_valid(parent_path):
            parent_prim = stage.DefinePrim(parent_path, "Xform")
        else:
            parent_prim = stage.GetPrimAtPath(parent_path)
        
        # 删除旧的线条段
        for child in parent_prim.GetChildren():
            if child.GetName().startswith("line_"):
                stage.RemovePrim(child.GetPath())
        
        # 创建新的线条段（每两个点之间一条线）
        for i in range(len(points) - 1):
            line_path = f"{parent_path}/line_{i}"
            line_prim = UsdGeom.BasisCurves.Define(stage, line_path)
            
            # 设置两个端点
            p1 = Gf.Vec3f(float(points[i][0]), float(points[i][1]), float(points[i][2]))
            p2 = Gf.Vec3f(float(points[i+1][0]), float(points[i+1][1]), float(points[i+1][2]))
            
            # 设置曲线点
            line_prim.GetPointsAttr().Set([p1, p2])
            
            # 设置曲线类型
            line_prim.GetBasisAttr().Set(UsdGeom.Tokens.linear)
            line_prim.GetTypeAttr().Set(UsdGeom.Tokens.linear)
            
            # 设置曲线顶点数（一条线有2个点）
            line_prim.GetCurveVertexCountsAttr().Set([2])
            
            # 设置颜色为绿色
            color_attr = line_prim.CreateDisplayColorAttr()
            color_attr.Set([Gf.Vec3f(0.0, 1.0, 0.0)])
            
            # 设置宽度（增加宽度以便更容易看到）
            widths_attr = line_prim.CreateWidthsAttr()
            widths_attr.Set([0.05])
    
    # 获取机器人资产
    robot = env.unwrapped.scene["robot"]
    
    # reset environment
    obs = env.get_observations()
    timestep = 0
    last_update_step = 0
    update_interval = 5  # 每5步更新一次路径可视化
    last_reset_buf = None  # 用于检测环境重置
    path_record_interval = 2  # 每2步记录一次路径点（减少点数量，提高性能）
    last_path_record_step = 0
    
    print("[INFO] 🎨 已启用路径可视化功能，将显示go2走过的路线（绿色线条）")
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # 检测环境重置并清除对应环境的路径
            if hasattr(env.unwrapped, 'episode_length_buf'):
                reset_buf = env.unwrapped.episode_length_buf
                if last_reset_buf is not None:
                    # 检测哪些环境被重置了（episode_length_buf从非0变为0或1）
                    reset_mask = (reset_buf <= 1) & (last_reset_buf > 1)
                    if reset_mask.any():
                        for env_idx in range(reset_mask.shape[0]):
                            if reset_mask[env_idx]:
                                # 清除该环境的路径点
                                if env_idx in path_points:
                                    path_points[env_idx] = []
                                # 清除可视化
                                prim_path = f"/World/PathVisualization/env_{env_idx}"
                                if is_prim_path_valid(prim_path):
                                    delete_prim(prim_path)
                last_reset_buf = reset_buf.clone()
            
            # 记录机器人位置（定期记录以减少点数量，提高性能）
            if robot.is_initialized and (timestep - last_path_record_step >= path_record_interval):
                robot_positions = robot.data.root_pos_w.clone()  # (num_envs, 3)
                # 为每个环境记录路径点
                for env_idx in range(robot_positions.shape[0]):
                    pos = robot_positions[env_idx].cpu().numpy()
                    path_points[env_idx].append((float(pos[0]), float(pos[1]), float(pos[2])))
                last_path_record_step = timestep
            
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # 定期更新路径可视化
            if timestep - last_update_step >= update_interval:
                for env_idx, points in path_points.items():
                    if len(points) > 1:
                        update_path_visualization(env_idx, points)
                last_update_step = timestep
            
            # 🔧 修复：timestep必须在循环中更新，不依赖于video标志
            timestep += 1
            
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    
    # 最终更新一次路径可视化
    print(f"[INFO] 路径可视化完成，共记录了 {len(path_points)} 个环境的路径")
    for env_idx, points in path_points.items():
        if len(points) > 1:
            update_path_visualization(env_idx, points)
            print(f"[INFO] 环境 {env_idx}: 记录了 {len(points)} 个路径点")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

