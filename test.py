import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import os
from PIL import Image, ImageDraw, ImageFont
import argparse
from train import PolicyNetwork, ValueNetwork, PPOAgent
from Utils.testUtils_PPO import *

def main():
    parser = argparse.ArgumentParser(description='测试PPO倒立摆智能体')
    parser.add_argument('--model', type=str, default='models/ppo_cartpole_model.pth',
                      help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=1,
                      help='测试回合数')
    parser.add_argument('--no-gif', action='store_true',
                      help='不保存gif文件')
    parser.add_argument('--comparison', action='store_true',
                      help='创建对比gif')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 找不到模型文件 {args.model}")
        print("请先运行 python train.py 训练模型")
        return
    
    # 创建智能体
    agent = PPOAgent(4, 2)  # CartPole的状态维度为4，动作维度为2
    
    print("=" * 50)
    print("PPO CartPole Agent Testing")
    print("=" * 50)
    
    # 测试智能体并保存gif
    test_rewards, test_lengths = test_agent_with_gif(
        agent, args.model, args.episodes, save_gif=not args.no_gif
    )
    
    # 显示测试结果统计
    print("\n" + "=" * 30)
    print("Test Results Statistics")
    print("=" * 30)
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Highest Reward: {np.max(test_rewards)}")
    print(f"Lowest Reward: {np.min(test_rewards)}")
    print(f"Average Episode Length: {np.mean(test_lengths):.2f}")
    print(f"Longest Episode: {np.max(test_lengths)}")
    
    # 可视化状态分析
    print("\nAnalyzing policy state changes...")
    visualize_policy_states(agent, args.model)
    
    # 创建对比gif
    if args.comparison:
        print("\nCreating comparison gif...")
        create_comparison_gif(args.model, min(args.episodes, 2))
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("Generated files:")
    print("- results/test_episode_1.gif: Test episode gif")
    print("- results/state_analysis_episode_1.png: State analysis chart")
    if args.comparison:
        print("- results/comparison.gif: Episode comparison gif")
    print("=" * 50)

if __name__ == "__main__":
    main()