import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

from Utils.netUtils_PPO import *
from Utils.trainUtils_PPO import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    # 定义PPOagent参数
    ppo_config = {
        "gamma": 0.99,  # 折扣因子
        "K_epochs": 20,  # PPO更新次数
        "eps_clip": 0.2,  # 剪辑因子
        "lr": 0.0003,  # 学习率
    }
    # 定义训练参数
    train_config = {
        "max_episodes": 500,
        "max_timesteps": 5000,
        "update_timestep": 1000,
    }
    # 训练智能体
    trained_agent, episode_rewards, policy_losses, value_losses = train_ppo(PPO_config = ppo_config, Train_config = train_config)
    
    # 保存模型
    trained_agent.save_model('models/ppo_cartpole_model.pth')
    
    # 保存训练数据
    save_training_data(episode_rewards, policy_losses, value_losses)
    
    # # 可视化训练结果
    # print("\n显示训练结果...")
    # visualize_training_results(episode_rewards, policy_losses, value_losses)
    
    print(f"\n训练完成!")
    print(f"最终100回合平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"模型已保存到: models/ppo_cartpole_model.pth")
    print(f"运行 python test.py 来测试模型并生成gif")