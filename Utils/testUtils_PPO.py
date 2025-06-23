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

def add_text_to_frame(frame, text, position=(10, 10), font_size=16, color=(255, 255, 255)):
    """在帧上添加文字"""
    # 将numpy数组转换为PIL图像
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # 尝试使用默认字体
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # 添加文字
    draw.text(position, text, fill=color, font=font)
    
    # 转换回numpy数组
    return np.array(img)

def test_agent_with_gif(agent, model_path, num_episodes=1, save_gif=True):
    """测试智能体并保存为gif"""
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # 加载模型
    agent.load_model(model_path)
    
    test_rewards = []
    test_episode_lengths = []
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        frames = []
        
        print(f"\n开始测试回合 {episode + 1}...")
        
        while True:
            # 渲染环境
            frame = env.render()
            
            # 在帧上添加信息
            info_text = f"Episode: {episode + 1}, Step: {episode_length + 1}, Reward: {episode_reward}"
            frame_with_text = add_text_to_frame(frame, info_text)
            frames.append(frame_with_text)
            
            # 选择动作
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                # 添加结束信息
                final_text = f"Episode {episode + 1} Complete! Final Reward: {episode_reward}, Length: {episode_length}"
                final_frame = add_text_to_frame(frame, final_text)
                # 添加几帧结束画面
                for _ in range(30):  # 约1秒的结束画面
                    frames.append(final_frame)
                break
        
        test_rewards.append(episode_reward)
        test_episode_lengths.append(episode_length)
        
        print(f"回合 {episode + 1} 完成: 奖励 = {episode_reward}, 长度 = {episode_length}")
        
        # 保存gif
        if save_gif and frames:
            gif_path = f'results/test_episode_{episode + 1}.gif'
            print(f"正在保存gif到 {gif_path}...")
            
            # 减少帧数以减小文件大小
            frames_reduced = frames[::2]  # 每隔一帧取一帧
            
            imageio.mimsave(gif_path, frames_reduced, fps=30, loop=0)
            print(f"Gif已保存: {gif_path}")
    
    env.close()
    return test_rewards, test_episode_lengths

def visualize_policy_states(agent, model_path, num_episodes=1):
    """可视化策略执行过程的状态变化"""
    env = gym.make('CartPole-v1', render_mode=None)
    
    # 加载模型
    agent.load_model(model_path)
    
    # 只分析第一个回合
    episode = 0
    state, _ = env.reset()
    states = [state]
    actions = []
    rewards = []
    action_probs_history = []
    
    print(f"\nAnalyzing state changes for episode {episode + 1}...")
    
    while True:
        # 获取动作概率
        if isinstance(state, tuple):
            state_tensor = torch.FloatTensor(np.array(state[0])).unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
            
        with torch.no_grad():
            action_probs = agent.policy_old(state_tensor)
            action_probs_history.append(action_probs.squeeze().numpy())
        
        action, _ = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            break
    
    # 绘制状态变化
    states = np.array(states)
    action_probs_history = np.array(action_probs_history)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 状态变化
    axes[0, 0].plot(states[:, 0])
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(states[:, 1])
    axes[0, 1].set_title('Cart Velocity')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(states[:, 2])
    axes[1, 0].set_title('Pole Angle')
    axes[1, 0].set_ylabel('Angle (radians)')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(states[:, 3])
    axes[1, 1].set_title('Pole Angular Velocity')
    axes[1, 1].set_ylabel('Angular Velocity')
    axes[1, 1].grid(True)
    
    # 动作序列
    axes[2, 0].step(range(len(actions)), actions, where='post')
    axes[2, 0].set_title('Action Sequence')
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Action (0: Left, 1: Right)')
    axes[2, 0].set_ylim(-0.1, 1.1)
    axes[2, 0].grid(True)
    
    # 动作概率
    axes[2, 1].plot(action_probs_history[:, 0], label='Left (0)', alpha=0.7)
    axes[2, 1].plot(action_probs_history[:, 1], label='Right (1)', alpha=0.7)
    axes[2, 1].set_title('Action Probabilities')
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].set_ylabel('Probability')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.suptitle(f'Episode {episode + 1} State Analysis (Length: {len(actions)})')
    plt.tight_layout()
    
    # 保存图片但不显示
    save_path = f'results/state_analysis_episode_{episode + 1}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存
    print(f"State analysis chart saved to: {save_path}")
    
    env.close()

def create_comparison_gif(model_path, episodes_to_compare=2):
    """创建对比gif，显示多个回合的表现"""
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # 创建智能体并加载模型
    agent = PPOAgent(4, 2)
    agent.load_model(model_path)
    
    all_episodes_frames = []
    
    for episode in range(episodes_to_compare):
        state, _ = env.reset()
        episode_frames = []
        episode_reward = 0
        episode_length = 0
        
        print(f"录制对比回合 {episode + 1}...")
        
        while True:
            frame = env.render()
            
            # 添加回合信息
            info_text = f"Episode {episode + 1}: Step {episode_length + 1}, Reward {episode_reward}"
            frame_with_text = add_text_to_frame(frame, info_text)
            episode_frames.append(frame_with_text)
            
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        all_episodes_frames.append(episode_frames)
    
    # 创建对比gif
    max_length = max(len(frames) for frames in all_episodes_frames)
    comparison_frames = []
    
    for i in range(max_length):
        # 为每个时间步创建并排的帧
        row_frames = []
        for episode_frames in all_episodes_frames:
            if i < len(episode_frames):
                row_frames.append(episode_frames[i])
            else:
                # 如果某个回合已经结束，使用最后一帧
                row_frames.append(episode_frames[-1])
        
        # 水平拼接帧
        if len(row_frames) > 1:
            combined_frame = np.hstack(row_frames)
        else:
            combined_frame = row_frames[0]
            
        comparison_frames.append(combined_frame)
    
    # 保存对比gif
    comparison_gif_path = 'results/comparison.gif'
    print(f"保存对比gif到 {comparison_gif_path}...")
    imageio.mimsave(comparison_gif_path, comparison_frames[::2], fps=30, loop=0)
    print(f"对比gif已保存: {comparison_gif_path}")
    
    env.close()