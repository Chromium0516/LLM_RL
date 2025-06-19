import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-5, gamma=0.995, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 复制参数到old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.memory = []
        
    def select_action(self, state):
        # 确保state是numpy数组并转换为tensor
        if isinstance(state, tuple):
            state = state[0]  # 处理gymnasium返回的(state, info)格式
        state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy_old(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))
    
    def calculate_returns_and_advantages(self, rewards, values, dones):
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1] if not dones[i] else 0
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return returns, advantages
    
    def update(self):
        # 提取经验
        states = []
        for transition in self.memory:
            state = transition[0]
            if isinstance(state, tuple):
                state = state[0]
            states.append(np.array(state))
        states = torch.FloatTensor(np.array(states))
        
        actions = torch.LongTensor([transition[1] for transition in self.memory])
        old_log_probs = torch.FloatTensor([transition[2] for transition in self.memory])
        rewards = [transition[3] for transition in self.memory]
        
        next_states = []
        for transition in self.memory:
            next_state = transition[4]
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_states.append(np.array(next_state))
        next_states = torch.FloatTensor(np.array(next_states))
        
        dones = [transition[5] for transition in self.memory]
        
        # 计算值函数
        values = self.value_net(states).squeeze().detach().numpy()
        
        # 计算回报和优势
        returns, advantages = self.calculate_returns_and_advantages(rewards, values, dones)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(self.k_epochs):
            # 计算当前策略的动作概率
            action_probs = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # 值函数损失
            current_values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(current_values, returns)
            
            # 更新网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # 更新old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空记忆
        self.memory = []
        
        return policy_loss.item(), value_loss.item()
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        print(f"模型已从 {filepath} 加载")

def train_ppo():
    env = gym.make('CartPole-v1', render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    
    max_episodes = 18000
    max_timesteps = 8000
    update_timestep = 1000
    
    # 用于记录训练过程
    episode_rewards = []
    policy_losses = []
    value_losses = []
    
    timestep = 0
    
    print("开始训练PPO智能体...")
    
    for episode in range(max_episodes):
        state, _ = env.reset()  # gymnasium返回(state, info)
        episode_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # 选择动作
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)  # gymnasium返回5个值
            done = terminated or truncated
            
            # 存储经验
            agent.store_transition(state, action, log_prob, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # 更新策略
            if timestep % update_timestep == 0:
                policy_loss, value_loss = agent.update()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 打印进度
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
            
            # 如果平均奖励超过195，认为解决了问题
            if avg_reward >= 500:
                print(f"环境在第 {episode} 回合被解决!")
                break
    
    env.close()
    return agent, episode_rewards, policy_losses, value_losses

def visualize_training_results(episode_rewards, policy_losses, value_losses):
    """可视化训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 回合奖励
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('每回合奖励')
    axes[0, 0].set_xlabel('回合')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].grid(True)
    
    # 滑动平均奖励
    window_size = 50
    if len(episode_rewards) >= window_size:
        moving_avg = [np.mean(episode_rewards[i:i+window_size]) 
                     for i in range(len(episode_rewards) - window_size + 1)]
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'{window_size}回合滑动平均奖励')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('平均奖励')
        axes[0, 1].grid(True)
    
    # 策略损失
    if policy_losses:
        axes[1, 0].plot(policy_losses)
        axes[1, 0].set_title('策略损失')
        axes[1, 0].set_xlabel('更新次数')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].grid(True)
    
    # 值函数损失
    if value_losses:
        axes[1, 1].plot(value_losses)
        axes[1, 1].set_title('值函数损失')
        axes[1, 1].set_xlabel('更新次数')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练结果图已保存为 training_results.png")

def save_training_data(episode_rewards, policy_losses, value_losses):
    """保存训练数据"""
    training_data = {
        'episode_rewards': episode_rewards,
        'policy_losses': policy_losses,
        'value_losses': value_losses
    }
    np.save('training_data.npy', training_data)
    print("训练数据已保存为 training_data.npy")

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 训练智能体
    trained_agent, episode_rewards, policy_losses, value_losses = train_ppo()
    
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