# 🎮 PPO Tutorial - Reinforcement Learning Framework

> **By RuoChen from ZJU**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Algorithm-PPO-red.svg" alt="PPO">
  <img src="https://img.shields.io/badge/Environment-CartPole-green.svg" alt="Environment">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📋 目录

- [环境配置](#-环境配置)
- [快速开始](#-快速开始)
- [训练模型](#-训练模型)
- [测试评估](#-测试评估)
- [效果展示](#-效果展示)
- [项目结构](#-项目结构)

---

## 🛠 环境配置

### 创建虚拟环境

```bash
# 创建Conda环境
conda create --name PPO_Tutorial python=3.10
conda activate PPO_Tutorial

# 安装依赖
conda install -r requirements.txt
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone [your-repository-url]
cd PPO_Tutorial
```

### 2. 配置环境

```bash
conda activate PPO_Tutorial
```

### 3. 运行测试

```bash
python test.py
```

---

## 🎯 训练模型

### 开始训练

```bash
python train.py
```

### 输出说明

| 输出类型 | 保存位置 | 说明 |
|---------|---------|-----|
| 训练好的模型 | `./models/` | PPO策略网络和价值网络 |
| 训练数据 | `./data/` | 训练过程中的日志和统计数据 |
| 训练曲线 | `./data/` | 奖励曲线、损失曲线等 |

---

## 🧪 测试评估

### 运行测试

```bash
python test.py
```

### 测试输出

- 测试结果保存在 `./results` 目录下
- 包含测试视频、性能指标等

---

## 🎨 效果展示

<div align="center">
  <h3>PPO CartPole 控制效果</h3>
  <img src="https://github.com/Chromium0516/RL_Tutorial/blob/PPO_CartPole/results/test_episode_1.gif" alt="PPO Pendulum Demo" width="400">
</div>

---

## 📁 项目结构

```
.
├── models/                     # 保存训练好的模型
│   ├── ppo_cartpole_model.pth  # 模型
├── data/                   # 训练数据和日志
│   ├── training_data.npy   # 训练数据
├── results/                # 测试结果
│   ├── test_episode_1.gif  # 测试动画
│   └── state_analysis.png  # 测试指标
├── Utils/                  # 工具函数
│   ├── netUtils_PPO.py    # 构建网络
│   └── testUtils_PPO.py   # 测试工具
│   └── trainUtils_PPO.py  # 训练工具
├── train.py               # 训练脚本
├── test.py                # 测试脚本
└── requirements.txt       # 项目依赖
```

---

## 📊 性能指标

| 指标 | 描述 |
|-----|-----|
| 平均奖励 | 训练过程中的平均episode奖励 |
| 收敛速度 | 达到目标性能所需的训练步数 |
| 稳定性 | 训练后策略的稳定程度 |

---

## 💡 使用提示

1. **调整超参数**：可以在 `train.py` 中修改学习率、批大小等超参数
2. **更换环境**：支持其他 Gym 环境，只需修改环境名称
3. **可视化**：运行训练时会自动生成训练曲线图

---

<p align="center">
  <i>如有问题，欢迎提交 Issue 或 PR！</i>
</p>
