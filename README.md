<span style="color:rgb(0,0,255)">By RuoChen from ZJU</span>

---

# 运行步骤
conda create --name PPO_Tutorial python=3.10
conda activate PPO_Tutorial
conda install -r requirements.txt

---

# 进行测试
python test.py
### 结果保存在`./results`目录下

---

# 进行训练
python train.py
### 模型保存在`./models`目录下
### 训练数据保存在`./data`目录下

---

# 效果
<div align="center">
  <img src="https://github.com/Chromium0516/LLM_RL/blob/PPO_pendulum/results/test_episode_1.gif" alt="PPO Pendulum Demo">
</div>
