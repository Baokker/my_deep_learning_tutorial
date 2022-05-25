import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym  # 包含CartPole的平台环境

import matplotlib.pyplot as plt # 绘图

# Hyperparameters
BATCH_SIZE = 32  # 样本数量
LR = 0.01  # 学习率
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # 目标网络更新频率
MEMORY_CAPACITY = 2000  # 记忆库容量

# 使用gym库中的环境：CartPole，且打开封装
# 在pycharm中可通过两次shift打开全局搜索，找到源码
env = gym.make('CartPole-v0').unwrapped  

N_ACTIONS = env.action_space.n  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]  # 杆子状态个数 (4个)

# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):  
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 30)  # 设置第一个全连接层
        self.fc1.weight.data.normal_(0, 0.1) 
        self.out = nn.Linear(30, N_ACTIONS)  # 设置第二个全连接层
        self.out.weight.data.normal_(0, 0.1) 

    def forward(self, x):  
        x = F.relu(self.fc1(x))  # 使用ReLU
        actions_value = self.out(x) 
        return actions_value


class DQN(object):
    def __init__(self):  # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()  # 利用Net创建评估网络和目标网络
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # Adam优化器
        self.loss_func = nn.MSELoss()  # loss(xi, yi)=(xi-yi)^2

    def choose_action(self, x):  # 根据x状态选择动作
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:  # 生成随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            np.random.shuffle(action) # 打乱顺序，避免总是选第0个
            action = action[0]  # 输出第0个
        else:  # 随机选择
            action = np.random.randint(0, N_ACTIONS) 
        return action  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 每100步触发，实现Fixed Q-targets
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # Experience Replay
        # 在[0, 2000)内随机抽取32个数，可能会重复
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # q_next不进行反向传递误差，所以detach
        q_next = self.target_net(b_s_).detach()
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；
        # .view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        
        # 输入32个评估值和32个目标值，使用均方损失函数
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数


dqn = DQN()

y = np.array([]) # 记录score

for i in range(300):  # 300个episode循环
    print('Episode: %s' % i)
    s = env.reset()  # 重置环境，先reset才能render

    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励（加速训练）
    score = 0 # 初始化得分

    while True:  # 开始一个episode (每一个循环代表一步)
        env.render()  # 显示实验动画

        a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)  # 执行动作，获得反馈

        # 修改奖励 根据x的偏离和theta的角度差打分
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        score += 1

        dqn.store_transition(s, a, new_r, s_)  # 存储

        episode_reward_sum += new_r

        s = s_  # 更新状态

        if dqn.memory_counter > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取32个transition，并对评估网络参数进行更新
            # 并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()

        if done:  # 如果done为True
            print('episode%s -- reward_sum: %s' % (i, score))
            break  # 该episode结束

    y = np.append(y, episode_reward_sum)

x = np.arange(0, 300) # x坐标

plt.title("Result")
plt.plot(x, y, color="red") # 绘图
plt.show()