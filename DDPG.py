import random
from sysmodel import task
import torch
import torch.nn as nn
import queue
import  numpy as np
import torch.nn.functional as F
import math
from collections import deque
from sysmodel import agent
from MARL import OUNoise

class DDPG_agent(agent):
    def __init__(self, args, index):
        agent.__init__(self, args, index)
        self.actor = DDPG_actor(args)
        self.target_actor = DDPG_actor(args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic = DDPG_critic(args, 128, 1)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.learning_rate)
        self.target_critic = DDPG_critic(args, 128, 1)
        self.replay_buffer = DDPG_ReplayBuffer(12800)


class DDPG_actor(nn.Module):
    def __init__(self, args):
        super(DDPG_actor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.neighbors_num * 3 + 3, 128)
        self.fc2 = nn.Linear(args.task_em_size, args.task_em_size * 3)
        self.fc4 = nn.Linear(128 + args.task_em_size * 3, args.neighbors_num + 1)
        self.ounoise = OUNoise(args.neighbors_num + 1)
        self.fc5 = nn.Linear(128, (args.neighbors_num + 1) * 11)
        self.fc6 = nn.Linear((args.neighbors_num + 1) * 12, 1)


    def forward(self, o):
        e = o[0]
        k = o[1]
        x = self.fc1(e)
        x = F.relu(x)
        y = self.fc2(k)
        y = F.relu(y)

        # 拼接输入张量
        concatenated_input1 = torch.cat([x, y], dim=-1)
        #添加模拟Ornstein-Uhlenbeck过程的时间相关的噪声
        out = self.fc4(concatenated_input1) + torch.from_numpy(self.ounoise.noise()).to(torch.float32)

        #进一步结合卸载策略生成本设备的运行功率
        z = self.fc5(x)
        concatenated_input2 = torch.cat([z, out], dim=-1)
        #将cpu输出变为0，1之间，然后基于最小的cpu频率要求来计算得到的具体的cpu的值
        cpu_fre = torch.sigmoid(self.fc6(concatenated_input2))

        return F.softmax(out, dim= -1), cpu_fre


class DDPG_critic(nn.Module):
    def __init__(self, args, hidden_dim, output_dim):
        super(DDPG_critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear((args.neighbors_num * 3 + 3 + args.task_em_size) * args.device_num,  (args.neighbors_num * 3 + 3 + args.task_em_size) * args.device_num)
        self.fc2 = nn.Linear(args.act_em_size,  hidden_dim)
        self.fc3 = nn.Linear((args.neighbors_num * 3 + 3 + args.task_em_size) * args.device_num + args.act_em_size,  hidden_dim)
        self.fc4 = nn.Linear( args.act_em_size, hidden_dim)
        self.fc5 = nn.Linear( hidden_dim * 2, output_dim)
        self.fc6 = nn.Linear(args.device_num * args.act_em_size, args.act_em_size)

    def forward(self, observations, actions, index):
        out_put = []
        for o, a in zip(observations, actions):
            #汇总所有观测并映射
            x1 = torch.cat([i[0] for i in o])
            x2 = torch.cat([i[1] for i in o])
            y = torch.cat([x1, x2])

            y = self.fc1(y)
            p = self.fc6(torch.cat(a))
            x = self.fc2(a[index])
            z = torch.cat([y, p])
            z = self.fc3(z)
            z =  F.relu(z)

            # m = self.fc4(x)
            m = F.relu(x)
            z = torch.cat([z, m])
            out = self.fc5(z)
            # out = F.relu(out)
            out_put.append(out)

        return torch.cat(out_put)

class DDPG_ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def store(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """随机采样一个批次的经验"""
        batch = random.sample(self.buffer, batch_size)

        # 将批次的经验解包成独立的 numpy 数组
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回当前 Replay Buffer 中的经验数量"""
        return len(self.buffer)