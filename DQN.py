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


class DQN_agent(agent):
    def __init__(self, args, index):
        agent.__init__(self, args, index)
        if args.alth == 'DQN':
            self.DQN = DQN(args).float()
            self.target_DQN = DQN(args).float()
        if args.alth == 'D3QN':
            self.DQN = Dueling_DQN(args).float()
            self.target_DQN = Dueling_DQN(args).float()
        self.optimizer = torch.optim.Adam(self.DQN.parameters(), lr=args.learning_rate)
        self.replay_buffer = DQN_ReplayBuffer(12800)



class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.neighbors_num * 3 + 3, 128)
        self.fc2 = nn.Linear(args.task_em_size, args.task_em_size * 3)
        self.fc4 = nn.Linear(128 + args.task_em_size * 3, args.neighbors_num + 1)

        self.fc7 = nn.Linear((args.neighbors_num * 3 + 3) * args.device_num, (args.neighbors_num * 3 + 3))
        self.fc8 = nn.Linear(args.task_em_size * args.device_num, args.task_em_size)

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.kaiming_uniform_(self.fc7.weight)
        nn.init.kaiming_uniform_(self.fc8.weight)

    def forward(self, o):
        e = []
        k = []
        for ob in o:
            e.append(ob[0])
            k.append(ob[1])
        e = self.fc7(torch.stack(e).reshape(-1))
        k = self.fc8(torch.stack(k).reshape(-1))
        x = self.fc1(e)
        x = F.relu(x)
        y = self.fc2(k)
        y = F.relu(y)

        # 拼接输入张量
        concatenated_input1 = torch.cat([x, y], dim=-1)
        #添加模拟Ornstein-Uhlenbeck过程的时间相关的噪声
        out = self.fc4(concatenated_input1)
        return F.softmax(out, dim= -1)

class Dueling_DQN(nn.Module):
    def __init__(self, args):
        super(Dueling_DQN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.neighbors_num * 3 + 3, 128)
        self.fc2 = nn.Linear(args.task_em_size, args.task_em_size * 3)
        self.fc3 = nn.Linear(128 + args.task_em_size * 3, args.neighbors_num + 1)

        #advantage stream
        self.advantage_fc1 = nn.Linear((args.neighbors_num * 3 + 3) * args.device_num + args.task_em_size * args.device_num, 128)
        self.advantage_fc2 = nn.Linear(128, args.neighbors_num + 1)

        #value stream
        self.value_fc1 = nn.Linear((args.neighbors_num * 3 + 3) * args.device_num + args.task_em_size * args.device_num, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self.fc7 = nn.Linear((args.neighbors_num * 3 + 3) * args.device_num, (args.neighbors_num * 3 + 3))
        self.fc8 = nn.Linear(args.task_em_size * args.device_num, args.task_em_size)

    def forward(self, o):
        e = []
        k = []
        for ob in o:
            e.append(ob[0])
            k.append(ob[1])
        e = torch.stack(e).reshape(-1)
        k = torch.stack(k).reshape(-1)

        # 拼接输入张量
        concatenated_input = torch.cat([e, k], dim=-1)

        advan = torch.relu(self.advantage_fc1(concatenated_input))
        advan = self.advantage_fc2(advan)

        val = torch.relu(self.value_fc1(concatenated_input))
        val = self.value_fc2(val)

        out = val + (advan - advan.mean(dim = -1, keepdim = True))

        return F.softmax(out, dim= -1)

class DQN_ReplayBuffer:
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