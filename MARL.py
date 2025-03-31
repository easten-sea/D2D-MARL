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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def simulate_task_arrivals(lam, time_slots, seed=None):
    """
    模拟多个时隙中的任务到达次数。

    :param lam: 泊松过程的λ参数（每个时隙内任务到达的平均率）
    :param time_slots: 时隙的数量
    :param seed: 随机种子（用于可重复性）
    :return: 每个时隙内的任务到达次数列表
    """
    if seed is not None:
        np.random.seed(seed)

    # 使用泊松分布生成每个时隙内的任务到达次数
    task_arrivals = np.random.poisson(lam, time_slots)
    return task_arrivals

class xxx_agent(agent):
    def __init__(self, args, index):
        agent.__init__(self, args, index)
        self.actor = Actor(args)
        self.target_actor = Actor(args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic = Critic(args, 128, 1)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.learning_rate)
        self.target_critic = Critic(args, 128, 1)
        self.env_cognition = neigh_cognition(args, args.neighbors_num * 3 + 3 + args.task_em_size,
                                             args.env_cognition_size, args.neighbors_num * 3 + 3 + args.task_em_size)
        self.env_optimizer = torch.optim.Adam(self.env_cognition.parameters(), lr=args.learning_rate * 0.001)
        self.act_cognition = neigh_cognition(args, args.act_em_size, args.act_cognition_size, args.act_em_size)
        self.act_optimizer = torch.optim.Adam(self.act_cognition.parameters(), lr=args.learning_rate * 0.001)
        self.pred_r_module = prediction_reward(args)
        self.r_optimizer = torch.optim.Adam(self.pred_r_module.parameters(), lr=args.learning_rate * 0.001)
        self.replay_buffer = ReplayBuffer(12800)


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

class Critic(nn.Module):
    def __init__(self, args, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.act_em_size,  hidden_dim)
        self.fc2 = nn.Linear(args.env_cognition_size + args.act_em_size,  hidden_dim)
        self.fc3 = nn.Linear(args.act_cognition_size +  hidden_dim,  hidden_dim * 2)
        self.fc4 = nn.Linear( hidden_dim * 3, output_dim)

    def forward(self, observations, actions, act_cogns):
        out_put = []
        for o, a, act_cog in zip(observations, actions, act_cogns):
            z = torch.cat([o, a])
            y = self.fc1(a)
            z = self.fc2(z)
            out = torch.cat([act_cog, z])
            out = self.fc3(out)
            out = F.relu(out)
            out = torch.cat([y, out])
            out = self.fc4(out)
            out_put.append(out)

        return torch.cat(out_put)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.rnn = nn.RNN(128, 128, batch_first=True)
        self.fc1 = nn.Linear(args.neighbors_num * 3 + 3, 128)
        self.fc2 = nn.Linear(args.task_em_size, args.task_em_size * 3)
        self.fc4 = nn.Linear(128 + args.task_em_size * 3, args.neighbors_num + 1)
        self.hidden = torch.zeros(1, 1, 128)
        self.ounoise = OUNoise(args.neighbors_num + 1)
        self.fc5 = nn.Linear(128, (args.neighbors_num + 1) * 11)
        self.fc6 = nn.Linear((args.neighbors_num + 1) * 12, 1)
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, o):
        e = o[0]
        k = o[1]
        x = self.fc1(e)
        x = F.relu(x)
        x = x.reshape(1, 1, 128)
        #必须把隐藏状态detach，不然的话会反向传播到上一副计算图
        x, hid  = self.rnn(x, self.hidden)
        self.hidden = hid.detach()
        x = x.reshape(128)
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

class neigh_cognition(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim):
        super(neigh_cognition, self).__init__()
        self.args = args
        #将输入的环境向量或者动作向量进行映射
        self.hidden_dim = hidden_dim
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear((args.neighbors_num + 1) * hidden_dim, hidden_dim)
        self.rnn = nn.RNN(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.hidden = torch.zeros(1, 1, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.refc1  = nn.Linear(self.hidden_dim, self.hidden_dim * output_dim)
        self.refc2 = nn.Linear(self.hidden_dim * output_dim, output_dim)

    def check_for_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(tensor)
            raise ValueError(f"NaN detected in {name}")

    def forward(self, x, neighbors):
        query = self.query_layer(x)
        query = query.reshape(-1, 1, self.hidden_dim)
        keys = [self.key_layer(i) for i in neighbors]
        keys = torch.stack(keys, dim=0)
        keys = keys.unsqueeze(1).permute(1, 2, 0)
        attend_logits = torch.bmm(query, keys)
        scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
        attend_weights = F.softmax(scaled_attend_logits, dim=2)
        values = [self.value_layer(j) for j in neighbors]
        values = torch.stack(values, dim=0)
        values = values.unsqueeze(1).permute(1, 2, 0)

        attention_value = torch.mul(values, attend_weights)
        attention_value = attention_value.permute(0, 2, 1)
        att = torch.cat([query.squeeze(0), attention_value.squeeze(0)]).reshape((self.args.neighbors_num + 1) * self.hidden_dim)
        H = F.relu(self.output_layer(att))
        C_mean, hid = self.rnn(H.reshape(1, 1, self.hidden_dim), self.hidden)
        self.hidden = hid.detach()
        C_mean = C_mean.reshape(self.hidden_dim)
        C_dot = self.fc(H)
        C = C_mean + C_dot * torch.normal(mean=0.0, std=1.0, size=C_dot.shape)


        #重建
        mid = self.refc1(C)
        x_hat = self.refc2(mid)

        return C, x_hat

class prediction_reward(nn.Module):
    def __init__(self, args):
        super(prediction_reward, self).__init__()
        self.fc1 = nn.Linear(args.task_em_size + args.neighbors_num * 3 + 3, args.act_em_size * 10)
        self.fc2 = nn.Linear(args.act_em_size * 10 + args.act_em_size, 11)
        self.fc3 = nn.Linear(12, 1)

        #处理上一个设备
        self.fc4 = nn.Linear(args.env_cognition_size + args.act_em_size, (args.env_cognition_size + args.act_em_size) * 10)
        self.fc5 = nn.Linear((args.env_cognition_size + args.act_em_size) * 10, 11)
        self.fc6 = nn.Linear(12, 1)
        self.fc7 = nn.Linear(args.env_cognition_size + args.task_em_size, args.env_cognition_size)
        self.scaling_factor = 0.01

        self.dropout = nn.Dropout(0.5)

    def check_for_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(tensor)
            raise ValueError(f"NaN detected in {name}")

    def forward(self, o, a, task_info):
        # 检查输入是否有 NaN
        self.check_for_nan(o[0], "o[0]")
        self.check_for_nan(o[1], "o[1]")
        self.check_for_nan(a, "a")
        if task_info[-1] is not None:
            self.check_for_nan(task_info[-1], "task_info[-1]")
            self.check_for_nan(task_info[2], "task_info[2]")
            self.check_for_nan(task_info[3], "task_info[3]")

        #任务卸载序列头时
        pre_r = torch.zeros([1])
        x = torch.cat([o[0], o[1]])
        if task_info[-1] == None:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x) * self.scaling_factor
            y = torch.cat([x, a])
            y = self.fc2(y)
            y = torch.cat([y, torch.zeros(1)])
            out = self.fc3(y)

        else:
            #获取上一个设备的预测奖励值标签
            z = torch.cat([task_info[3], o[1]])
            z = self.fc7(z)
            # z = F.relu(z)
            z = torch.cat([z, task_info[2]])
            z = self.fc4(z)
            z = F.relu(z)
            z = self.dropout(z) * self.scaling_factor
            z = self.fc5(z)
            pre_r = self.fc6(torch.cat([z, task_info[-1]]))


            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x) * self.scaling_factor
            y = torch.cat([x, a])
            y = self.fc2(y)
            y = torch.cat([y, pre_r])
            out = self.fc3(y)
        # 检查输出是否有 NaN
        self.check_for_nan(out, "out")
        self.check_for_nan(pre_r, "pre_r")

        return torch.tanh(out), torch.tanh(pre_r)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def store(self, state, action, act_cong, reward, next_state, observation,  new_ob, done):
        """存储经验"""
        experience = (state, action, act_cong, reward, next_state, observation, new_ob, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """随机采样一个批次的经验"""
        batch = random.sample(self.buffer, batch_size)

        # 将批次的经验解包成独立的 numpy 数组
        states, actions, act_congs, rewards, next_states, observations, new_obs, dones = zip(*batch)
        return states, actions, act_congs, rewards, next_states, observations, new_obs, dones

    def __len__(self):
        """返回当前 Replay Buffer 中的经验数量"""
        return len(self.buffer)








