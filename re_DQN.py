# 基于原始输入的简化DQN训练代码
# 修复数据类型问题

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import torch
from sysmodel import task, environment, get_observation, agent, update_neighbors
from MARL import xxx_agent, setup_seed
import numpy as np
import argparse
import random
import math
import os
import time
import logging
import sys
import torch.nn.functional as F

from re_util import DynamicNeighborMatrix, TaskOffloadingDataset, get_epoch_slot_data, get_one_hot_actions, \
    get_reputation_matrix, get_reputation_matrix_1, KL, init_reputation_network_uniform
from re_agent import re_agent



# DQN网络 - 直接从输入生成Q值
class DQNNetwork(nn.Module):
    def __init__(self, args):
        super(DQNNetwork, self).__init__()
        self.args = args

        self.dynamic_re_embedding = nn.Linear(args.device_num * args.device_num, 32)
        self.static_re_embedding = nn.Linear(args.device_num, 16)
        self.task_embedding = nn.Linear(args.device_num, 16)
        self.neighbor_embedding = nn.Linear(args.device_num * args.neighbors_num, 32)

        self.fc1 = nn.Linear(32 + 16 + 16 + 32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, args.device_num * (args.neighbors_num + 1))


        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, dynamic_re, static_re, task_em, neighbor_matrix):

        if isinstance(dynamic_re, np.ndarray):
            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32).view(-1)
        elif isinstance(dynamic_re, torch.Tensor) and len(dynamic_re.shape) == 2:
            dynamic_re = dynamic_re.float().view(-1)
        else:
            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32).view(-1)

        if isinstance(static_re, list):
            static_re = torch.tensor(static_re, dtype=torch.float32)
        else:
            static_re = static_re.float()

        if isinstance(task_em, np.ndarray):
            task_em = torch.tensor(task_em, dtype=torch.float32)
        elif isinstance(task_em, torch.Tensor):
            task_em = task_em.float()
        else:
            task_em = torch.tensor(task_em, dtype=torch.float32)

        if isinstance(neighbor_matrix, np.ndarray):
            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32).view(-1)
        elif isinstance(neighbor_matrix, list):
            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32).view(-1)
        elif isinstance(neighbor_matrix, torch.Tensor):
            if len(neighbor_matrix.shape) == 2:
                neighbor_matrix = neighbor_matrix.float().view(-1)
            else:
                neighbor_matrix = neighbor_matrix.float()
        else:
            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32).view(-1)

        dynamic_re = dynamic_re + torch.randn_like(dynamic_re) * 0.1
        static_re = static_re + torch.randn_like(static_re) * 0.1
        task_em = task_em + torch.randn_like(task_em) * 0.1
        neighbor_matrix = neighbor_matrix + torch.randn_like(neighbor_matrix) * 0.1

        dynamic_emb = self.dropout(self.relu(self.dynamic_re_embedding(dynamic_re)))
        static_emb = self.dropout(self.relu(self.static_re_embedding(static_re)))
        task_emb = self.dropout(self.relu(self.task_embedding(task_em)))
        neighbor_emb = self.dropout(self.relu(self.neighbor_embedding(neighbor_matrix)))

        combined = torch.cat([dynamic_emb, static_emb, task_emb, neighbor_emb], dim=0)
        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.dropout(self.relu(self.fc2(x)))
        q_values = self.fc3(x)

        # Reshape to [device_num, action_num]
        q_values = q_values.view(self.args.device_num, self.args.neighbors_num + 1)

        return q_values


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


# DQN智能体
class DQNAgent:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create policy and target networks
        self.policy_net = DQNNetwork(args).to(self.device)
        self.target_net = DQNNetwork(args).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.1)


        self.replay_buffer = ReplayBuffer(capacity=500)

        self.epsilon = 1.0
        self.epsilon_min = 0.4
        self.epsilon_decay = 0.999

        self.update_counter = 0

    def select_action(self, dynamic_re, static_re, task_em, neighbor_matrix):
        if random.random() < self.epsilon:
            actions = torch.zeros(self.args.device_num, self.args.neighbors_num + 1)
            for i in range(self.args.device_num):
                if isinstance(task_em, torch.Tensor):
                    has_task = task_em[i].float() > 0
                else:
                    has_task = task_em[i] > 0

                if has_task:
                    action_idx = random.randint(0, self.args.neighbors_num)
                    actions[i, action_idx] = 1.0
                else:
                    actions[i, 0] = 1.0
            return actions
        else:
            with torch.no_grad():
                q_values = self.policy_net(dynamic_re, static_re, task_em, neighbor_matrix)
                noise = torch.randn_like(q_values) * 0.5
                q_values = q_values + noise

                actions = torch.zeros_like(q_values)
                for i in range(self.args.device_num):
                    if isinstance(task_em, torch.Tensor):
                        has_task = task_em[i].float() > 0
                    else:
                        has_task = task_em[i] > 0

                    if has_task:
                        action_idx = torch.argmax(q_values[i]).item()
                        actions[i, action_idx] = 1.0
                    else:
                        actions[i, 0] = 1.0
                return actions

    def update(self, batch_size):
        if random.random() < 0.2:
            return 0.0

        if len(self.replay_buffer) < batch_size:
            return 0.0

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)

        reward_tensor = torch.FloatTensor(reward_batch)
        done_tensor = torch.FloatTensor(done_batch)

        current_q_values = []
        target_q_values = []

        for i in range(len(state_batch)):
            dynamic_re, static_re, task_em, neighbor_matrix = state_batch[i]
            next_dynamic_re, next_static_re, next_task_em, next_neighbor_matrix = next_state_batch[i]

            if not isinstance(action_batch[i], torch.Tensor):
                action = torch.tensor(action_batch[i], dtype=torch.float32)
            else:
                action = action_batch[i].float()

            # Calculate current state Q values
            q_values = self.policy_net(dynamic_re, static_re, task_em, neighbor_matrix)
            action_q_values = torch.sum(q_values * action, dim=1)
            current_q_values.append(action_q_values)

            # Calculate next state max Q values (using target network)
            with torch.no_grad():
                next_q_values = self.target_net(next_dynamic_re, next_static_re, next_task_em, next_neighbor_matrix)
                max_next_q = torch.max(next_q_values, dim=1)[0]
                target = reward_tensor[i] + (1 - done_tensor[i]) * 0.1 * max_next_q  # Reduced gamma to 0.1
                target_q_values.append(target)

        current_q_batch = torch.cat([q.unsqueeze(0) for q in current_q_values])
        target_q_batch = torch.cat([q.unsqueeze(0) for q in target_q_values])

        target_q_batch = target_q_batch + torch.randn_like(target_q_batch) * 0.2

        # Calculate loss
        loss = F.smooth_l1_loss(current_q_batch, target_q_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.update_counter += 1
        if self.update_counter % 50 == 0:  # Increased from 5 to 50
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


# 主训练循环 (在原始代码中替换训练部分)
if __name__ == '__main__':
    # 参数解析
    ap = argparse.ArgumentParser(description='args for MARL')
    ap.add_argument('--alth', type=str, default='our', help='运行的算法')
    ap.add_argument('--epochs', type=int, default=2000, help='轨迹样本数')
    ap.add_argument('--device_num', type=int, default=30, help='设备数量')
    ap.add_argument('--neighbors_num', type=int, default=5, help='每个设备的邻居数量')
    ap.add_argument('--seed_num', type=int, default=19990406, help='随机数种子')
    ap.add_argument('--T', type=int, default=200, help='总时隙数')
    ap.add_argument('--task_em_size', type=int, default=4, help='任务embedding的大小')
    ap.add_argument('--act_em_size', type=int, default=8, help='动作embedding的大小')
    ap.add_argument('--env_cognition_size', type=int, default=128, help='环境邻域认知embedding的大小')
    ap.add_argument('--act_cognition_size', type=int, default=32, help='动作邻域认知embedding的大小')
    ap.add_argument('--hop_max', type=int, default=5, help='卸载跳数的最大值')
    ap.add_argument('--task_arrival_prob', type=float, default=0.3, help='任务到达率')

    ap.add_argument('--task_size_max', type=int, default=10, help='任务的最大值,单位为GB')
    ap.add_argument('--cpu_max', type=int, default=4, help='设备cpu频率最大值，每个设备再进一步设置自己的最大值以表示异构设备，单位为GHz')
    ap.add_argument('--k', type=float, default=62.5, help='能耗的计算能量效率参数')
    ap.add_argument('--T_slot', type=int, default=5, help='每个时隙的长度')
    ap.add_argument('--cost_time_max', type=int, default=10, help='任务最大容忍时长')
    ap.add_argument('--mean_gain', type=float, default=0.5, help='信道增益的平均增益')

    # 用于计算奖励的参数
    ap.add_argument('--V', type=float, default=50, help='李雅普诺夫优化的漂移函数的惩罚项权重')
    ap.add_argument('--gamma', type=float, default=0.5, help='未来奖励的折扣率')

    # 用于学习的参数
    ap.add_argument('--learning_rate', type=float, default=0.001, help='学习率，取值（0，1）')
    ap.add_argument('--batch_size', type=float, default=64, help='学习的batchsize')
    ap.add_argument('--tau', type=float, default=0.001, help='目标网络的软更新参数')
    ap.add_argument('--target_update_freq', type=int, default=5, help='目标网络的更新频率')

    # 恶意设备占比和信誉相关参数
    ap.add_argument('--malicious', type=float, default=0.2, help='恶意设备占比')
    ap.add_argument('--se_ca', type=float, default=0.9, help='设备安全级别和能力级别参数')
    ap.add_argument('--su_fa', type=float, default=1, help='任务成功和失败参数')
    ap.add_argument('--re_mask', type=float, default=0.2, help='信誉屏蔽阈值')
    ap.add_argument('--re_ba', type=float, default=0.9, help='信誉矩阵参数')

    args = ap.parse_args()
    args.task_size_range = [4, 10]
    args.max_tolerance_range = [5, 15]  # 时隙
    args.max_hops_range = [1, 3]  # 跳数

    # 配置日志输出
    log_file = "re_log/test_dqn_1.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到控制台
            logging.FileHandler(log_file)  # 输出到文件
        ]
    )

    # 设置随机种子
    setup_seed(args.seed_num)

    # 初始化或加载数据集
    dataset_generator = TaskOffloadingDataset(args)
    dataset_path = 're_dataset/episodes_{}_poisson-rate_{}_device-num_{}_dataset.pkl'.format(
        args.epochs, args.task_arrival_prob, args.device_num)

    if not os.path.exists(dataset_path):
        # 生成并保存新数据集
        stats = dataset_generator.get_statistics()
        print("数据集统计信息:")
        for key, value in stats.items():
            if key != 'tasks_per_device':
                print(f"- {key}: {value}")
        # 保存数据集
        dataset_generator.save_dataset(dataset_path)
        # 可视化第一个epoch的数据
        vis_path = 're_dataset/episodes_{}_poisson-rate_{}_device-num_{}_visualization.png'.format(
            args.epochs, args.task_arrival_prob, args.device_num)
        dataset_generator.visualize_dataset(epoch=0, save_path=vis_path)

    # 从文件加载数据集
    loaded_dataset = TaskOffloadingDataset.load_dataset(dataset_path)

    # 获取PyTorch形式的数据集
    torch_dataset = dataset_generator.get_torch_dataset()
    devices = {}
    for i in range(args.device_num):
        if i < args.device_num * args.malicious:  # 恶意设备
            devices.update({
                i: [
                    0,  # 恶意标志
                    random.randint(3, 7) / 10,  # 较低的安全级别
                    random.randint(8, 10) / 10,  # 能力级别
                ]
            })
        else:  # 正常设备
            devices.update({
                i: [
                    1,  # 正常标志
                    random.randint(5, 10) / 10,  # 较高的安全级别
                    random.randint(8, 10) / 10,  # 能力级别
                ]
            })

    # 初始化邻居管理器和智能体
    neighbor_manager = DynamicNeighborMatrix(args)

    # 创建DQN智能体替换原来的agent
    agent = DQNAgent(args)

    # 开始训练循环
    for epoch in range(args.epochs):
        # 初始化任务记录矩阵
        task_suc = np.zeros([args.device_num, args.device_num])  # 成功任务统计
        task_fail = np.zeros([args.device_num, args.device_num])  # 失败任务统计
        reward_buffer = []  # 用于存储近期奖励

        # 更新邻居矩阵
        neighbor_matrix = neighbor_manager.update_epoch(epoch=0)

        # 获取当前时隙的数据
        epoch_slot_data = get_epoch_slot_data(torch_dataset, epoch, 0)

        # 任务向量
        task_em = epoch_slot_data['task_arrivals']

        # 计算静态安全向量
        static_re = [0 for i in range(args.device_num)]
        for i in range(args.device_num):
            co = 0
            for j in range(args.neighbors_num):
                co += devices[neighbor_matrix[i][j]][1] + devices[neighbor_matrix[i][j]][2]
            # 结合自身安全能力和邻居安全能力计算静态信誉
            static_re[i] = args.se_ca * (devices[i][1] + devices[i][2]) + (1 - args.se_ca) * co / args.neighbors_num

        # 计算动态安全向量 (基于历史交互)
        dynamic_re = (task_suc + 1) / (task_suc + args.su_fa * task_fail + 2)

        # 设置当前状态
        current_state = (dynamic_re, static_re, task_em, neighbor_matrix)

        # 初始化性能指标
        ave_reward = 0
        ave_fail = 0
        ave_en = 0
        pre_fail = 0

        # 时隙循环
        for t in range(args.T):
            # 获取动作
            actions = agent.select_action(dynamic_re, static_re, task_em, neighbor_matrix)

            # 统计信息
            fail_task = 0
            task_num = 0
            e_reward = 0

            # 计算奖励和统计任务执行结果
            for i in range(args.device_num):
                # 确保使用浮点类型比较
                if isinstance(actions, torch.Tensor):
                    not_offload = (actions[i, 0] == 1) or (torch.argmax(actions[i]).item() == 0)
                else:
                    not_offload = (actions[i][0] == 1) or (np.argmax(actions[i]) == 0)

                # 确保task_em的类型正确
                if isinstance(task_em, torch.Tensor):
                    has_task = task_em[i].float() > 0
                else:
                    has_task = task_em[i] > 0

                if not_offload:  # 不卸载
                    if has_task:  # 有任务但不卸载，视为失败
                        fail_task += 1
                        task_num += 1
                    continue
                else:
                    # 获取卸载目标设备索引
                    if isinstance(actions, torch.Tensor):
                        neighbor_index = torch.argmax(actions[i]).item() - 1
                    else:
                        neighbor_index = np.argmax(actions[i]) - 1

                    target_device_id = neighbor_matrix[i][neighbor_index]

                    # 计算能源奖励 (与设备距离成正比)
                    hop_cost = neighbor_index + 1  # 跳数作为能源成本
                    e_reward -= hop_cost
                    task_num += 1
                    if devices[target_device_id][0] == 0:  # 恶意设备
                        fail_task += 1
                        task_fail[i][target_device_id] += 1  # 记录失败
                    else:  # 正常设备
                        task_suc[i][target_device_id] += 1  # 记录成功

            # 计算总奖励 (任务失败率 + 能耗)
            if task_num > 0:
                failure_rate = fail_task / task_num
                # 使用指数惩罚使高失败率受到更严厉的惩罚
                failure_penalty = -100 * np.exp(2.5 * failure_rate)
                energy_reward = e_reward
                reward = failure_penalty + energy_reward
            else:
                reward = e_reward

            reward_buffer.append(reward)
            if len(reward_buffer) > 1000:  # 保持固定大小的缓冲区
                reward_buffer.pop(0)
            reward_mean = sum(reward_buffer) / len(reward_buffer)
            reward_std = math.sqrt(sum((r - reward_mean) ** 2 for r in reward_buffer) / len(reward_buffer)) + 1e-6
            normalized_reward = (reward - reward_mean) / reward_std

            # 更新统计值
            ave_reward += reward
            ave_en -= e_reward * 10
            pre_fail = fail_task / task_num if task_num > 0 else pre_fail
            ave_fail += fail_task / task_num if task_num > 0 else pre_fail

            # 更新邻居矩阵和任务
            neighbor_matrix = neighbor_manager.update_epoch(epoch=t + 1)
            epoch_slot_data = get_epoch_slot_data(torch_dataset, epoch, t + 1)
            task_em = epoch_slot_data['task_arrivals']

            # 重新计算静态信誉
            for i in range(args.device_num):
                co = 0
                for j in range(args.neighbors_num):
                    co += devices[neighbor_matrix[i][j]][1] + devices[neighbor_matrix[i][j]][2]
                static_re[i] = args.se_ca * (devices[i][1] + devices[i][2]) + (1 - args.se_ca) * co / args.neighbors_num

            # 更新动态信誉
            dynamic_re = (task_suc + 1) / (task_suc + args.su_fa * task_fail + 2)

            # 设置下一状态
            next_state = (dynamic_re, static_re, task_em, neighbor_matrix)

            # 判断是否为最后一个时间步
            done = (t == args.T - 1)

            # 将经验存入回放缓冲区
            agent.replay_buffer.push(current_state, actions, reward, next_state, done)

            # 更新当前状态
            current_state = next_state

            # 训练网络
            if len(agent.replay_buffer) >= args.batch_size:
                loss = agent.update(args.batch_size)

        # 记录epoch性能
        logging.info("epoch:{} | 奖励： {} | 任务失败率: {} | 能耗:{}".format(
            epoch, ave_reward / args.T, ave_fail / args.T, ave_en / args.T))

        # 定期保存模型
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            save_path = f"re_model/dqn_model_epoch_{epoch}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'policy_net': agent.policy_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            }, save_path)
            logging.info(f"模型保存至: {save_path}")