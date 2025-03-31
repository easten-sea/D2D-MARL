import torch
from MARL import xxx_agent, setup_seed
import numpy as np
import argparse
import random
import math
import os
import logging
import sys
import torch.nn.functional as F

from re_util import DynamicNeighborMatrix, TaskOffloadingDataset, get_epoch_slot_data, calculate_transmission_energy, KL,  get_reputation_matrix_2
from re_agent import multi_re_agent

# 检查CUDA是否可用
cuda_enable = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EdgeServerManager:
    """管理多个边缘服务器的类"""

    def __init__(self, args, num_servers=10):
        self.args = args
        self.num_servers = num_servers
        self.servers = []
        self.server_devices = {}  # 每个服务器管理的设备映射
        self.global_device_count = args.device_num * num_servers

        # 初始化设备属性（包括是否为恶意设备）
        self.all_devices = self._initialize_devices()

        # 为每个服务器分配设备
        self._assign_devices_to_servers()

        # 初始化每个服务器的邻居管理器和智能体
        self._initialize_servers()

    def _initialize_devices(self):
        """初始化所有设备的属性"""
        devices = {}
        malicious_count = int(self.global_device_count * self.args.malicious)

        for i in range(self.global_device_count):
            if i < malicious_count:  # 恶意设备
                devices[i] = [
                    0,  # 恶意标志
                    random.randint(3, 7) / 10,  # 较低的安全级别
                    random.randint(8, 10) / 10,  # 能力级别
                ]
            else:  # 正常设备
                devices[i] = [
                    1,  # 正常标志
                    random.randint(5, 10) / 10,  # 较高的安全级别
                    random.randint(8, 10) / 10,  # 能力级别
                ]

        return devices

    def _assign_devices_to_servers(self):
        """将设备分配给各个服务器"""
        for server_id in range(self.num_servers):
            start_idx = server_id * self.args.device_num
            end_idx = start_idx + self.args.device_num

            # 为每个服务器分配一组设备
            self.server_devices[server_id] = list(range(start_idx, end_idx))

    def _initialize_servers(self):
        """初始化每个服务器的邻居管理器和智能体"""
        for server_id in range(self.num_servers):
            # 创建服务器专用的参数副本
            server_args = self._create_server_args(server_id)

            # 创建邻居管理器和智能体
            neighbor_manager = DynamicNeighborMatrix(server_args)
            agent = multi_re_agent(server_args)
            r_d = torch.ones(self.args.device_num)
            C_i = C_i = torch.rand(32) * 0.1  # 初始化认知向量
            server_neighbor = [server_id - 1 if server_id > 0 else self.num_servers - 1,
                               server_id + 1 if server_id < self.num_servers - 1 else 0]

            # 存储服务器信息
            self.servers.append({
                'id': server_id,
                'args': server_args,
                'neighbor_manager': neighbor_manager,
                'agent': agent,
                'device_indices': self.server_devices[server_id],
                'r_d': r_d,
                'neighbors': server_neighbor,
                'C_i': C_i
            })

    def _create_server_args(self, server_id):
        """为每个服务器创建参数副本，确保不同服务器间的隔离"""
        # 复制参数
        server_args = argparse.Namespace(**vars(self.args))

        # 添加服务器特定参数
        server_args.server_id = server_id
        server_args.global_device_offset = server_id * self.args.device_num

        return server_args

    def get_server_local_device_id(self, server_id, global_device_id):
        """将全局设备ID转换为服务器内部的本地设备ID"""
        if global_device_id in self.server_devices[server_id]:
            return global_device_id - (server_id * self.args.device_num)
        return None

    def get_global_device_id(self, server_id, local_device_id):
        """将服务器内部的本地设备ID转换为全局设备ID"""
        return server_id * self.args.device_num + local_device_id

    def get_server_for_device(self, global_device_id):
        """获取管理指定设备的服务器ID"""
        server_id = global_device_id // self.args.device_num
        if 0 <= server_id < self.num_servers:
            return server_id
        return None

    def get_device_info(self, global_device_id):
        """获取设备信息"""
        return self.all_devices.get(global_device_id)

    def get_server(self, server_id):
        """获取服务器对象"""
        if 0 <= server_id < self.num_servers:
            return self.servers[server_id]
        return None


def prepare_datasets(args, num_servers):
    """准备数据集，为每个服务器生成独立的数据集"""
    datasets = {}

    for server_id in range(num_servers):
        # 为服务器创建专用数据集生成器
        dataset_generator = TaskOffloadingDataset(args)

        # 构建服务器特定的数据集路径
        dataset_path = 're_dataset/server_{}_episodes_{}_poisson-rate_{}_device-num_{}_dataset.pkl'.format(
            server_id, args.epochs, args.task_arrival_prob, args.device_num)

        if not os.path.exists(dataset_path):
            # 生成并保存新数据集
            stats = dataset_generator.get_statistics()
            print(f"服务器{server_id}数据集统计信息:")
            for key, value in stats.items():
                if key != 'tasks_per_device':
                    print(f"- {key}: {value}")

            # 保存数据集
            dataset_generator.save_dataset(dataset_path)

            # 可视化第一个epoch的数据
            vis_path = 're_dataset/server_{}_episodes_{}_poisson-rate_{}_device-num_{}_visualization.png'.format(
                server_id, args.epochs, args.task_arrival_prob, args.device_num)
            dataset_generator.visualize_dataset(epoch=0, save_path=vis_path)

        # 从文件加载数据集
        loaded_dataset = TaskOffloadingDataset.load_dataset(dataset_path)

        # 获取PyTorch形式的数据集
        torch_dataset = dataset_generator.get_torch_dataset()
        datasets[server_id] = torch_dataset

    return datasets


def train_multi_server_system(args, server_manager, datasets):
    """训练多服务器系统"""

    for epoch in range(args.epochs):
        # 每个服务器在本epoch的表现统计
        server_stats = {sid: {
            'ave_reward': 0,
            'ave_fail': 0,
            'ave_en': 0,
            'ave_malicious': 0
        } for sid in range(server_manager.num_servers)}

        # 每个服务器单独初始化任务记录矩阵
        server_task_records = {}
        server_reward_buffers = {}

        for server_id in range(server_manager.num_servers):
            server = server_manager.get_server(server_id)
            server_task_records[server_id] = {
                'task_suc': np.zeros([args.device_num, args.device_num]),
                'task_fail': np.zeros([args.device_num, args.device_num])
            }
            server_reward_buffers[server_id] = []  # 用于存储近期奖励

        # 按服务器初始化第一个时隙的状态
        server_states = {}
        for server_id in range(server_manager.num_servers):
            server = server_manager.get_server(server_id)
            neighbor_matrix = server['neighbor_manager'].update_epoch(epoch=0)

            # 获取当前时隙的数据
            epoch_slot_data = get_epoch_slot_data(datasets[server_id], epoch, 0)

            # 任务向量
            task_em = epoch_slot_data['task_arrivals']
            task_size = epoch_slot_data['task_sizes']

            # 计算静态安全向量
            static_re = [0 for i in range(args.device_num)]
            for i in range(args.device_num):
                co = 0
                for j in range(args.neighbors_num):
                    # 获取邻居的全局ID
                    global_neighbor_id = server_manager.get_global_device_id(server_id, neighbor_matrix[i][j])
                    neighbor_device = server_manager.get_device_info(global_neighbor_id)
                    co += neighbor_device[1] + neighbor_device[2]

                # 获取当前设备的全局ID
                global_device_id = server_manager.get_global_device_id(server_id, i)
                device = server_manager.get_device_info(global_device_id)

                # 结合自身安全能力和邻居安全能力计算静态信誉
                static_re[i] = args.se_ca * (device[1] + device[2]) + (1 - args.se_ca) * co / args.neighbors_num

            # 计算动态安全向量（基于历史交互）
            dynamic_re = (server_task_records[server_id]['task_suc'] + 1) / (
                        server_task_records[server_id]['task_suc'] + args.su_fa * server_task_records[server_id][
                    'task_fail'] + 2)

            # 初始化邻居服务器的设备信誉向量
            ex_r_d = []
            for neighbor_server_id in server['neighbors']:
                neighbor_server = server_manager.get_server(neighbor_server_id)
                ex_r_d.append(neighbor_server['r_d'].clone())  # 使用clone确保获取的是副本

            # 初始化邻居服务器的认知向量
            ex_C_i = []
            for neighbor_server_id in server['neighbors']:
                neighbor_server = server_manager.get_server(neighbor_server_id)
                ex_C_i.append(neighbor_server['C_i'].clone())  # 使用clone确保获取的是副本

            # 设置当前状态（添加ex_r_d和ex_C_i作为状态的一部分）
            server_states[server_id] = (
                task_em.clone(),
                task_size.clone(),
                dynamic_re.copy(),
                static_re.copy(),
                neighbor_matrix.copy(),
                [r.clone() for r in ex_r_d],
                [c.clone() for c in ex_C_i]
            )

        # 时隙循环
        for t in range(args.T):
            # 每个服务器在当前时隙独立运行
            for server_id in range(server_manager.num_servers):
                server = server_manager.get_server(server_id)
                agent = server['agent']

                # 获取当前状态
                current_state = server_states[server_id]
                task_em, task_size, dynamic_re, static_re, neighbor_matrix, ex_r_d, ex_C_i = current_state  # 解包状态，包括ex_r_d和ex_C_i

                # 初始化时隙统计
                fail_task = 0
                task_num = 0
                de_fail = 0  # 决策失败
                ex_fail = 0  # 执行失败

                # 生成信誉矩阵（数据驱动部分）
                agent.re_net.eval()
                with torch.no_grad():
                    r_data = agent.re_net(dynamic_re, static_re, neighbor_matrix)
                r_llm = get_reputation_matrix_2(args)
                reputation_matrix = args.re_ba * r_data.detach() + (1 - args.re_ba) * r_llm
                r_d = torch.mean(reputation_matrix, dim=1)
                server['r_d'] = r_d.clone()  # 使用clone确保保存的是副本

                # 使用信誉感知网络处理信誉信息
                agent.re_aware.eval()
                with torch.no_grad():
                    _, C_i, _, _ = agent.re_aware(reputation_matrix, r_d, ex_r_d)
                    server['C_i'] = C_i.clone()  # 更新服务器的认知向量

                    # 使用存储的邻居认知向量计算信誉感知
                    _, reputation_aware, _, _ = agent.re_aware(reputation_matrix, r_d, ex_r_d)

                # 生成信誉掩码
                r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

                # 生成卸载决策
                agent.actor.eval()
                actions = agent.actor(reputation_matrix, reputation_aware, task_em, neighbor_matrix, r_mask)
                # actions[0]是设备卸载决策，大小为args.device_num * (args.neighbors_num + 2)
                # actions[1]是服务器选择决策，大小为args.device_num * 2
                # 计算奖励和统计任务执行结果
                e_reward = 0
                for i in range(len(actions[0])):
                    # 如果没有任务请求，跳过
                    if not task_em[i]:
                        continue
                    task_num += 1  # 有任务请求
                    # 获取设备的决策
                    device_decision = torch.argmax(actions[0][i])
                    # 决策结果处理:
                    # 如果决策是不卸载(index = args.neighbors_num + 1)
                    if device_decision == args.neighbors_num + 1:
                        fail_task += 1
                        continue
                    # 如果决策是向相邻服务器卸载(index = args.neighbors_num)
                    elif device_decision == args.neighbors_num:
                        # 确定卸载到哪个相邻服务器
                        server_choice = torch.argmax(actions[1][i])
                        target_server_id = server['neighbors'][server_choice]

                        # 获取目标服务器的平均信誉
                        target_server = server_manager.get_server(target_server_id)
                        target_server_reputation = torch.mean(target_server['r_d'])
                        # 计算能耗与信誉相关
                        energy_cost = calculate_transmission_energy(args.neighbors_num + 1, task_size[i]) * (1 - target_server_reputation)
                        # energy_cost = (args.neighbors_num + 1)
                        e_reward -= energy_cost
                        # 根据目标服务器的平均信誉决定任务是否成功
                        if torch.rand(1) > target_server_reputation:
                            fail_task += 1
                    # 否则，向本服务器管理的邻居设备卸载
                    else:
                        # 获取目标设备的索引
                        neighbor_index = device_decision
                        neighbor_device_local_id = neighbor_matrix[i][neighbor_index]
                        # 计算能耗 - 与邻居距离和信誉相关
                        energy_cost = calculate_transmission_energy(neighbor_index + 1, task_size[i]) * (1 - r_d[neighbor_device_local_id])
                        e_reward -= energy_cost
                        # 获取邻居设备的全局ID
                        global_neighbor_id = server_manager.get_global_device_id(server_id, neighbor_device_local_id)
                        # 获取邻居设备信息
                        neighbor_device = server_manager.get_device_info(global_neighbor_id)
                        # 检查是否是恶意设备
                        if neighbor_device[0] == 0:  # 恶意设备
                            ex_fail += 1
                            fail_task += 1
                            server_task_records[server_id]['task_fail'][i][neighbor_device_local_id] += 1
                        else:
                            server_task_records[server_id]['task_suc'][i][neighbor_device_local_id] += 1
                reputation_diff_penalty = 0.0
                if ex_r_d:
                    for neighbor_r_d in ex_r_d:
                        server_rep_mean = torch.mean(server['r_d'])
                        neighbor_rep_mean = torch.mean(neighbor_r_d)
                        avg_rep = (server_rep_mean + neighbor_rep_mean) / 2
                        diff = torch.abs(server_rep_mean - neighbor_rep_mean)
                        reputation_diff_penalty -= avg_rep * diff

                if task_num > 0:
                    failure_rate = fail_task / task_num
                    failure_penalty = -100 * np.exp(2.5 * failure_rate)
                    energy_reward = e_reward
                    reward = failure_penalty + energy_reward + reputation_diff_penalty
                else:
                    reward = e_reward + reputation_diff_penalty


                # 更新奖励缓冲区
                server_reward_buffers[server_id].append(reward)
                if len(server_reward_buffers[server_id]) > 1000:
                    server_reward_buffers[server_id].pop(0)

                # 计算归一化奖励
                reward_mean = sum(server_reward_buffers[server_id]) / len(server_reward_buffers[server_id])
                reward_std = math.sqrt(sum((r - reward_mean) ** 2 for r in server_reward_buffers[server_id]) / len(
                    server_reward_buffers[server_id])) + 1e-6
                normalized_reward = (reward - reward_mean) / reward_std

                # 更新统计值
                server_stats[server_id]['ave_reward'] += normalized_reward
                server_stats[server_id]['ave_en'] -= e_reward * 10
                server_stats[server_id]['ave_fail'] += fail_task / task_num if task_num > 0 else 0
                server_stats[server_id]['ave_malicious'] += ex_fail / task_num if task_num > 0 else 0

                # 更新邻居矩阵
                next_neighbor_matrix = server['neighbor_manager'].update_epoch(epoch=t + 1)

                # 获取下一时隙的数据
                epoch_slot_data = get_epoch_slot_data(datasets[server_id], epoch, t + 1)
                next_task_em = epoch_slot_data['task_arrivals']
                next_task_size = epoch_slot_data['task_sizes']

                # 重新计算静态信誉
                next_static_re = [0 for i in range(args.device_num)]
                for i in range(args.device_num):
                    co = 0
                    for j in range(args.neighbors_num):
                        global_neighbor_id = server_manager.get_global_device_id(server_id, next_neighbor_matrix[i][j])
                        neighbor_device = server_manager.get_device_info(global_neighbor_id)
                        co += neighbor_device[1] + neighbor_device[2]

                    # 获取当前设备的全局ID
                    global_device_id = server_manager.get_global_device_id(server_id, i)
                    device = server_manager.get_device_info(global_device_id)

                    # 结合自身安全能力和邻居安全能力计算静态信誉
                    next_static_re[i] = args.se_ca * (device[1] + device[2]) + (1 - args.se_ca) * co / args.neighbors_num

                # 更新动态信誉
                next_dynamic_re = (server_task_records[server_id]['task_suc'] + 1) / (
                        server_task_records[server_id]['task_suc'] + args.su_fa * server_task_records[server_id][
                    'task_fail'] + 2)

                # 获取邻居服务器最新的设备信誉向量
                next_ex_r_d = []
                for neighbor_server_id in server['neighbors']:
                    neighbor_server = server_manager.get_server(neighbor_server_id)
                    next_ex_r_d.append(neighbor_server['r_d'].clone())  # 使用clone确保获取的是副本

                # 获取邻居服务器最新的认知向量
                next_ex_C_i = []
                for neighbor_server_id in server['neighbors']:
                    neighbor_server = server_manager.get_server(neighbor_server_id)
                    next_ex_C_i.append(neighbor_server['C_i'].clone())  # 使用clone确保获取的是副本

                # 设置下一状态（包括邻居服务器的设备信誉向量和认知向量）
                next_state = (
                    next_task_em.clone(),
                    next_task_size.clone(),
                    next_dynamic_re.copy(),
                    next_static_re.copy(),
                    next_neighbor_matrix.copy(),
                    [r.clone() for r in next_ex_r_d],  # 确保存储的是副本
                    [c.clone() for c in next_ex_C_i]  # 确保存储的是副本
                )

                # 判断是否为最后一个时间步
                done = (t == args.T - 1)

                # 将经验存入回放缓冲区
                agent.replay_buffer.push(current_state, actions, reward, next_state, done)

                # 更新当前状态
                server_states[server_id] = next_state

                # 存储当前设备信誉向量和认知向量，供其他服务器使用
                server['r_d'] = r_d.clone()
                server['C_i'] = C_i.clone()

        # 记录每个服务器的epoch性能
        for server_id in range(server_manager.num_servers):
            stats = server_stats[server_id]
            logging.info("服务器:{} | epoch:{} | 奖励: {} | 任务失败率: {} | 能耗:{} ".format(
                server_id, epoch,
                           stats['ave_reward'] / args.T,
                           stats['ave_fail'] / args.T,
                           stats['ave_en'] / args.T
            ))

        # 训练每个服务器的智能体
        for server_id in range(server_manager.num_servers):
            server = server_manager.get_server(server_id)
            agent = server['agent']

            if len(agent.replay_buffer) >= args.batch_size:
                logging.info(f"服务器:{server_id} | Epoch {epoch} 完成，开始训练...")

                # 训练次数
                train_iterations = 5
                total_actor_loss = 0
                total_critic_loss = 0
                total_reputation_loss = 0
                total_reputation_aware_loss = 0

                for _ in range(train_iterations):
                    # 从回放缓冲区采样
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = agent.replay_buffer.sample(int(args.batch_size))

                    # 转换为张量
                    reward_tensor = torch.FloatTensor(reward_batch)
                    done_tensor = torch.FloatTensor(done_batch)

                    # ==================== 训练Critic网络 ====================
                    agent.critic.train()
                    critic_values = []
                    next_values = []
                    torch.autograd.set_detect_anomaly(True)

                    for i in range(len(state_batch)):
                        # 提取状态信息，包括ex_r_d和ex_C_i
                        task_em, _, dynamic_re, static_re, neighbor_matrix, stored_ex_r_d, stored_ex_C_i = state_batch[i]
                        next_task_em, _, next_dynamic_re, next_static_re, next_neighbor_matrix, next_stored_ex_r_d, next_stored_ex_C_i = next_state_batch[i]

                        # 提取动作信息
                        current_action = action_batch[i]
                        for i in range(len(current_action)):
                            current_action[i] = current_action[i].clone().detach()

                        # 计算当前状态的信誉矩阵和掩码
                        with torch.no_grad():
                            r_data = agent.re_net(dynamic_re, static_re, neighbor_matrix)
                            r_llm = get_reputation_matrix_2(args)
                            reputation_matrix = args.re_ba * r_data + (1 - args.re_ba) * r_llm
                            r_d = torch.mean(reputation_matrix, dim=1)
                            r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

                            # 使用存储的邻居服务器信誉向量
                            # 生成信誉感知向量
                            _, reputation_aware, _, _ = agent.re_aware(reputation_matrix, r_d, stored_ex_r_d)

                            # 计算下一状态的信誉矩阵和掩码
                            next_r_data = agent.re_net(next_dynamic_re, next_static_re, next_neighbor_matrix)
                            next_r_llm = get_reputation_matrix_2(args)
                            next_reputation_matrix = args.re_ba * next_r_data + (1 - args.re_ba) * next_r_llm
                            next_r_d = torch.mean(next_reputation_matrix, dim=1)
                            next_r_mask = torch.sigmoid((next_r_d - args.re_mask) * 10)

                            # 使用存储的下一状态的邻居服务器信誉向量
                            _, next_reputation_aware, _, _ = agent.re_aware(next_reputation_matrix, next_r_d, next_stored_ex_r_d)

                            # 获取下一状态的最佳动作
                            next_actions = agent.actor(next_reputation_matrix, next_reputation_aware, next_task_em,
                                                       next_neighbor_matrix, next_r_mask)

                        # 获取当前Q值
                        q_value = agent.critic(dynamic_re, static_re, task_em, neighbor_matrix, r_mask, reputation_aware, current_action)

                        # 获取下一状态的Q值
                        with torch.no_grad():
                            next_q_value = agent.critic(next_dynamic_re, next_static_re, next_task_em,
                                                        next_neighbor_matrix,
                                                        next_r_mask, next_reputation_aware, next_actions)

                        critic_values.append(q_value)
                        next_values.append(next_q_value)

                    # 转换为张量 - 确保一致的形状
                    critic_values_tensor = torch.stack(critic_values).squeeze(-1)
                    next_values_tensor = torch.stack(next_values).squeeze(-1)

                    # 计算目标Q值 (TD目标)
                    target_q_values = reward_tensor + (1 - done_tensor) * args.gamma * next_values_tensor

                    # 计算TD误差 - 确保维度匹配
                    critic_loss = F.smooth_l1_loss(critic_values_tensor, target_q_values)

                    # 更新Critic
                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)
                    agent.critic_optimizer.step()
                    total_critic_loss += critic_loss.item()

                    # ==================== 训练Reputation网络 ====================
                    agent.re_net.train()
                    reputation_loss = 0
                    for i in range(len(state_batch)):
                        task_em, _, dynamic_re, static_re, neighbor_matrix, _, _ = state_batch[i]

                        # 确保输入是张量
                        if not isinstance(dynamic_re, torch.Tensor):
                            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32)
                        if not isinstance(static_re, torch.Tensor):
                            static_re = torch.tensor(static_re, dtype=torch.float32)
                        if not isinstance(neighbor_matrix, torch.Tensor):
                            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32)

                        # 生成信誉矩阵 (数据驱动)
                        reputation_matrix = agent.re_net(dynamic_re, static_re, neighbor_matrix)
                        r_llm = get_reputation_matrix_2(args)

                        reputation_loss += KL(reputation_matrix, r_llm)
                    reputation_loss = reputation_loss / len(state_batch)

                    # 反向传播
                    agent.re_optimizer.zero_grad()
                    reputation_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.re_net.parameters(), max_norm=5.0)
                    agent.re_optimizer.step()
                    total_reputation_loss += reputation_loss.item()

                    # ==================== 训练ReputationAware网络 ====================
                    agent.re_aware.train()
                    reputation_aware_loss = 0

                    for i in range(len(state_batch)):
                        task_em, _, dynamic_re, static_re, neighbor_matrix, stored_ex_r_d, stored_ex_C_i = state_batch[i]

                        # 生成信誉矩阵
                        r_data = agent.re_net(dynamic_re, static_re, neighbor_matrix)
                        r_llm = get_reputation_matrix_2(args)
                        reputation_matrix = args.re_ba * r_data + (1 - args.re_ba) * r_llm
                        r_d = torch.mean(reputation_matrix, dim=1)

                        # 使用存储的邻居服务器信誉向量和认知向量
                        # 使用信誉感知网络处理，得到重建的信誉矩阵和VAE损失
                        reconstructed_R, reputation_aware, C_i_mean, C_i_logvar = agent.re_aware(reputation_matrix, r_d, stored_ex_r_d)

                        # 计算VAE损失（重建损失和KL散度），包括邻居认知向量的KL损失
                        batch_reputation_aware_loss, _, kl_loss = agent.re_aware.compute_loss(reputation_matrix, reconstructed_R, reputation_aware, stored_ex_C_i)

                        reputation_aware_loss += batch_reputation_aware_loss

                    reputation_aware_loss = reputation_aware_loss / len(state_batch)

                    # 反向传播
                    agent.re_aw_optimizer.zero_grad()
                    reputation_aware_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.re_aware.parameters(), max_norm=5.0)
                    agent.re_aw_optimizer.step()
                    total_reputation_aware_loss += reputation_aware_loss.item()

                    # ==================== 训练Actor网络 ====================
                    agent.actor.train()
                    actor_loss = 0

                    for i in range(len(state_batch)):
                        task_em, _, dynamic_re, static_re, neighbor_matrix, stored_ex_r_d, _ = state_batch[i]

                        # 确保输入是张量
                        if not isinstance(dynamic_re, torch.Tensor):
                            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32)
                        if not isinstance(static_re, torch.Tensor):
                            static_re = torch.tensor(static_re, dtype=torch.float32)
                        if not isinstance(task_em, torch.Tensor):
                            task_em = torch.tensor(task_em, dtype=torch.float32)
                        if not isinstance(neighbor_matrix, torch.Tensor):
                            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32)

                        # 生成信誉矩阵
                        r_data = agent.re_net(dynamic_re, static_re, neighbor_matrix)
                        r_llm = get_reputation_matrix_2(args)
                        reputation_matrix = args.re_ba * r_data + (1 - args.re_ba) * r_llm

                        # 计算信誉掩码
                        r_d = torch.mean(reputation_matrix, dim=1)
                        r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

                        # 使用存储的邻居服务器信誉向量，确保与记录经验时保持一致
                        _, reputation_aware, _, _ = agent.re_aware(reputation_matrix, r_d, stored_ex_r_d)

                        # 获取Actor的当前策略动作
                        current_actions = agent.actor(reputation_matrix, reputation_aware, task_em, neighbor_matrix, r_mask)

                        # 获取当前策略的Q值
                        agent.critic.eval()
                        q_value = agent.critic(dynamic_re, static_re, task_em, neighbor_matrix, r_mask, reputation_aware, current_actions)

                        # 策略损失 = -Q值 (我们希望最大化Q值，因此取负)
                        batch_policy_loss = -q_value.mean()
                        actor_loss += batch_policy_loss

                    # 每批样本的平均损失
                    actor_loss = actor_loss / len(state_batch)

                    # 反向传播
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
                    agent.actor_optimizer.step()

                    # 记录损失
                    total_actor_loss += actor_loss.item()

                # 记录训练结果
                avg_actor_loss = total_actor_loss / train_iterations
                avg_critic_loss = total_critic_loss / train_iterations
                avg_reputation_loss = total_reputation_loss / train_iterations
                avg_reputation_aware_loss = total_reputation_aware_loss / train_iterations

                logging.info(f"服务器:{server_id} | Epoch {epoch} 训练完成:")
                logging.info(f"  - Actor损失: {avg_actor_loss:.4f}")
                logging.info(f"  - Critic损失: {avg_critic_loss:.4f}")
                logging.info(f"  - Reputation损失: {avg_reputation_loss:.4f}")
                logging.info(f"  - ReputationAware损失: {avg_reputation_aware_loss:.4f}")

        # 定期保存模型 (每个服务器单独保存)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            for server_id in range(server_manager.num_servers):
                server = server_manager.get_server(server_id)
                agent = server['agent']

                save_path = f"re_model/server_{server_id}_model_device_{args.device_num}_task_{args.task_arrival_prob}_malicious_{args.malicious}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    're_net': agent.re_net.state_dict(),
                    're_aware': agent.re_aware.state_dict(),
                }, save_path)
                logging.info(f"服务器:{server_id} | 模型保存至: {save_path}")

    # 返回服务器性能统计
    return server_stats


if __name__ == '__main__':
    # 参数解析
    ap = argparse.ArgumentParser(description='args for Multi-Server MARL')
    ap.add_argument('--alth', type=str, default='our', help='运行的算法')
    ap.add_argument('--epochs', type=int, default=2000, help='轨迹样本数')
    ap.add_argument('--device_num', type=int, default=30, help='每个服务器管理的设备数量')
    ap.add_argument('--server_num', type=int, default=10, help='边缘服务器数量')
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
    log_file = "re_log/multi_server_episodes_{}_servers_{}_malicious_{}_device_{}_task_{}.txt".format(
        args.epochs, args.server_num, args.malicious, args.device_num, args.task_arrival_prob)

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

    logging.info(f"开始多服务器系统训练 (服务器数量: {args.server_num}, 每服务器设备数: {args.device_num})")

    # 初始化服务器管理器
    server_manager = EdgeServerManager(args, num_servers=args.server_num)

    # 准备数据集
    datasets = prepare_datasets(args, args.server_num)

    # 开始训练
    server_stats = train_multi_server_system(args, server_manager, datasets)

    # 打印最终性能统计
    logging.info("训练完成！最终性能统计:")
    for server_id, stats in server_stats.items():
        logging.info(f"服务器 {server_id}:")
        logging.info(f"  - 平均奖励: {stats['ave_reward'] / args.T:.4f}")
        logging.info(f"  - 平均任务失败率: {stats['ave_fail'] / args.T:.4f}")
        logging.info(f"  - 平均能耗: {stats['ave_en'] / args.T:.4f}")