import torch
import numpy as np
import argparse
import logging
import os
import sys
import math
from MARL import setup_seed

# 导入我们的模块
from re_util import TaskOffloadingDataset, get_epoch_slot_data, get_reputation_matrix_2
from cross_module import InterServerCommunication, GlobalReputationService
from multi_re_main import EdgeServerManager


# 由于这是整合脚本，我们假设EdgeServerManager已在之前的代码中定义

def calculate_static_reputation(server_manager, server_id, neighbor_matrix, args, global_reputation_service):
    """
    计算静态信誉向量，考虑全局信誉信息

    Args:
        server_manager: 服务器管理器
        server_id: 当前服务器ID
        neighbor_matrix: 邻居矩阵
        args: 参数
        global_reputation_service: 全局信誉服务

    Returns:
        list: 静态信誉向量
    """
    static_re = [0 for i in range(args.device_num)]

    # 获取全局信誉信息
    global_reputations = global_reputation_service.get_server_device_reputations(server_id)

    for i in range(args.device_num):
        co = 0
        for j in range(args.neighbors_num):
            # 获取邻居的全局ID
            global_neighbor_id = server_manager.get_global_device_id(server_id, neighbor_matrix[i][j])
            neighbor_device = server_manager.get_device_info(global_neighbor_id)

            # 结合设备属性和全局信誉
            neighbor_global_rep = global_reputation_service.get_global_reputation(global_neighbor_id)
            neighbor_attr_rep = neighbor_device[1] + neighbor_device[2]  # 安全级别 + 能力级别

            # 融合属性信誉和全局信誉
            neighbor_rep = 0.7 * neighbor_attr_rep + 0.3 * neighbor_global_rep
            co += neighbor_rep

        # 获取当前设备的全局ID和属性
        global_device_id = server_manager.get_global_device_id(server_id, i)
        device = server_manager.get_device_info(global_device_id)
        device_global_rep = global_reputation_service.get_global_reputation(global_device_id)
        device_attr_rep = device[1] + device[2]  # 安全级别 + 能力级别

        # 融合自身属性信誉、全局信誉和邻居信誉
        self_rep = 0.7 * device_attr_rep + 0.3 * device_global_rep
        static_re[i] = args.se_ca * self_rep + (1 - args.se_ca) * co / args.neighbors_num

    return static_re


def prepare_datasets(args, num_servers):
    """准备数据集，为每个服务器生成独立的数据集"""
    datasets = {}

    for server_id in range(num_servers):
        # 为服务器创建专用数据集生成器
        dataset_generator = TaskOffloadingDataset(args)

        # 构建服务器特定的数据集路径
        dataset_path = 're_dataset/server_{}_episodes_{}_poisson-rate_{}_device-num_{}_dataset.pkl'.format(
            server_id, args.epochs * 10, args.task_arrival_prob, args.device_num)

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
                server_id, args.epochs * 10, args.task_arrival_prob, args.device_num)
            dataset_generator.visualize_dataset(epoch=0, save_path=vis_path)

        # 从文件加载数据集
        loaded_dataset = TaskOffloadingDataset.load_dataset(dataset_path)

        # 获取PyTorch形式的数据集
        torch_dataset = dataset_generator.get_torch_dataset()
        datasets[server_id] = torch_dataset

    return datasets


def run_multi_server_system(args):
    """运行多服务器系统主函数"""

    # 配置日志输出
    log_file = "re_log/multi_server_integrated_episodes_{}_servers_{}_malicious_{}_device_{}_task_{}.txt".format(
        args.epochs * 10, args.server_num, args.malicious, args.device_num, args.task_arrival_prob)

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

    logging.info(f"开始多服务器集成系统训练 (服务器数量: {args.server_num}, 每服务器设备数: {args.device_num})")

    # 初始化服务器管理器
    server_manager = EdgeServerManager(args, num_servers=args.server_num)

    # 初始化跨服务器通信模块和全局信誉服务
    inter_server_comm = InterServerCommunication(server_manager, args)
    global_reputation_service = GlobalReputationService(server_manager, args)

    # 准备数据集
    datasets = prepare_datasets(args, args.server_num)

    # 开始训练
    for epoch in range(args.epochs):
        logging.info(f"开始Epoch {epoch * 10}")

        # 每个服务器在本epoch的表现统计
        server_stats = {sid: {
            'ave_reward': 0,
            'ave_fail': 0,
            'ave_en': 0,
            'ave_malicious': 0,
            'cross_server_offloads': 0
        } for sid in range(args.server_num)}

        # 每个服务器单独初始化任务记录矩阵
        server_task_records = {}
        server_reward_buffers = {}

        for server_id in range(args.server_num):
            server = server_manager.get_server(server_id)
            server_task_records[server_id] = {
                'task_suc': np.zeros([args.device_num, args.device_num]),
                'task_fail': np.zeros([args.device_num, args.device_num])
            }
            server_reward_buffers[server_id] = []  # 用于存储近期奖励

        # 按服务器初始化第一个时隙的状态
        server_states = {}
        for server_id in range(args.server_num):
            server = server_manager.get_server(server_id)
            neighbor_matrix = server['neighbor_manager'].update_epoch(epoch=0)

            # 获取当前时隙的数据
            epoch_slot_data = get_epoch_slot_data(datasets[server_id], epoch, 0)

            # 任务向量
            task_em = epoch_slot_data['task_arrivals']

            # 计算静态安全向量（考虑全局信誉信息）
            static_re = calculate_static_reputation(
                server_manager, server_id, neighbor_matrix, args, global_reputation_service)

            # 计算动态安全向量（基于历史交互）
            dynamic_re = (server_task_records[server_id]['task_suc'] + 1) / (
                    server_task_records[server_id]['task_suc'] + args.su_fa * server_task_records[server_id][
                'task_fail'] + 2)

            # 设置当前状态
            server_states[server_id] = (task_em.clone(), dynamic_re.copy(), static_re.copy(), neighbor_matrix.copy())

        # 时隙循环
        for t in range(args.T):
            # 收集各服务器的信誉数据用于全局共享
            local_reputation_data = {}

            # 每个服务器在当前时隙独立运行
            for server_id in range(args.server_num):
                server = server_manager.get_server(server_id)
                agent = server['agent']

                # 获取当前状态
                current_state = server_states[server_id]
                task_em, dynamic_re, static_re, neighbor_matrix = current_state

                # 初始化时隙统计
                fail_task = 0
                task_num = 0
                de_fail = 0  # 决策失败
                ex_fail = 0  # 执行失败
                cross_server_offloads = 0  # 跨服务器卸载次数

                # 生成信誉矩阵（数据驱动部分）
                agent.re_net.eval()
                with torch.no_grad():
                    r_data = agent.re_net(dynamic_re, static_re, neighbor_matrix)

                # 生成基于LLM辅助的信誉矩阵
                r_llm = get_reputation_matrix_2(args)

                # 融合两种信誉
                reputation_matrix = args.re_ba * r_data.detach() + (1 - args.re_ba) * r_llm

                # 计算设备平均信誉
                r_d = torch.mean(reputation_matrix, dim=1)

                # 记录当前时隙信誉状态
                logging.info("服务器:{} | epoch:{} | 时隙: {} | 恶意设备平均信誉: {} | 正常设备平均信誉:{}".format(
                    server_id, epoch * 10, t,
                               sum(r_d[:int(args.device_num * args.malicious)]).item() / max(1,
                                                                                             int(args.device_num * args.malicious)),
                               sum(r_d[int(args.device_num * args.malicious):]).item() / max(1, args.device_num - int(
                                   args.device_num * args.malicious))
                ))

                # 收集本地信誉数据用于全局共享
                local_reputation_data[server_id] = {
                    'reputation': r_d,
                    'device_indices': server['device_indices']
                }

                # 生成信誉掩码
                r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

                # 生成卸载决策
                agent.actor.eval()
                actions = agent.actor(reputation_matrix, task_em, neighbor_matrix, r_mask)

                # 计算奖励和统计任务执行结果
                e_reward = 0
                for i in range(len(actions)):
                    if torch.argmax(actions[i]) == 0:
                        if task_em[i]:  # 有任务但不卸载，视为失败
                            fail_task += 1
                            task_num += 1
                            de_fail += 1
                        continue
                    else:
                        neighbor_index = torch.argmax(actions[i]) - 1

                        # 获取邻居的全局设备ID
                        global_neighbor_id = server_manager.get_global_device_id(
                            server_id, neighbor_matrix[i][neighbor_index])

                        # 确定邻居设备所属服务器
                        neighbor_server_id = server_manager.get_server_for_device(global_neighbor_id)

                        # 能耗成本，考虑信誉因素
                        e_reward -= ((neighbor_index + 1) * (1 - r_d[neighbor_matrix[i][neighbor_index]]))
                        task_num += 1

                        # 判断是否为跨服务器卸载
                        if neighbor_server_id != server_id:
                            # 跨服务器任务卸载
                            cross_server_offloads += 1
                            success = inter_server_comm.handle_cross_server_offload(
                                server_id, i, neighbor_server_id, global_neighbor_id)

                            if not success:
                                fail_task += 1
                                ex_fail += 1
                                # 记录失败
                                server_task_records[server_id]['task_fail'][i][neighbor_matrix[i][neighbor_index]] += 1
                            else:
                                # 记录成功
                                server_task_records[server_id]['task_suc'][i][neighbor_matrix[i][neighbor_index]] += 1

                            # 记录全局交互结果
                            global_device_id = server_manager.get_global_device_id(server_id, i)
                            global_reputation_service.record_interaction(
                                global_device_id, global_neighbor_id, success)

                        else:
                            # 本地服务器内的任务卸载
                            # 获取邻居设备的信息
                            neighbor_device = server_manager.get_device_info(global_neighbor_id)

                            if neighbor_device[0] == 0:  # 恶意设备
                                ex_fail += 1
                                fail_task += 1
                                server_task_records[server_id]['task_fail'][i][neighbor_matrix[i][neighbor_index]] += 1

                                # 记录全局交互结果
                                global_device_id = server_manager.get_global_device_id(server_id, i)
                                global_reputation_service.record_interaction(
                                    global_device_id, global_neighbor_id, False)

                            else:  # 正常设备
                                server_task_records[server_id]['task_suc'][i][neighbor_matrix[i][neighbor_index]] += 1

                                # 记录全局交互结果
                                global_device_id = server_manager.get_global_device_id(server_id, i)
                                global_reputation_service.record_interaction(
                                    global_device_id, global_neighbor_id, True)


import numpy as np
import random
import math
import torch


class DynamicNeighborMatrix:
    """动态维护设备邻居关系的类"""

    def __init__(self, args):
        """
        初始化邻居矩阵管理器

        Args:
            args: 参数命名空间
        """
        self.args = args
        self.device_num = args.device_num
        self.neighbors_num = args.neighbors_num
        self.server_id = args.server_id if hasattr(args, 'server_id') else 0
        self.global_device_offset = self.server_id * self.device_num

        # 当前的邻居矩阵
        self.current_matrix = np.zeros((self.device_num, self.neighbors_num), dtype=int)

        # 邻居更新频率（每n个时隙更新一次）
        self.update_frequency = 10

        # 初始化随机邻居矩阵
        self._initialize_neighbors()

    def _initialize_neighbors(self):
        """初始化随机邻居矩阵"""
        for i in range(self.device_num):
            # 为每个设备随机选择不同的邻居
            # 注意：这里我们先使用本地设备索引，而不是全局设备索引
            neighbors = list(range(self.device_num))
            neighbors.remove(i)  # 移除自身
            if len(neighbors) >= self.neighbors_num:
                selected_neighbors = random.sample(neighbors, self.neighbors_num)
            else:
                # 如果邻居数量不足，则允许重复选择
                selected_neighbors = random.choices(neighbors, k=self.neighbors_num)

            # 存储到邻居矩阵
            self.current_matrix[i] = selected_neighbors

    def update_epoch(self, epoch):
        """
        根据epoch更新邻居矩阵

        Args:
            epoch: 当前的epoch或时隙

        Returns:
            numpy.ndarray: 更新后的邻居矩阵
        """
        # 判断是否需要更新邻居关系
        if epoch % self.update_frequency == 0 and epoch > 0:
            self._update_neighbors()

        return self.current_matrix

    def _update_neighbors(self):
        """动态更新邻居矩阵，保持一定比例的邻居变化"""
        new_matrix = np.copy(self.current_matrix)

        for i in range(self.device_num):
            # 决定是否更新该设备的邻居（50%概率）
            if random.random() < 0.5:
                continue

            # 当前邻居集合
            current_neighbors = set(self.current_matrix[i])

            # 确定要更换的邻居数量（1-2个）
            change_count = random.randint(1, min(2, self.neighbors_num))

            # 随机选择要更换的邻居索引
            change_indices = random.sample(range(self.neighbors_num), change_count)

            # 所有可能的新邻居
            all_possible_neighbors = set(range(self.device_num)) - {i}

            # 选择新邻居
            for idx in change_indices:
                # 尝试找到一个不在当前邻居集中的新邻居
                available_neighbors = list(all_possible_neighbors - current_neighbors)

                if available_neighbors:
                    # 有可用的新邻居
                    new_neighbor = random.choice(available_neighbors)
                    new_matrix[i][idx] = new_neighbor
                    current_neighbors.add(new_neighbor)
                else:
                    # 如果没有可用的新邻居，则随机选择一个（可能与其他邻居重复）
                    new_neighbor = random.choice(list(all_possible_neighbors))
                    new_matrix[i][idx] = new_neighbor

        # 更新当前邻居矩阵
        self.current_matrix = new_matrix

    def get_neighbor_matrix(self):
        """
        获取当前邻居矩阵

        Returns:
            numpy.ndarray: 当前邻居矩阵
        """
        return self.current_matrix

    def get_global_neighbor_matrix(self):
        """
        获取全局索引的邻居矩阵

        Returns:
            numpy.ndarray: 全局索引的邻居矩阵
        """
        global_matrix = np.copy(self.current_matrix)

        # 将本地索引转换为全局索引
        for i in range(self.device_num):
            for j in range(self.neighbors_num):
                global_matrix[i][j] += self.global_device_offset

        return global_matrix

    def get_neighbor_tensor(self):
        """
        获取PyTorch张量形式的邻居矩阵

        Returns:
            torch.Tensor: 邻居矩阵张量
        """
        return torch.tensor(self.current_matrix, dtype=torch.long)