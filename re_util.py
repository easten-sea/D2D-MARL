
import requests
import json
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt


class DynamicNeighborMatrix:
    def __init__(self, args):
        """
        初始化动态邻居矩阵生成器

        参数:
            args: 包含配置的对象
                - device_num: 设备总数
                - neighbors_num: 每个设备的邻居数量
        """
        self.device_num = args.device_num
        self.neighbors_num = args.neighbors_num

        # 确保邻居数不超过设备总数-1
        if self.neighbors_num >= self.device_num:
            self.neighbors_num = self.device_num - 1
            print(f"警告: 邻居数量超过可能的最大值，已调整为 {self.neighbors_num}")

        # 初始化邻居矩阵
        self.neighbor_matrix = self._generate_neighbor_matrix()

    def _generate_neighbor_matrix(self):
        """
        生成一个新的邻居关系矩阵

        返回:
            neighbor_matrix: 形状为 [device_num, neighbors_num] 的矩阵
                             表示每个设备的邻居设备编号
        """
        neighbor_matrix = np.zeros((self.device_num, self.neighbors_num), dtype=np.int64)

        for device_id in range(self.device_num):
            # 创建可能的邻居列表（除了设备自身）
            possible_neighbors = list(range(self.device_num))
            possible_neighbors.remove(device_id)

            # 随机选择指定数量的邻居
            selected_neighbors = np.random.choice(
                possible_neighbors,
                size=self.neighbors_num,
                replace=False
            )

            neighbor_matrix[device_id] = selected_neighbors

        return neighbor_matrix

    def update_epoch(self, epoch=None, change_prob=0.3):
        """
        在新的epoch开始时更新邻居矩阵

        参数:
            epoch: 当前epoch数（可用于基于epoch的特定变化规则）
            change_prob: 每个设备改变邻居的概率

        返回:
            neighbor_matrix: 更新后的邻居矩阵
        """
        # 决定哪些设备需要更新邻居
        devices_to_update = np.random.random(self.device_num) < change_prob

        for device_id in range(self.device_num):
            if devices_to_update[device_id]:
                # 创建可能的邻居列表（除了设备自身）
                possible_neighbors = list(range(self.device_num))
                possible_neighbors.remove(device_id)

                # 随机选择指定数量的邻居
                selected_neighbors = np.random.choice(
                    possible_neighbors,
                    size=self.neighbors_num,
                    replace=False
                )

                self.neighbor_matrix[device_id] = selected_neighbors

        return self.neighbor_matrix

    def get_neighbor_matrix(self):
        """
        获取当前的邻居矩阵

        返回:
            neighbor_matrix: 当前的邻居矩阵，形状为 [device_num, neighbors_num]
        """
        return self.neighbor_matrix

    def get_torch_neighbor_matrix(self, device='cpu'):
        """
        获取PyTorch张量形式的邻居矩阵

        参数:
            device: PyTorch设备（如'cpu'或'cuda:0'）

        返回:
            neighbor_matrix: PyTorch张量形式的邻居矩阵
        """
        return torch.tensor(self.neighbor_matrix, dtype=torch.long, device=device)



class TaskOffloadingDataset:
    def __init__(self, args):
        """
        初始化任务卸载数据集生成器

        参数:
            args: 包含配置的对象
                - device_num: 设备总数
                - epochs: 总epoch数
                - T: 每个epoch中的时隙数
                - task_arrival_prob: 每个时隙任务到达的概率 (默认0.3)
                - task_size_range: 任务大小范围 [min, max] (默认[50, 200])，单位KB
                - max_tolerance_range: 最大容忍时隙范围 [min, max] (默认[5, 15])
                - max_hops_range: 最大容忍跳数范围 [min, max] (默认[1, 3])
                - seed: 随机种子 (可选)
        """
        self.device_num = args.device_num
        self.epochs = args.epochs
        self.T = args.T + 1

        # 设置默认值
        self.task_arrival_prob = getattr(args, 'task_arrival_prob', 0.3)
        self.task_size_range = getattr(args, 'task_size_range', [50, 200])
        self.max_tolerance_range = getattr(args, 'max_tolerance_range', [5, 15])
        self.max_hops_range = getattr(args, 'max_hops_range', [1, 3])

        # 设置随机种子以保证可重复性（如果提供）
        if hasattr(args, 'seed'):
            np.random.seed(args.seed)

        # 初始化数据集
        self.dataset = self._generate_dataset()

    def _generate_dataset(self):
        """
        生成整个数据集

        返回:
            dataset: 字典，包含所有数据
                {
                    'task_arrivals': 形状为 [epochs, T, device_num] 的布尔数组，
                                    表示每个epoch、时隙、设备是否有任务到达
                    'task_sizes': 形状为 [epochs, T, device_num] 的数组，
                                表示任务大小（KB），无任务时为0
                    'max_tolerance_slots': 形状为 [epochs, T, device_num] 的数组，
                                        表示最大容忍时隙，无任务时为0
                    'max_tolerance_hops': 形状为 [epochs, T, device_num] 的数组，
                                        表示最大容忍跳数，无任务时为0
                }
        """
        # 初始化数据结构
        task_arrivals = np.zeros((self.epochs, self.T, self.device_num), dtype=bool)
        task_sizes = np.zeros((self.epochs, self.T, self.device_num), dtype=np.float32)
        max_tolerance_slots = np.zeros((self.epochs, self.T, self.device_num), dtype=np.int32)
        max_tolerance_hops = np.zeros((self.epochs, self.T, self.device_num), dtype=np.int32)

        # 为每个epoch、时隙和设备生成任务
        for epoch in range(self.epochs):
            for t in range(self.T):
                # 确定哪些设备在当前时隙有任务到达
                arrivals = np.random.random(self.device_num) < self.task_arrival_prob
                task_arrivals[epoch, t] = arrivals

                # 为有任务的设备生成任务参数
                for device_id in range(self.device_num):
                    if arrivals[device_id]:
                        # 生成任务大小（KB）
                        task_sizes[epoch, t, device_id] = np.random.uniform(
                            self.task_size_range[0],
                            self.task_size_range[1]
                        )

                        # 生成最大容忍时隙
                        max_tolerance_slots[epoch, t, device_id] = np.random.randint(
                            self.max_tolerance_range[0],
                            self.max_tolerance_range[1] + 1
                        )

                        # 生成最大容忍跳数
                        max_tolerance_hops[epoch, t, device_id] = np.random.randint(
                            self.max_hops_range[0],
                            self.max_hops_range[1] + 1
                        )

        # 封装为字典
        dataset = {
            'task_arrivals': task_arrivals,
            'task_sizes': task_sizes,
            'max_tolerance_slots': max_tolerance_slots,
            'max_tolerance_hops': max_tolerance_hops
        }

        return dataset

    def save_dataset(self, filepath='task_offloading_dataset.pkl'):
        """
        保存数据集到文件

        参数:
            filepath: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # 保存数据集
        with open(filepath, 'wb') as f:
            pickle.dump(self.dataset, f)

        print(f"数据集已保存到: {filepath}")

    @staticmethod
    def load_dataset(filepath):
        """
        从文件加载数据集

        参数:
            filepath: 文件路径

        返回:
            dataset: 加载的数据集
        """
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)

        print(f"数据集已从 {filepath} 加载")
        return dataset

    def get_torch_dataset(self, device='cpu'):
        """
        获取PyTorch张量形式的数据集

        参数:
            device: PyTorch设备（如'cpu'或'cuda:0'）

        返回:
            torch_dataset: PyTorch张量形式的数据集
        """
        torch_dataset = {
            'task_arrivals': torch.tensor(self.dataset['task_arrivals'], dtype=torch.bool, device=device),
            'task_sizes': torch.tensor(self.dataset['task_sizes'], dtype=torch.float32, device=device),
            'max_tolerance_slots': torch.tensor(self.dataset['max_tolerance_slots'], dtype=torch.long, device=device),
            'max_tolerance_hops': torch.tensor(self.dataset['max_tolerance_hops'], dtype=torch.long, device=device)
        }

        return torch_dataset

    def visualize_dataset(self, epoch=0, save_path=None):
        """
        可视化特定epoch的数据集，展示任务到达情况

        参数:
            epoch: 要可视化的epoch索引
            save_path: 保存图表的路径（可选）
        """
        task_arrivals = self.dataset['task_arrivals'][epoch]

        plt.figure(figsize=(12, 8))
        plt.imshow(task_arrivals.T, cmap='Blues', aspect='auto')
        plt.colorbar(label='Task Arrival')
        plt.xlabel('Time Slot')
        plt.ylabel('Device ID')
        # plt.title(f'Epoch {epoch} Task Arrival')

        # 添加网格线
        plt.grid(False)

        if save_path:
            plt.savefig(save_path)
            print(f"可视化图表已保存到: {save_path}")

        plt.show()

    def get_statistics(self):
        """
        计算并返回数据集的统计信息

        返回:
            stats: 包含统计信息的字典
        """
        task_arrivals = self.dataset['task_arrivals']
        task_sizes = self.dataset['task_sizes']

        # 计算总任务数
        total_tasks = np.sum(task_arrivals)

        # 计算每个设备的平均任务数
        tasks_per_device = np.sum(task_arrivals, axis=(0, 1))
        avg_tasks_per_device = np.mean(tasks_per_device)

        # 计算每个时隙的平均任务数
        tasks_per_slot = np.sum(task_arrivals, axis=2)
        avg_tasks_per_slot = np.mean(tasks_per_slot)

        # 计算平均任务大小
        avg_task_size = np.sum(task_sizes) / total_tasks if total_tasks > 0 else 0

        stats = {
            'total_tasks': total_tasks,
            'avg_tasks_per_device': avg_tasks_per_device,
            'avg_tasks_per_slot': avg_tasks_per_slot,
            'avg_task_size': avg_task_size,
            'tasks_per_device': tasks_per_device
        }

        return stats

    def get_epoch_data(self, epoch):
        """
        获取特定epoch的数据

        参数:
            epoch: epoch索引

        返回:
            epoch_data: 该epoch的数据字典
        """
        if epoch < 0 or epoch >= self.epochs:
            raise ValueError(f"Epoch索引 {epoch} 超出范围 [0, {self.epochs - 1}]")

        epoch_data = {
            'task_arrivals': self.dataset['task_arrivals'][epoch],
            'task_sizes': self.dataset['task_sizes'][epoch],
            'max_tolerance_slots': self.dataset['max_tolerance_slots'][epoch],
            'max_tolerance_hops': self.dataset['max_tolerance_hops'][epoch]
        }

        return epoch_data



def get_epoch_slot_data(torch_dataset, epoch, t):
    epoch_slot_data = {
        'task_arrivals': torch_dataset['task_arrivals'][epoch, t],
        'task_sizes': torch_dataset['task_sizes'][epoch, t],
        'max_tolerance_slots': torch_dataset['max_tolerance_slots'][epoch, t],
        'max_tolerance_hops': torch_dataset['max_tolerance_hops'][epoch, t]
    }
    return epoch_slot_data


def get_one_hot_actions(actor_output, task_em, args):
    """
    Convert actor network output to one-hot action vectors.

    Args:
        actor_output: Tensor of shape [device_num, neighbors_num + 1] containing action probabilities
        task_em: Tensor of shape [device_num] indicating which devices have tasks
        args: Arguments containing configuration parameters

    Returns:
        actions: Tensor of shape [device_num, device_num + 1] containing one-hot action vectors
    """
    # Initialize output tensor with zeros
    # Note: We need device_num + 1 columns to handle all possible offload targets plus keeping locally
    actions = torch.zeros(args.device_num, args.device_num + 1)

    # Count tasks
    task_count = torch.sum(task_em).item()

    # Add debug logging
    if task_count == 0:
        print("WARNING: No tasks in current batch! Generating artificial tasks.")
        # Generate some artificial tasks for training
        random_devices = random.sample(range(args.device_num), max(1, int(args.device_num * 0.3)))
        for device_id in random_devices:
            task_em[device_id] = 1.0

    for device_id in range(args.device_num):
        if task_em[device_id]:  # Only process devices with tasks
            # Get probabilities for this device's actions
            action_probs = actor_output[device_id]

            # If all actions are masked out or very low probability
            if torch.max(action_probs) < 0.1:
                # Force some offloading for training
                valid_actions = list(range(1, min(args.neighbors_num + 1, len(action_probs))))
                if valid_actions:
                    action_idx = random.choice(valid_actions)
                else:
                    action_idx = 0  # Default to not offloading
            else:
                # Get the index of the highest probability action
                action_idx = torch.argmax(action_probs).item()

            # Map the action index to the correct position in our one-hot vector
            # The output from actor has dimensions [device_num, neighbors_num + 1]
            # But our one-hot vector has dimensions [device_num, device_num + 1]

            # action_idx == 0 means "don't offload" (keep locally)
            if action_idx == 0:
                target_idx = 0
            else:
                # Map neighbor index to actual device index
                # For simplicity, assume the neighbors are indexed from 1 to neighbors_num
                # In a more complex scenario, you might need a mapping function here
                target_idx = min(action_idx, args.device_num)

            # Create one-hot vector
            one_hot = torch.zeros(args.device_num + 1)
            one_hot[target_idx] = 1.0
            actions[device_id] = one_hot
        else:
            # No task, default to not offloading (index 0)
            one_hot = torch.zeros(args.device_num + 1)
            one_hot[0] = 1.0
            actions[device_id] = one_hot

    # Debug: ensure we have at least some offloading happening
    if torch.sum(actions[:, 1:]) == 0:
        print("WARNING: All actions are 'don't offload'. Forcing some offloading.")
        # Force at least one device to offload
        device_id = random.randint(0, args.device_num - 1)
        neighbor_id = random.randint(1, min(args.neighbors_num, args.device_num))
        actions[device_id] = torch.zeros(args.device_num + 1)
        actions[device_id, neighbor_id] = 1.0

    return actions


def get_access_token(api_key, secret_key):
    """获取百度API访问令牌"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }
    response = requests.get(url, params=params)
    return response.json().get("access_token")


def get_reputation_matrix(dynamic_re, static_re, neighbor_matrix, args):
    # 准备输入数据
    if torch.is_tensor(dynamic_re):
        dynamic_re = dynamic_re.detach().cpu().numpy()
    if torch.is_tensor(static_re):
        static_re = static_re.detach().cpu().numpy()
    if torch.is_tensor(neighbor_matrix):
        neighbor_matrix = neighbor_matrix.detach().cpu().numpy()
    input_data = {
        "dynamic_reputation": dynamic_re.tolist() if isinstance(dynamic_re, np.ndarray) else dynamic_re,
        "static_reputation": static_re.tolist() if isinstance(static_re, np.ndarray) else static_re,
        "neighbor_matrix": neighbor_matrix.tolist() if isinstance(neighbor_matrix, np.ndarray) else neighbor_matrix
    }

    # 转换为JSON字符串
    # prompt = f"""
    # 请作为专业的信誉机制和D2D卸载专家，分析设备网络的信誉数据并生成信誉矩阵。
    #
    # 动态信誉数据: {json.dumps(input_data['dynamic_reputation'])}
    # 静态信誉数据: {json.dumps(input_data['static_reputation'])}
    # 邻居关系: {json.dumps(input_data['neighbor_matrix'])}
    #
    # 基于以上数据，生成一个大小为{args.device_num} * {args.device_num}的设备间信誉矩阵，每一行和每一列对应一个设备，矩阵的每个值在0到1之间。高值表示高信誉。
    # 只输出JSON格式的信誉矩阵，不要解释。
    # """
    prompt = f"""
    您是一位信誉计算专家。请基于以下数据生成一个精确的设备间信誉矩阵：

    1. 设备数量：{args.device_num}
    2. 动态信誉数据：{json.dumps(input_data['dynamic_reputation'])}
    3. 静态信誉数据：{json.dumps(input_data['static_reputation'])}
    4. 邻居关系矩阵：{json.dumps(input_data['neighbor_matrix'])}

    请生成一个{args.device_num}×{args.device_num}的信誉矩阵，其中：
    - 矩阵[i][j]表示设备i对设备j的信誉度
    - 所有值必须在0到1之间，高值表示高信誉
    - 设备对自身的信誉值应为1.0
    - 邻居之间的信誉应基于动态和静态信誉的相似度
    - 非邻居之间的信誉应基于共同邻居传播

    仅返回JSON格式的矩阵，不要添加任何解释、注释或额外文本。格式必须为：
    [[v11, v12, ..., v1n], [v21, v22, ..., v2n], ..., [vn1, vn2, ..., vnn]]
    """

    # Hugging Face API设置
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer "}

    # 设置Clash代理
    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890"
    }

    try:
        # # 获取access_token
        # access_token = get_access_token('bce-v3/ALTAK-hDiUnY3Hv7DladlzCfUdK/f073b19c4a9834ced59a824868ed3e97a5bea8f5', 'db62d9cf1f374349a7a8d6c8d55ec980')
        #
        # # 文心一言/千帆大模型API接口
        # url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={access_token}"
        #
        # # 构建请求参数
        # payload = json.dumps({
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ],
        #     "temperature": 0.1,  # 低温度，确保输出更确定性
        #     "top_p": 0.8,
        #     "penalty_score": 1.0,
        #     # 可以根据需要添加其他参数
        # })
        #
        # headers = {
        #     'Content-Type': 'application/json'
        # }
        # response = requests.post(url, headers=headers, data=payload, timeout=60)

        # 使用代理发送请求，并设置更长的超时时间
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt},
            proxies=proxies,
            timeout=60  # 增加超时时间到60秒
        )
        result = response.json()
        # print("API完整返回结果:", result)

        # 解析返回的JSON结果
        try:
            # 尝试从文本中提取JSON
            import re
            json_match = re.search(r'(\[\[.*?\]\])', result[0]['generated_text'], re.DOTALL)
            if json_match:
                reputation_matrix = torch.tensor(json.loads(json_match.group(1)))
            else:
                reputation_matrix = torch.tensor(json.loads(result[0]['generated_text']))
        except Exception as e:
            # print(f"解析API返回结果失败: {e}")
            # 创建备选的信誉矩阵
            device_num = len(static_re)
            # reputation_matrix = [[0.5 for _ in range(device_num)] for _ in range(device_num)]
            reputation_matrix = get_reputation_matrix_2(args)

    except requests.exceptions.RequestException as e:
        # print(f"API请求失败: {e}")
        # 创建备选的信誉矩阵
        device_num = len(static_re)
        # reputation_matrix = [[0.5 for _ in range(device_num)] for _ in range(device_num)]
        reputation_matrix = get_reputation_matrix_2(args)

    return reputation_matrix.clone().detach()

def get_reputation_matrix_2(args):
    reputation_matrix = torch.ones(args.device_num, args.device_num)
    malicious_count = int(args.malicious * args.device_num)

    for i in range(args.device_num):
        for j in range(args.device_num):
            if i >= malicious_count and j < malicious_count:
                reputation_matrix[i][j] = 0

    return reputation_matrix


def get_reputation_matrix_1(dynamic_re, static_re, neighbor_matrix, time_slot=None, accumulated_success=None,
                            accumulated_failure=None):
    """
    简化的时序感知信誉计算函数 - 能够随时间有效调整设备信誉，移除了neighbor_lookup依赖

    参数:
        dynamic_re: 动态信誉矩阵 [device_num, device_num] - 基于历史任务成功/失败
        static_re: 静态信誉向量 [device_num] - 基于设备安全级别和能力
        neighbor_matrix: 邻居矩阵 [device_num, neighbors_num]
        time_slot: 当前时隙（可选）- 用于时间衰减调整
        accumulated_success: 累积成功次数矩阵（可选）
        accumulated_failure: 累积失败次数矩阵（可选）

    返回:
        reputation_matrix: 信誉矩阵 [device_num, device_num]
    """
    # 确保输入为numpy数组
    if isinstance(dynamic_re, torch.Tensor):
        dynamic_re = dynamic_re.detach().cpu().numpy()
    if isinstance(static_re, torch.Tensor):
        static_re = static_re.detach().cpu().numpy()
    if isinstance(neighbor_matrix, torch.Tensor):
        neighbor_matrix = neighbor_matrix.detach().cpu().numpy()

    # 获取设备数量
    device_num = len(static_re)

    # 估计恶意设备 - 通过分析dynamic_re来识别
    failure_rates = np.zeros(device_num)
    interaction_counts = np.zeros(device_num)

    # 计算每个设备的平均动态信誉
    for j in range(device_num):
        ratings = []
        for i in range(device_num):
            if i != j and dynamic_re[i, j] > 0:  # 有交互记录
                ratings.append(dynamic_re[i, j])
                interaction_counts[j] += 1

        if len(ratings) > 0:
            avg_rating = np.mean(ratings)
            # 将平均评分转化为失败率估计
            failure_rates[j] = 1.0 - avg_rating
        else:
            # 无交互数据，使用中等偏保守的估计
            failure_rates[j] = 0.3

    # 识别潜在恶意设备 - 失败率高或静态信誉低的设备
    potential_malicious = []
    for j in range(device_num):
        malicious_score = 0.0

        # 基于失败率判断
        if failure_rates[j] > 0.4:
            malicious_score += 0.6
        elif failure_rates[j] > 0.25:
            malicious_score += 0.3

        # 基于静态信誉判断 (假设较低的静态信誉可能表示恶意设备)
        if static_re[j] < np.median(static_re) * 0.8:
            malicious_score += 0.4

        # 最终判断
        if malicious_score > 0.5:
            potential_malicious.append(j)

    # 初始化信誉矩阵
    reputation_matrix = np.zeros((device_num, device_num))

    # 估算恶意设备比例
    malicious_ratio = len(potential_malicious) / device_num

    # 构建清晰的二分类信誉矩阵
    for i in range(device_num):
        for j in range(device_num):
            if j in potential_malicious:  # j是潜在恶意设备
                # 低信誉基准值
                base_value = 0.1 + np.random.uniform(0, 0.1)

                # 添加动态信誉影响
                if dynamic_re[i, j] > 0:
                    dynamic_factor = dynamic_re[i, j] * 0.2  # 动态信誉影响较小
                    base_value = min(0.3, base_value + dynamic_factor)

                reputation_matrix[i, j] = base_value
            else:  # j是正常设备
                # 高信誉基准值
                base_value = 0.7 + np.random.uniform(0, 0.2)

                # 添加动态信誉影响
                if dynamic_re[i, j] > 0:
                    dynamic_factor = (1 - dynamic_re[i, j]) * 0.2  # 转化为惩罚项
                    base_value = max(0.7, base_value - dynamic_factor)

                reputation_matrix[i, j] = base_value

    # 添加邻居关系影响
    for i in range(device_num):
        for j in range(device_num):
            # 检查j是否为i的邻居
            is_neighbor = False
            for n in range(neighbor_matrix.shape[1]):
                if int(neighbor_matrix[i, n]) == j:
                    is_neighbor = True
                    break

            if is_neighbor:
                if j in potential_malicious:  # j是恶意设备且是邻居
                    # 邻居关系使恶意设备信誉略微提升，但仍保持低值
                    reputation_matrix[i, j] = min(0.4, reputation_matrix[i, j] * 1.1)
                else:  # j是正常设备且是邻居
                    # 邻居关系使正常设备信誉进一步提升
                    reputation_matrix[i, j] = min(0.95, reputation_matrix[i, j] * 1.05)

    # 对角线元素处理 (自评价)
    for i in range(device_num):
        # 基于静态信誉设置适度的自评价
        if i in potential_malicious:  # 恶意设备
            reputation_matrix[i, i] = 0.4
        else:  # 正常设备
            reputation_matrix[i, i] = 0.6

    # 应用Sigmoid变换增强对比度
    beta = 8.0  # Sigmoid曲线陡度
    midpoint = 0.5  # Sigmoid中点

    # 应用Sigmoid变换
    enhanced_matrix = 1 / (1 + np.exp(-beta * (reputation_matrix - midpoint)))

    # 进一步增强区分度 - 确保两类设备之间有明显的信誉差距
    final_matrix = np.zeros_like(enhanced_matrix)
    for i in range(device_num):
        for j in range(device_num):
            if j in potential_malicious:  # j是恶意设备
                # 确保恶意设备信誉值在明显较低的范围
                final_matrix[i, j] = enhanced_matrix[i, j] * 0.5
            else:  # j是正常设备
                # 确保正常设备信誉值在明显较高的范围
                final_matrix[i, j] = enhanced_matrix[i, j] * 0.3 + 0.7

    # 最终裁剪确保范围
    final_matrix = np.clip(final_matrix, 0.05, 0.95)

    return final_matrix

def KL(pred, target, epsilon=1e-10):
    """
    计算两个信誉矩阵之间的KL散度

    参数:
    - pred: 信誉网络预测的信誉矩阵
    - target: 规则模型生成的信誉矩阵
    - epsilon: 数值稳定性小值

    返回:
    - KL散度值
    """

    # 确保输入是浮点类型
    pred = pred.float()
    target = target.float()

    # 保存原始形状用于调试
    original_shape = pred.shape

    # 扁平化张量
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    # 归一化为概率分布
    pred_softmax = F.softmax(pred_flat, dim=0)
    target_softmax = F.softmax(target_flat, dim=0)

    # 确保没有零值
    pred_log = torch.log(pred_softmax + epsilon)

    # 按PyTorch实现规范，交换参数顺序
    # 计算 KL(target_softmax || pred_softmax)
    kl_loss = F.kl_div(
        pred_log,  # 预测的对数概率
        target_softmax,  # 目标概率
        reduction='sum'  # 使用sum以获得更有意义的值
    )

    # 返回标准化后的KL散度值乘以缩放因子，使其与actor_loss在同一数量级
    scale_factor = 1000.0  # 可调整
    return kl_loss * scale_factor


def init_reputation_network_uniform(model, target_value=0.5):
    """
    对Reputation网络进行合理的初始化

    参数:
        model (Reputation): 待初始化的Reputation网络实例
        target_value (float): 目标输出值，默认为0.5
    """
    # 1. 动态编码器初始化
    for layer in model.dynamic_encoder:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    # 2. 静态特征编码器初始化
    for layer in model.static_encoder:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    # 3. 邻居关系编码器初始化
    for layer in model.neighbor_encoder:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    # 4. 融合层初始化
    for layer in model.fusion:
        if isinstance(layer, nn.Linear):
            # 最后一层特殊处理，使输出更接近中性
            if layer == model.fusion[-1]:
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0)
            else:
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    model.training_step = 0

    return model


# MovingAverageStats类缺失，这里添加一个基本实现
class MovingAverageStats:
    def __init__(self, device_num, window_size=5):
        self.device_num = device_num
        self.window_size = window_size
        self.stats = torch.zeros(device_num)
        self.windows = [[] for _ in range(device_num)]

    def update(self, failure_indicators):
        if isinstance(failure_indicators, torch.Tensor):
            failure_indicators = failure_indicators.detach().cpu()

        for i in range(self.device_num):
            # 计算设备i的平均失败率
            device_failure = torch.mean(failure_indicators[i]).item()

            # 更新滑动窗口
            if len(self.windows[i]) >= self.window_size:
                self.windows[i].pop(0)
            self.windows[i].append(device_failure)

            # 更新统计值
            self.stats[i] = sum(self.windows[i]) / len(self.windows[i])

    def is_high_failure(self, device_id):
        return self.stats[device_id] > 0.6  # 阈值设为0.6

    def get_failure_score(self, device_id):
        return self.stats[device_id]

    def reset(self):
        self.stats = torch.zeros(self.device_num)
        self.windows = [[] for _ in range(self.device_num)]





def calculate_transmission_energy(neighbor_index, task_size):
    """
    计算任务传输的能耗

    参数:
    neighbor_index - 邻居索引列表
    task_size - 任务大小列表

    返回:
    transmission_energy - 每个设备的传输能耗
    """

    distance = neighbor_index + 1
    energy = distance ** 2 * task_size / 18

    return energy