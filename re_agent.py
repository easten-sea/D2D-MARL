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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class re_agent:
    def __init__(self, args):
        self.re_net = Reputation(args)
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.re_optimizer = torch.optim.Adam(self.re_net.parameters(), lr=0.0003)
        self.replay_buffer = ReplayBuffer(12800)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args

        # 调整输入维度
        reputation_matrix_dim = args.device_num * args.device_num  # 信誉矩阵
        task_vector_dim = args.device_num  # 任务向量
        network_topology_dim = args.device_num * args.neighbors_num  # 网络拓扑矩阵
        reputation_mask_dim = args.device_num  # 信誉屏蔽向量

        self.input_dim = reputation_matrix_dim + task_vector_dim + network_topology_dim + reputation_mask_dim
        self.hidden_dim = 256

        # 网络结构
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout1 = nn.Dropout(0.2)  # 添加dropout防止过拟合
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(0.2)  # 添加dropout防止过拟合

        # 决策头 - 每个设备选择是否卸载及卸载的目标设备
        self.action_head = nn.Linear(self.hidden_dim, args.device_num * (args.neighbors_num + 1))

        # 初始化参数，帮助训练开始
        self._init_weights()

    def _init_weights(self):
        # 对网络参数进行Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, reputation_matrix, task_vector, network_topology, reputation_mask):
        # 处理输入张量，确保维度正确，但不要断开梯度流
        if not isinstance(reputation_matrix, torch.Tensor):
            reputation_matrix_tensor = torch.tensor(reputation_matrix, dtype=torch.float32).flatten()
        else:
            reputation_matrix_tensor = reputation_matrix.float().flatten()
        if not isinstance(task_vector, torch.Tensor):
            task_vector_tensor = torch.tensor(task_vector, dtype=torch.float32)
        else:
            task_vector_tensor = task_vector.float()

        if not isinstance(network_topology, torch.Tensor):
            network_topology_tensor = torch.tensor(network_topology, dtype=torch.float32).flatten()
        else:
            network_topology_tensor = network_topology.float().flatten()

        if not isinstance(reputation_mask, torch.Tensor):
            reputation_mask_tensor = torch.tensor(reputation_mask, dtype=torch.float32)
        else:
            reputation_mask_tensor = reputation_mask.float()

        # 添加小的噪声帮助打破梯度为0的情况
        noise_factor = 1e-4
        if self.training:
            reputation_matrix_tensor = reputation_matrix_tensor + torch.randn_like(reputation_matrix_tensor) * noise_factor
            task_vector_tensor = task_vector_tensor + torch.randn_like(task_vector_tensor) * noise_factor
            network_topology_tensor = network_topology_tensor + torch.randn_like(network_topology_tensor) * noise_factor

        # 合并输入特征
        combined_input = torch.cat([
            reputation_matrix_tensor,
            task_vector_tensor,
            network_topology_tensor,
            reputation_mask_tensor
        ])

        # 前向传播
        x = F.relu(self.fc1(combined_input))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # 生成动作概率
        action_logits = self.action_head(x)
        action_probs = action_logits.reshape(self.args.device_num, self.args.neighbors_num + 1)

        # 使用softmax得到初始概率分布
        action_probs = F.softmax(action_probs, dim=1)

        # 创建掩码矩阵，初始所有行为1（允许所有动作）
        mask_with_self = torch.ones(self.args.device_num, self.args.neighbors_num + 1)

        # 第一列（不卸载选项）保持为1，其他列根据reputation_mask设置
        for i in range(self.args.device_num):
            for j in range(1, self.args.neighbors_num + 1):
                neighbor_idx = int(network_topology[i][j - 1])
                mask_with_self[i, j] = reputation_mask_tensor[neighbor_idx]
        # 应用掩码并重新归一化
        masked_probs = action_probs * mask_with_self
        row_sums = masked_probs.sum(dim=1, keepdim=True)
        masked_probs = masked_probs / row_sums.clamp(min=1e-8)

        return masked_probs


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        # 基本状态维度
        dynamic_re_dim = args.device_num * args.device_num
        static_re_dim = args.device_num
        task_em_dim = args.device_num
        r_mask_dim = args.device_num
        neighbor_matrix_dim = args.device_num * args.neighbors_num
        self.action_dim = args.device_num * (args.neighbors_num + 1)
        self.state_input_dim = dynamic_re_dim + static_re_dim + task_em_dim + r_mask_dim + neighbor_matrix_dim

        self.state_fc1 = nn.Linear(self.state_input_dim, 256)
        self.state_fc2 = nn.Linear(256, 128)

        self.action_fc1 = nn.Linear(self.action_dim, 128)

        self.joint_fc1 = nn.Linear(256, 128)
        self.joint_fc2 = nn.Linear(128, 64)
        self.joint_fc3 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dynamic_re, static_re, task_em, neighbor_matrix, r_mask, action):
        if not isinstance(dynamic_re, torch.Tensor):
            dynamic_re_tensor = torch.tensor(dynamic_re, dtype=torch.float32)
        else:
            dynamic_re_tensor = dynamic_re.detach().clone().to(torch.float32)
        if not isinstance(static_re, torch.Tensor):
            static_re_tensor = torch.tensor(static_re, dtype=torch.float32)
        else:
            static_re_tensor = static_re.detach().clone().to(torch.float32)
        if not isinstance(task_em, torch.Tensor):
            task_em_tensor = torch.tensor(task_em, dtype=torch.float32)
        else:
            task_em_tensor = task_em.detach().clone().to(torch.float32)
        if not isinstance(neighbor_matrix, torch.Tensor):
            neighbor_matrix_tensor = torch.tensor(neighbor_matrix, dtype=torch.float32)
        else:
            neighbor_matrix_tensor = neighbor_matrix.detach().clone().to(torch.float32)
        if not isinstance(r_mask, torch.Tensor):
            r_mask_tensor = torch.tensor(r_mask, dtype=torch.float32)
        else:
            r_mask_tensor = r_mask.detach().clone().to(torch.float32)
        if not isinstance(action, torch.Tensor):
            action_tensor = torch.tensor(action, dtype=torch.float32)
        else:
            action_tensor = action

        dynamic_re_flat = dynamic_re_tensor.reshape(-1)
        neighbor_matrix_flat = neighbor_matrix_tensor.reshape(-1)
        action_flat = action_tensor.reshape(-1)

        state_input = torch.cat([
            dynamic_re_flat,
            static_re_tensor,
            task_em_tensor,
            neighbor_matrix_flat,
            r_mask_tensor
        ], dim=0)

        state_x = F.relu(self.state_fc1(state_input))
        state_x = F.dropout(state_x, p=0.2, training=self.training)
        state_features = F.relu(self.state_fc2(state_x))

        action_features = F.relu(self.action_fc1(action_flat))
        action_features = F.dropout(action_features, p=0.2, training=self.training)
        combined = torch.cat([state_features, action_features])

        x = F.relu(self.joint_fc1(combined))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.joint_fc2(x))
        q_value = self.joint_fc3(x)

        return q_value


class Reputation(nn.Module):
    def __init__(self, args):
        super(Reputation, self).__init__()
        self.args = args
        self.last_reputation_matrix = torch.zeros(args.device_num, args.device_num)
        device_num = args.device_num

        self.dynamic_encoder = nn.Sequential(
            nn.Linear(device_num, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.1)
        )

        self.neighbor_encoder = nn.Sequential(
            nn.Linear(args.neighbors_num, 16),
            nn.LeakyReLU(0.1)
        )

        self.failure_correlation = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 16 + 16 + 16, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )

        self.reputation_norm = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, device_num),
            nn.Tanh()
        )

        self.out = nn.Sequential(
            nn.Linear(device_num * 2, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, device_num),
            nn.Sigmoid()
        )

        self.training_step = 0

    def forward(self, dynamic_re, static_re, neighbor_matrix, reset=False):
        """
        结合动态安全变量与设备特征生成信誉评价

        Args:
            dynamic_re: 动态安全变量 [device_num, device_num]
            static_re: 静态安全向量 [device_num]
            neighbor_matrix: 邻居矩阵 [device_num, neighbors_num]
        """
        if not isinstance(dynamic_re, torch.Tensor):
            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32)
        if not isinstance(static_re, torch.Tensor):
            static_re = torch.tensor(static_re, dtype=torch.float32)
        if not isinstance(neighbor_matrix, torch.Tensor):
            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32)

        if self.training:
            self.last_reputation_matrix = torch.zeros(self.args.device_num, self.args.device_num)
        # 计算各设备的失败率
        device_failure_rates = torch.mean(dynamic_re, dim=0)
        failure_deviation = device_failure_rates - 0.5
        failure_features = self.failure_correlation(failure_deviation.unsqueeze(-1))

        rater_features = self.dynamic_encoder(dynamic_re)
        target_features = self.dynamic_encoder(dynamic_re.t())
        static_features = self.static_encoder(static_re.unsqueeze(-1))
        neighbor_features = self.neighbor_encoder(neighbor_matrix)

        # 合并特征，加入失败率特征
        device_features = torch.cat([
            rater_features,
            target_features,
            static_features,
            neighbor_features,
            failure_features
        ], dim=1)

        device_embeddings = self.fusion(device_features)

        # 创建信誉矩阵
        reputation_mat = self.reputation_norm(device_embeddings)
        reputation_feature = torch.cat([reputation_mat, self.last_reputation_matrix], dim = 1)
        reputation_mat = self.out(reputation_feature)

        alpha = max(0.3, 0.7 - self.training_step / 1000)
        failure_factor = torch.sigmoid((0.5 - device_failure_rates) * 5)  # 将偏差放大并映射到0-1
        failure_factor_column = failure_factor.unsqueeze(0).expand_as(reputation_mat)
        reputation_matrix = alpha * dynamic_re + (1 - alpha) * reputation_mat * failure_factor_column
        reputation_matrix += torch.where(torch.mean(dynamic_re, dim=0) >= 0.5, 0.1, -0.1).unsqueeze(1).expand_as(reputation_matrix)
        reputation_matrix = torch.clamp(reputation_matrix, 0.05, 0.95)

        self.last_reputation_matrix = reputation_matrix

        return reputation_matrix


class MovingAverageStats:
    """跟踪设备的失败率统计"""

    def __init__(self, device_num, window_size=5):
        self.device_num = device_num
        self.window_size = window_size
        self.failure_history = [[] for _ in range(device_num)]

    def update(self, failure_indicators):
        """更新失败统计"""
        # 计算每个设备作为执行者的平均失败率
        device_failure_rates = torch.mean(failure_indicators, dim=0)  # 列平均

        for i in range(self.device_num):
            self.failure_history[i].append(device_failure_rates[i].item())
            # 保持窗口大小
            if len(self.failure_history[i]) > self.window_size:
                self.failure_history[i].pop(0)

    def is_high_failure(self, device_idx):
        """判断设备是否有高失败率"""
        if len(self.failure_history[device_idx]) < 3:
            return False

        recent_failures = self.failure_history[device_idx][-3:]
        return sum(recent_failures) / len(recent_failures) > 0.6

    def get_failure_score(self, device_idx):
        """获取设备的失败评分"""
        if len(self.failure_history[device_idx]) == 0:
            return 0.0

        return sum(self.failure_history[device_idx]) / len(self.failure_history[device_idx])


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # 解包但不尝试堆叠
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换成适当的格式
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # 保持actions和states作为列表返回
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)







#SWAT
class RobustTrustComputing:
    """
    SWAT:
    多维评估：同时考虑动态信誉（基于历史交互）和静态信誉（设备固有安全能力）
    滞后函数：使用非线性sigmoid滞后函数来平滑信誉变化，防止小波动引起信誉大幅波动
    统计验证：使用置信区间验证性能变化是否为暂时性，只对真实性能下降做出反应
    参数组合：通过加权平均合并不同维度的信誉参数
    """
    def __init__(self, args):
        self.args = args
        self.device_num = args.device_num
        self.confidence_level = 0.95  # 可以调整的置信水平
        self.hysteresis_k = 0.5  # 滞后函数的水平偏移
        self.temp_storage = {}  # 用于存储历史性能值

        # 初始化权重参数
        self.weights = {}

    def sigmoid(self, x):
        """S型函数 - 用于滞后函数计算"""
        # 确保x是tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return (1 - torch.exp(-x)) / (1 + torch.exp(-x))

    def hysteresis(self, x, prev_x):
        """实现滞后函数，基于当前值和前一个值的比较"""
        # 确保x和prev_x是float类型
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(prev_x, torch.Tensor):
            prev_x = prev_x.item()

        if x > prev_x:
            return self.sigmoid(torch.tensor(x - self.hysteresis_k, dtype=torch.float32))
        else:
            return self.sigmoid(torch.tensor(x + self.hysteresis_k, dtype=torch.float32))

    def compute_confidence_interval(self, observed_values):
        """计算置信区间，用于验证性能波动是否为暂时性"""
        if len(observed_values) < 2:
            return -float('inf'), float('inf')  # 数据不足时返回无穷区间

        values = torch.tensor(observed_values, dtype=torch.float32)
        mean = torch.mean(values)
        std = torch.std(values)

        # 根据置信水平计算t值（简化处理，使用正态分布近似）
        if self.confidence_level == 0.95:
            t_value = 1.96
        elif self.confidence_level == 0.90:
            t_value = 1.645
        else:
            t_value = 2.0  # 默认值

        margin = t_value * std / math.sqrt(len(observed_values))
        return mean.item() - margin.item(), mean.item() + margin.item()

    def __call__(self, dynamic_re, static_re, neighbor_matrix):
        """计算信誉矩阵 - 替代原来的re_net函数"""
        # 转换为张量
        if not isinstance(dynamic_re, torch.Tensor):
            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32)
        if not isinstance(static_re, torch.Tensor):
            static_re = torch.tensor(static_re, dtype=torch.float32)
        if not isinstance(neighbor_matrix, torch.Tensor):
            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.int64)

        # 初始化信誉矩阵
        reputation_matrix = torch.zeros((self.device_num, self.device_num), dtype=torch.float32)
        for i in range(self.device_num):
            # 处理每个设备
            for j in range(self.device_num):
                # 跳过自己
                if i == j:
                    reputation_matrix[i, j] = 1.0  # 对自己的信任为满分
                    continue

                # 首先检查j是否为i的直接邻居
                is_neighbor = False
                neighbor_idx = -1
                for k in range(len(neighbor_matrix[i])):
                    if neighbor_matrix[i][k].item() == j:
                        is_neighbor = True
                        neighbor_idx = k
                        break

                if is_neighbor:
                    # 直接邻居处理
                    # 获取动态信誉和静态信誉
                    tau_dynamic = dynamic_re[i, j].item() if i < dynamic_re.shape[0] and j < dynamic_re.shape[1] else 0.5
                    tau_static = static_re[j].item() if j < len(static_re) else 0.5

                    # 存储观察到的性能值用于统计验证
                    key = f"{i}-{j}"
                    if key not in self.temp_storage:
                        self.temp_storage[key] = []
                    self.temp_storage[key].append(tau_dynamic)

                    # 保留最近的30个观察值
                    if len(self.temp_storage[key]) > 30:
                        self.temp_storage[key].pop(0)

                    # 计算置信区间
                    lower_bound, upper_bound = self.compute_confidence_interval(self.temp_storage[key])

                    # 统计验证 - 如果性能在置信区间内，则视为正常波动
                    if lower_bound <= tau_dynamic <= upper_bound and len(self.temp_storage[key]) > 5:
                        # 性能波动在可接受范围内，维持先前的信誉值
                        if len(self.temp_storage[key]) > 1:
                            prev_tau = self.temp_storage[key][-2]
                            # 应用滞后函数来防止小波动
                            reputation_matrix[i, j] = self.hysteresis(tau_dynamic, prev_tau)
                        else:
                            reputation_matrix[i, j] = torch.tensor(tau_dynamic, dtype=torch.float32)
                    else:
                        # 计算参数权重
                        alpha_dynamic = 0.7  # 动态信誉权重
                        alpha_static = 0.3  # 静态信誉权重

                        # 计算归一化参数
                        tau_norm_dynamic = tau_dynamic
                        tau_norm_static = tau_static

                        # 合并参数 - 多维信任计算的核心
                        combined_tau = (alpha_dynamic * tau_norm_dynamic + alpha_static * tau_norm_static) / (alpha_dynamic + alpha_static)

                        # 应用滞后函数来增强稳健性
                        prev_tau = self.temp_storage[key][-2] if len(self.temp_storage[key]) > 1 else combined_tau
                        reputation_matrix[i, j] = self.hysteresis(combined_tau, prev_tau)
                else:
                    # 非邻居设备，基于传递性信任计算
                    # 找出共同邻居
                    common_neighbors = []
                    neighbor_i_list = [neighbor_matrix[i][k].item() for k in range(len(neighbor_matrix[i]))]
                    neighbor_j_list = [neighbor_matrix[j][k].item() for k in range(len(neighbor_matrix[j]))]

                    common_neighbors = list(set(neighbor_i_list) & set(neighbor_j_list))

                    if common_neighbors:
                        # 通过共同邻居计算间接信任
                        indirect_trust = 0.0
                        for common in common_neighbors:
                            common_idx = int(common)
                            indirect_trust += reputation_matrix[i, common_idx].item() * reputation_matrix[common_idx, j].item()
                        indirect_trust /= len(common_neighbors)
                        reputation_matrix[i, j] = torch.tensor(indirect_trust * 0.8, dtype=torch.float32)  # 间接信任打折扣
                    else:
                        # 无共同邻居，使用全局平均值和静态信誉
                        reputation_matrix[i, j] = static_re[j] * 0.5 if j < len(static_re) else torch.tensor(0.3, dtype=torch.float32)

        # 确保信誉值在[0,1]范围内
        reputation_matrix = torch.clamp(reputation_matrix, 0.0, 1.0)

        return reputation_matrix


# 包装类，兼容原有接口
class RobustTrustWrapper(nn.Module):
    def __init__(self, args):
        super(RobustTrustWrapper, self).__init__()
        self.trust_mechanism = RobustTrustComputing(args)

    def forward(self, dynamic_re, static_re, neighbor_matrix, reset=False):
        # reset参数保留但不使用，以兼容原有接口
        return self.trust_mechanism(dynamic_re, static_re, neighbor_matrix)

    def eval(self):
        return self

    def train(self):
        return self

def create_robust_trust_mechanism(args):
    return RobustTrustWrapper(args)

class swat_agent:
    def __init__(self, args):
        # 使用鲁棒多维信任计算机制替代原有的Reputation网络
        self.re_net = create_robust_trust_mechanism(args)
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(12800)





#RTCM

class RTCM:
    """
    基于多源反馈和雾计算的可靠信任计算机制 (RTCM)
    """
    def __init__(self, args):
        self.args = args
        self.device_num = args.device_num
        self.neighbors_num = args.neighbors_num

        # 初始化直接信任矩阵
        self.direct_trust_matrix = torch.ones(args.device_num, args.device_num) * 0.5
        # 初始化推荐信任矩阵
        self.recommendation_trust_matrix = torch.ones(args.device_num, args.device_num) * 0.5
        # 初始化最终信任矩阵
        self.trust_matrix = torch.ones(args.device_num, args.device_num) * 0.5

        # 时间衰减参数 (λ)
        self.time_decay = 0.9

        # 信任阈值
        self.trust_threshold = 0.3

        # 直接信任计算参数
        self.alpha = 1.0  # 成功交互权重
        self.beta = 1.0  # 失败交互权重

        # 信任融合权重
        self.w_direct = 0.6  # 直接信任权重
        self.w_recommendation = 0.3  # 推荐信任权重
        self.w_capability = 0.05  # 设备能力权重
        self.w_security = 0.05  # 设备安全权重

        # 恶意检测阈值
        self.deviation_threshold = 0.3  # 反馈偏差阈值
        self.frequency_threshold = 0.4  # 反馈频率阈值
        self.consistency_threshold = 0.3  # 邻居一致性阈值

        # 记录历史评估结果，用于时间序列分析
        self.historical_assessments = []
        self.max_history_length = 10

    def forward(self, dynamic_re, static_re, neighbor_matrix, reset=False):
        """
        计算设备间的信任关系

        Args:
            dynamic_re: 交互历史矩阵 [device_num, device_num]，值表示设备i对设备j的交互成功率
            static_re: 设备静态安全和能力属性 [device_num]，静态安全性能指标
            neighbor_matrix: 邻居关系矩阵 [device_num, neighbors_num]，表示雾计算层的节点关系
            reset: 是否重置信任矩阵，用于重新初始化

        Returns:
            trust_matrix: 更新后的信任矩阵 [device_num, device_num]
        """
        # 确保输入是张量
        if not isinstance(dynamic_re, torch.Tensor):
            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32)
        if not isinstance(static_re, torch.Tensor):
            static_re = torch.tensor(static_re, dtype=torch.float32)
        if not isinstance(neighbor_matrix, torch.Tensor):
            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32)

        # 重置信任状态（如果需要）
        if reset:
            self.direct_trust_matrix = torch.ones(self.device_num, self.device_num) * 0.5
            self.recommendation_trust_matrix = torch.ones(self.device_num, self.device_num) * 0.5
            self.trust_matrix = torch.ones(self.device_num, self.device_num) * 0.5
            self.historical_assessments = []

        # 提取设备安全性能和计算能力
        device_security = torch.zeros(self.device_num)
        device_capability = torch.zeros(self.device_num)

        for i in range(self.device_num):
            device_security[i] = static_re[i]
            device_capability[i] = static_re[i]

        success_matrix = dynamic_re.clone()
        failure_matrix = 1.0 - dynamic_re  # 失败率

        direct_trust = dynamic_re.clone()

        # 更新直接信任矩阵
        self.direct_trust_matrix = direct_trust.clone()

        # ----------雾计算层恶意反馈检测 ----------#
        feedback_credibility = torch.ones(self.device_num)

        # 使用类中定义的阈值
        deviation_threshold = self.deviation_threshold
        frequency_threshold = self.frequency_threshold
        consistency_threshold = self.consistency_threshold

        for i in range(self.device_num):
            feedback_deviation = torch.mean(torch.abs(direct_trust[i, :] - torch.mean(direct_trust[:, :], dim=0)))
            feedback_frequency = torch.sum(success_matrix[i, :] + failure_matrix[i, :]) / self.device_num
            neighbor_consistency = 0.0
            neighbor_count = 0
            for n in range(self.neighbors_num):
                neighbor_id = int(neighbor_matrix[i, n])
                if neighbor_id != i:
                    # 计算设备i和其邻居对其他设备评价的平均差异
                    diff = torch.mean(torch.abs(direct_trust[i, :] - direct_trust[neighbor_id, :]))
                    neighbor_consistency += diff
                    neighbor_count += 1
            if neighbor_count > 0:
                neighbor_consistency = neighbor_consistency / neighbor_count
            else:
                neighbor_consistency = 0.0

            # 根据指标判断反馈可信度
            # 如果多个指标异常，认为反馈不可信
            anomaly_count = 0
            if feedback_deviation > deviation_threshold:
                anomaly_count += 1
            if feedback_frequency < frequency_threshold:
                anomaly_count += 1
            if neighbor_consistency > consistency_threshold:
                anomaly_count += 1

            # 根据异常计数调整可信度
            if anomaly_count >= 2:  # 如果至少有两个指标异常
                feedback_credibility[i] = 0.3  # 低可信度
            elif anomaly_count == 1:  # 一个指标异常
                feedback_credibility[i] = 0.7  # 中等可信度
            else:  # 没有异常
                feedback_credibility[i] = 1.0  # 高可信度

        # ----------雾计算层推荐信任计算 ----------#
        recommendation_trust = torch.zeros(self.device_num, self.device_num)

        for i in range(self.device_num):
            for j in range(self.device_num):
                if i != j:
                    # 收集邻居对目标j的推荐意见
                    neighbor_recommendations = []
                    neighbor_credibility = []

                    for n in range(self.neighbors_num):
                        neighbor_id = int(neighbor_matrix[i, n])
                        if neighbor_id != i and neighbor_id != j:  # 避免自身和目标节点
                            # 邻居的直接信任值作为推荐
                            neighbor_recommendations.append(direct_trust[neighbor_id, j])
                            # 考虑推荐者的可信度
                            neighbor_credibility.append(feedback_credibility[neighbor_id])

                    # 如果有足够的推荐
                    if len(neighbor_recommendations) > 0:
                        # 将推荐信息转为张量
                        rec_tensor = torch.tensor(neighbor_recommendations, dtype=torch.float32)
                        cred_tensor = torch.tensor(neighbor_credibility, dtype=torch.float32)

                        # 根据论文公式，加权聚合推荐
                        # RecommendationTrust(i,j) = Σ(Credibility(k) * DirectTrust(k,j)) / Σ(Credibility(k))
                        weighted_rec = rec_tensor * cred_tensor
                        recommendation_trust[i, j] = torch.sum(weighted_rec) / torch.sum(cred_tensor)
                    else:
                        # 无可用推荐时，使用中等信任值
                        recommendation_trust[i, j] = 0.5
                else:
                    recommendation_trust[i, j] = 1.0

        # 更新推荐信任矩阵
        self.recommendation_trust_matrix = recommendation_trust.clone()

        # ----------信任融合算法 ----------#
        new_trust_matrix = torch.zeros_like(self.trust_matrix)

        # 使用类中定义的信任融合权重
        w_direct = self.w_direct
        w_recommendation = self.w_recommendation
        w_capability = self.w_capability
        w_security = self.w_security

        for i in range(self.device_num):
            for j in range(self.device_num):
                if i != j:
                    fused_trust = (
                            w_direct * direct_trust[i, j] +
                            w_recommendation * recommendation_trust[i, j] +
                            w_capability * device_capability[j] +
                            w_security * device_security[j]
                    )

                    # 时间平滑处理
                    new_trust_matrix[i, j] = (
                            self.time_decay * self.trust_matrix[i, j] +
                            (1.0 - self.time_decay) * fused_trust
                    )
                else:
                    new_trust_matrix[i, j] = 1.0  # 自信任

        # 确保信任值在合理范围内
        new_trust_matrix = torch.clamp(new_trust_matrix, 0.05, 0.95)

        # 更新全局信任矩阵
        self.trust_matrix = new_trust_matrix

        return self.trust_matrix


class RTCM_agent:
    def __init__(self, args):
        self.re_net = RTCM(args)
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(12800)



class RLTCM:
    """
    基于多源反馈信息融合的物联网边缘设备可靠轻量级信任计算机制
    """

    def __init__(self, args):
        self.args = args
        self.device_num = args.device_num
        self.neighbors_num = args.neighbors_num
        self.last_reputation_matrix = torch.zeros((args.device_num, args.device_num))
        self.alpha = 0.5
        self.su_fa = args.su_fa if hasattr(args, 'su_fa') else 1.0
        self.se_ca = args.se_ca if hasattr(args, 'se_ca') else 0.9

    def _calculate_direct_trust(self, task_suc, task_fail):
        """
        使用论文中的公式 (2) 计算所有 D-to-D 直接信任

        参数：
        task_suc：成功矩阵 [device_num, device_num]
        task_fail：失败矩阵 [device_num, device_num]

        返回：
        直接信任矩阵 [device_num, device_num]
        """
        # 创建特殊情况的掩码
        case1_mask = (task_suc != 0) & (task_fail == 0)
        case2_mask = (task_suc == 0) & (task_fail == 0)

        # 计算一般情况下的直接信任
        direct_trust = (task_suc + 1) / (task_suc + self.su_fa * task_fail + 2)

        # 应用特殊情况
        direct_trust = torch.where(case1_mask, torch.ones_like(direct_trust), direct_trust)
        direct_trust = torch.where(case2_mask, torch.zeros_like(direct_trust), direct_trust)

        return direct_trust

    def _calculate_information_entropy_weights(self, dynamic_re):
        """
        使用客观信息熵理论计算权重

        参数：
        dynamic_re：矩阵 [device_num, device_num]，包含来自设备的反馈

        返回：
        每个设备反馈的权重 [device_num]
        """
        # 归一化矩阵
        row_sums = torch.sum(dynamic_re, dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        normalized_matrix = dynamic_re / row_sums

        # 计算熵
        # 添加小的epsilon防止log(0)
        epsilon = 1e-10
        log_norm = torch.log(normalized_matrix + epsilon)
        entropy_per_element = -normalized_matrix * log_norm

        # 将非有效值（0*log(0)应为0）替换为0
        entropy_per_element = torch.where(
            normalized_matrix > 0,
            entropy_per_element,
            torch.zeros_like(entropy_per_element)
        )

        # 按行求和得到每个设备的熵值
        entropy_values = torch.sum(entropy_per_element, dim=1)

        # 归一化熵值
        normalized_entropy = entropy_values / math.log(self.device_num)

        # 计算权重
        weights = 1 - normalized_entropy
        sum_weights = torch.sum(weights)

        if sum_weights == 0:
            weights = torch.ones(self.device_num) / self.device_num
        else:
            weights = weights / sum_weights

        return weights

    def _calculate_feedback_trust_vectorized(self, dynamic_re):
        """
        使用算法2计算所有设备的B-to-D反馈信任（向量化）

        参数：
        dynamic_re：动态信誉矩阵 [device_num, device_num]

        返回：
        所有设备的反馈信任 [device_num]
        """
        weights = self._calculate_information_entropy_weights(dynamic_re)

        # 创建对角线掩码以排除自评估
        mask = 1 - torch.eye(self.device_num, device=dynamic_re.device)

        # 计算每个设备对其他设备的评价
        feedback_trust = torch.zeros(self.device_num, device=dynamic_re.device)

        for j in range(self.device_num):
            weighted_sum = 0.0
            weighted_count = 0

            for i in range(self.device_num):
                if i != j:  # 排除自评估
                    weighted_sum += dynamic_re[i, j] * weights[i]
                    weighted_count += weights[i]

            if weighted_count > 0:
                feedback_trust[j] = weighted_sum / weighted_count
            else:
                feedback_trust[j] = 0.5  # 默认值

        return feedback_trust

    def _calculate_adaptive_weight_vectorized(self, dynamic_re):
        """
        向量化计算信任聚合的自适应权重

        参数：
        dynamic_re：动态信誉矩阵 [device_num, device_num]

        返回：
        所有设备的自适应权重 [device_num]
        """
        # 计算每个设备j收到的负面评价数量
        negative_ratings = torch.sum(dynamic_re < 0.5, dim=0).float()

        # 使用公式(5)计算自适应权重: ω = 1 / (1 + sqrt(negative_ratings))
        adaptive_weights = 1.0 / (1.0 + torch.sqrt(negative_ratings))

        return adaptive_weights

    def _calculate_static_trust_factor_vectorized(self, static_re, neighbor_matrix):
        """
        向量化计算所有设备对的静态信任因子

        参数：
        static_re：静态安全向量 [device_num]
        neighbor_matrix：邻居矩阵 [device_num, neighbors_num]

        返回：
        静态信任因子矩阵 [device_num, device_num]
        """
        # 初始化静态信任矩阵
        static_trust_matrix = torch.zeros((self.device_num, self.device_num), device=static_re.device)

        # 为每个设备计算邻居的平均安全能力
        neighbor_security_sum = torch.zeros(self.device_num, device=static_re.device)
        neighbor_count = torch.zeros(self.device_num, device=static_re.device)
        valid_neighbor_mask = (neighbor_matrix >= 0) & (neighbor_matrix < self.device_num)

        for i in range(self.device_num):
            # 获取当前设备的有效邻居索引
            valid_neighbors = neighbor_matrix[i][valid_neighbor_mask[i]].long()

            if len(valid_neighbors) > 0:
                # 计算邻居的安全能力总和
                neighbor_security_sum[i] = torch.sum(static_re[valid_neighbors])
                neighbor_count[i] = len(valid_neighbors)

        # 计算平均邻居安全能力
        avg_neighbor_security = neighbor_security_sum / torch.clamp(neighbor_count, min=1)

        # 构建邻居关系矩阵（设备i是否为设备j的邻居）
        is_neighbor = torch.zeros((self.device_num, self.device_num), dtype=torch.bool, device=static_re.device)

        for i in range(self.device_num):
            for n in range(self.neighbors_num):
                if neighbor_matrix[i, n] >= 0 and neighbor_matrix[i, n] < self.device_num:
                    j = int(neighbor_matrix[i, n])
                    is_neighbor[i, j] = True

        # 计算每个目标设备的静态信任因子
        for j in range(self.device_num):
            # 计算设备j的静态信任
            target_security = static_re[j]
            avg_security = avg_neighbor_security[j]

            # 静态信任计算
            static_trust_j = self.se_ca * target_security + (1 - self.se_ca) * avg_security

            # 只对邻居关系应用静态信任
            static_trust_matrix[:, j] = torch.where(
                is_neighbor[:, j],
                static_trust_j,
                torch.zeros_like(static_trust_matrix[:, j])
            )

        return static_trust_matrix

    def forward(self, dynamic_re, static_re, neighbor_matrix, reset=False):
        """
        结合直接信任、反馈信任和静态因素计算全局信任（向量化实现）

        参数：
        dynamic_re：动态信誉矩阵 [device_num, device_num]
        static_re：静态安全向量 [device_num]
        neighbor_matrix：邻居矩阵 [device_num, neighbors_num]
        reset：是否重置历史数据

        返回：
        全局信任矩阵 [device_num, device_num]
        """
        # 确保输入是PyTorch张量
        if not isinstance(dynamic_re, torch.Tensor):
            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32)
        if not isinstance(static_re, torch.Tensor):
            static_re = torch.tensor(static_re, dtype=torch.float32)
        if not isinstance(neighbor_matrix, torch.Tensor):
            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.float32)

        if reset:
            self.last_reputation_matrix = torch.zeros_like(self.last_reputation_matrix)

        # 初始化声誉矩阵
        reputation_matrix = torch.zeros((self.device_num, self.device_num), device=dynamic_re.device)

        # 设置对角线值为0.95（自我信任）
        diagonal_mask = torch.eye(self.device_num, device=dynamic_re.device, dtype=torch.bool)
        reputation_matrix[diagonal_mask] = 0.95

        # 计算非对角线元素
        non_diagonal_mask = ~diagonal_mask

        # 1. 直接信任已在dynamic_re中给出
        direct_trust = dynamic_re

        # 2. 计算所有设备的反馈信任
        feedback_trust = self._calculate_feedback_trust_vectorized(dynamic_re)

        # 3. 计算所有设备的自适应权重
        adaptive_weights = self._calculate_adaptive_weight_vectorized(dynamic_re)

        # 4. 计算静态信任因子矩阵
        static_trust_matrix = self._calculate_static_trust_factor_vectorized(static_re, neighbor_matrix)

        # 5. 结合直接信任和反馈信任
        for i in range(self.device_num):
            for j in range(self.device_num):
                if i != j:  # 排除对角线元素
                    # 使用自适应权重合并直接信任和反馈信任
                    omega = float(adaptive_weights[j])
                    direct = float(direct_trust[i, j])
                    feedback = float(feedback_trust[j])

                    # 计算全局信任
                    global_trust = omega * direct + (1 - omega) * feedback

                    # 应用静态信任因子（如果有）
                    static_trust = float(static_trust_matrix[i, j])
                    if static_trust > 0:
                        static_factor = 0.8 + 0.4 * static_trust
                        global_trust = global_trust * static_factor

                    reputation_matrix[i, j] = global_trust

        # 应用历史信息平滑
        if hasattr(self, 'last_reputation_matrix') and self.last_reputation_matrix.sum() > 0:
            reputation_matrix = self.alpha * self.last_reputation_matrix + (1 - self.alpha) * reputation_matrix

        # 处理设备故障率
        device_failure_rates = torch.mean(dynamic_re, dim=0)

        # 应用设备故障率调整
        high_failure_mask = device_failure_rates >= 0.5
        low_failure_mask = ~high_failure_mask

        # 调整有高故障率的设备
        for j in range(self.device_num):
            if high_failure_mask[j]:
                reputation_matrix[:, j] = torch.clamp(reputation_matrix[:, j] - 0.1, min=0.05)
            else:
                reputation_matrix[:, j] = torch.clamp(reputation_matrix[:, j] + 0.05, max=0.95)

        # 应用全局边界
        reputation_matrix = torch.clamp(reputation_matrix, 0.05, 0.95)

        # 更新历史数据
        self.last_reputation_matrix = reputation_matrix.clone()

        return reputation_matrix


class RLTCM_agent:
    def __init__(self, args):
        self.re_net = RLTCM(args)
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(12800)





#multi-edge server
class multi_re_agent:
    def __init__(self, args):
        # 初始化网络
        self.re_net = Reputation(args)
        self.actor = Multi_Actor(args)
        self.critic = Multi_Critic(args)
        self.re_aware = ReputationAwareNetwork(args)

        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.re_optimizer = torch.optim.Adam(self.re_net.parameters(), lr=0.0003)
        self.re_aw_optimizer = torch.optim.Adam(self.re_aware.parameters(), lr=0.0003)

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(12800)


class ReputationAwareNetwork(nn.Module):
    def __init__(self, args, hidden_dim=64, latent_dim=32):
        """
        自编码器结构的信誉感知网络

        参数:
            args: 系统参数
            hidden_dim: 隐藏层维度
            latent_dim: 潜在空间维度
        """
        super(ReputationAwareNetwork, self).__init__()

        # 设置关键维度
        self.device_num = args.device_num
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 注意力机制的权重矩阵 - 使用更小的初始化范围避免大数值
        self.W_q = nn.Parameter(torch.randn(hidden_dim, self.device_num) * 0.1)
        self.W_k = nn.Parameter(torch.randn(hidden_dim, self.device_num) * 0.1)
        self.W_v = nn.Parameter(torch.randn(hidden_dim, self.device_num) * 0.1)

        # 计算投入特征的维度
        # R矩阵展平后的维度 + 注意力输出的维度
        input_feature_dim = (self.device_num * self.device_num) + hidden_dim

        # 编码器 (计算H_i)
        self.encoder = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        # 均值和方差投影层
        self.mean_projection = nn.Linear(hidden_dim, latent_dim)
        # 对数方差投影层 - 添加 tanh 激活函数限制输出范围
        self.var_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),  # 将输出限制在 [-1, 1] 范围内
            nn.Hardtanh(-4, 4)  # 进一步限制范围，防止指数爆炸
        )

        # 解码器 (重建R)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.device_num * self.device_num),
            nn.Sigmoid()  # 信誉值在[0,1]范围内
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化方法
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_H_i(self, R, R_Di, neighbor_R_Djs):
        """
        计算中间状态H_i

        参数:
            R: 信誉矩阵 [device_num x device_num]
            R_Di: 当前边缘服务器的设备信誉向量 [device_num]
            neighbor_R_Djs: 邻居边缘服务器的设备信誉向量列表 [list of device_num]
        """
        # 确保输入是张量
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32)
        if not isinstance(R_Di, torch.Tensor):
            R_Di = torch.tensor(R_Di, dtype=torch.float32)

        # 计算查询向量 (W_q R_Di)
        q = torch.matmul(self.W_q, R_Di.unsqueeze(-1)).squeeze(-1)  # [hidden_dim]

        # 初始化注意力和累计向量
        attention_sum = torch.zeros(self.hidden_dim, device=R_Di.device)

        # 处理每个邻居的信誉向量
        if neighbor_R_Djs and len(neighbor_R_Djs) > 0:
            # 缩放因子，用于注意力机制的数值稳定性
            scaling_factor = 1.0 / math.sqrt(self.hidden_dim)

            for R_Dj in neighbor_R_Djs:
                if not isinstance(R_Dj, torch.Tensor):
                    R_Dj = torch.tensor(R_Dj, dtype=torch.float32)

                # 计算键向量 (W_k R_Dj)
                k = torch.matmul(self.W_k, R_Dj.unsqueeze(-1)).squeeze(-1)  # [hidden_dim]

                # 计算注意力权重 (W_k R_Dj)^T (W_q R_Di)，添加缩放因子提高数值稳定性
                attention_weight = torch.dot(k, q) * scaling_factor

                # 限制注意力权重范围，防止极端值
                attention_weight = torch.clamp(attention_weight, -10.0, 10.0)

                # 计算值向量 (W_v R_Dj)
                v = torch.matmul(self.W_v, R_Dj.unsqueeze(-1)).squeeze(-1)  # [hidden_dim]

                # 加权求和
                weighted_v = v * attention_weight
                attention_sum += weighted_v

        # 展平信誉矩阵
        R_flat = R.view(-1)  # [device_num * device_num]

        # 连接注意力输出和展平的信誉矩阵
        combined = torch.cat([attention_sum, R_flat])

        # 通过编码器获取H_i
        H_i = self.encoder(combined)

        return H_i

    def forward(self, R, R_Di, neighbor_R_Djs):
        """
        前向传播

        参数:
            R: 信誉矩阵 [device_num x device_num]
            R_Di: 当前边缘服务器的设备信誉向量 [device_num]
            neighbor_R_Djs: 邻居边缘服务器的设备信誉向量列表 [list of device_num]
        返回:
            reconstructed_R: 重建的信誉矩阵
            C_i: 信誉认知向量
            C_i_mean: 信誉认知均值
            C_i_logvar: 信誉认知标准差
        """
        # 计算中间状态H_i
        H_i = self.compute_H_i(R, R_Di, neighbor_R_Djs)

        # 投影到潜在空间
        C_i_mean = self.mean_projection(H_i)  # 均值 [latent_dim]
        C_i_logvar = self.var_projection(H_i)  # 对数方差 [latent_dim]，现在已经被限制在安全范围内

        # 计算方差和标准差 - 添加数值稳定性
        C_i_var = torch.exp(C_i_logvar)  # 方差
        C_i_std = torch.sqrt(C_i_var + 1e-8)  # 添加小常数防止在零点附近的数值不稳定

        # 采样扰动 ε ~ N(0, 1)
        epsilon = torch.randn_like(C_i_std)

        # 构建信誉认知 C_i = \dot{C}_i + ε · \hat{C}_i
        C_i = C_i_mean + epsilon * C_i_std

        # 通过解码器重建信誉矩阵
        reconstructed_R_flat = self.decoder(C_i)

        # 确保输入信誉矩阵是张量
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32)

        # 重塑为矩阵形式
        reconstructed_R = reconstructed_R_flat.view(R.size())

        return reconstructed_R, C_i, C_i_mean, C_i_logvar

    def compute_loss(self, R, reconstructed_R, C_i, neighbor_C_i=None):
        """
        计算VAE损失函数

        参数:
            R: 原始信誉矩阵
            reconstructed_R: 重建的信誉矩阵
            C_i_mean: 潜在分布均值
            C_i_logvar: 潜在分布对数方差
            neighbor_C_i: 邻居的信誉认知向量列表 (用于邻居损失计算)
        返回:
            total_loss: 总损失
            recon_loss: 重建损失
            kl_loss: KL散度损失
        """
        # 重建损失 (MSE)
        recon_loss = F.mse_loss(reconstructed_R, R, reduction='sum')

        # 总损失
        total_loss = recon_loss
        neighbor_loss = 0

        # 如果有邻居信誉认知向量，计算邻居损失
        if neighbor_C_i is not None and len(neighbor_C_i) > 0:
            count = 0

            for C_j in neighbor_C_i:
                if not isinstance(C_j, torch.Tensor):
                    C_j = torch.tensor(C_j, dtype=torch.float32)

                # 确保张量在同一设备上
                if C_j.device != C_i.device:
                    C_j = C_j.to(C_i.device)

                C_i_log_softmax = F.log_softmax(C_i, dim=-1)
                C_j_softmax = F.softmax(C_j, dim=-1)

                # 计算KL散度
                kl = F.kl_div(C_i_log_softmax, C_j_softmax, reduction='sum')
                neighbor_loss += kl
                count += 1

            if count > 0:
                neighbor_loss = neighbor_loss / count
                # 将邻居损失添加到总损失中，使用权重因子来控制其影响
                total_loss += 0.1 * neighbor_loss

        return total_loss, recon_loss, neighbor_loss


class Multi_Actor(nn.Module):
    def __init__(self, args):
        super(Multi_Actor, self).__init__()
        self.args = args

        # 调整输入维度
        reputation_matrix_dim = args.device_num * args.device_num  # 信誉矩阵
        reputation_aware_dim = 32  # 信誉感知向量的维度，与ReputationAwareNetwork中的latent_dim相匹配
        task_vector_dim = args.device_num  # 任务向量
        network_topology_dim = args.device_num * args.neighbors_num  # 网络拓扑矩阵
        reputation_mask_dim = args.device_num  # 信誉屏蔽向量

        self.input_dim = reputation_matrix_dim + reputation_aware_dim + task_vector_dim + network_topology_dim + reputation_mask_dim
        self.hidden_dim = 256

        # 网络结构
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout1 = nn.Dropout(0.2)  # 添加dropout防止过拟合
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(0.2)  # 添加dropout防止过拟合

        # 决策头 - 每个设备选择卸载目标设备
        # neighbors_num + 2: neighbors_num个邻居 + 1个不卸载选项 + 1个向相邻服务器卸载选项
        self.offload_head = nn.Linear(self.hidden_dim, args.device_num * (args.neighbors_num + 2))

        # 服务器选择头 - 如果向相邻服务器卸载，选择哪个服务器
        self.server_head = nn.Linear(self.hidden_dim, args.device_num * 2)

        # 初始化参数，帮助训练开始
        self._init_weights()

    def _init_weights(self):
        # 对网络参数进行Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, reputation_matrix, reputation_aware, task_vector, neighbor_matrix, reputation_mask):
        # 处理输入张量，确保维度正确
        if not isinstance(reputation_matrix, torch.Tensor):
            reputation_matrix_tensor = torch.tensor(reputation_matrix, dtype=torch.float32).flatten()
        else:
            reputation_matrix_tensor = reputation_matrix.float().flatten()

        if not isinstance(reputation_aware, torch.Tensor):
            reputation_aware_tensor = torch.tensor(reputation_aware, dtype=torch.float32).flatten()
        else:
            reputation_aware_tensor = reputation_aware.float().flatten()

        if not isinstance(task_vector, torch.Tensor):
            task_vector_tensor = torch.tensor(task_vector, dtype=torch.float32)
        else:
            task_vector_tensor = task_vector.float()

        if not isinstance(neighbor_matrix, torch.Tensor):
            neighbor_matrix_tensor = torch.tensor(neighbor_matrix, dtype=torch.float32).flatten()
        else:
            neighbor_matrix_tensor = neighbor_matrix.float().flatten()

        if not isinstance(reputation_mask, torch.Tensor):
            reputation_mask_tensor = torch.tensor(reputation_mask, dtype=torch.float32)
        else:
            reputation_mask_tensor = reputation_mask.float()

        # 添加小的噪声帮助打破梯度为0的情况
        noise_factor = 1e-4
        if self.training:
            reputation_matrix_tensor = reputation_matrix_tensor + torch.randn_like(
                reputation_matrix_tensor) * noise_factor
            reputation_aware_tensor = reputation_aware_tensor + torch.randn_like(reputation_aware_tensor) * noise_factor
            task_vector_tensor = task_vector_tensor + torch.randn_like(task_vector_tensor) * noise_factor
            neighbor_matrix_tensor = neighbor_matrix_tensor + torch.randn_like(neighbor_matrix_tensor) * noise_factor

        # 合并输入特征
        combined_input = torch.cat([
            reputation_matrix_tensor,
            reputation_aware_tensor,
            task_vector_tensor,
            neighbor_matrix_tensor,
            reputation_mask_tensor
        ])



        # 前向传播
        x = F.relu(self.fc1(combined_input))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # 生成卸载目标决策
        offload_logits = self.offload_head(x)
        offload_probs = offload_logits.reshape(self.args.device_num, self.args.neighbors_num + 2)
        offload_probs = F.softmax(offload_probs, dim=1)

        # 生成服务器选择决策
        server_logits = self.server_head(x)
        server_probs = server_logits.reshape(self.args.device_num, 2)
        server_probs = F.softmax(server_probs, dim=1)

        mask_with_self = torch.ones(self.args.device_num, self.args.neighbors_num + 2)

        for i in range(self.args.device_num):
            for j in range(1, self.args.neighbors_num + 1):
                neighbor_idx = int(neighbor_matrix[i][j - 1])  # 防止索引越界
                mask_with_self[i, j] = reputation_mask_tensor[neighbor_idx]

        # 应用掩码并重新归一化
        masked_offload_probs = offload_probs * mask_with_self
        row_sums = masked_offload_probs.sum(dim=1, keepdim=True)
        masked_offload_probs = masked_offload_probs / row_sums.clamp(min=1e-8)  # 防止除以0

        return [masked_offload_probs, server_probs]


class Multi_Critic(nn.Module):
    def __init__(self, args):
        super(Multi_Critic, self).__init__()
        self.args = args

        # 基本状态维度
        dynamic_re_dim = args.device_num * args.device_num
        static_re_dim = args.device_num
        task_em_dim = args.device_num
        r_mask_dim = args.device_num
        neighbor_matrix_dim = args.device_num * args.neighbors_num
        reputation_aware_dim = 32  # 信誉感知向量的维度，与ReputationAwareNetwork中的latent_dim相匹配

        # 动作维度更新
        offload_action_dim = args.device_num * (args.neighbors_num + 2)  # 卸载决策
        server_action_dim = args.device_num * 2  # 服务器选择
        self.action_dim = offload_action_dim + server_action_dim

        self.state_input_dim = dynamic_re_dim + static_re_dim + task_em_dim + r_mask_dim + neighbor_matrix_dim + reputation_aware_dim

        # 状态编码层
        self.state_fc1 = nn.Linear(self.state_input_dim, 256)
        self.state_fc2 = nn.Linear(256, 128)

        # 动作编码层
        self.action_fc1 = nn.Linear(self.action_dim, 128)

        # 联合层
        self.joint_fc1 = nn.Linear(256, 128)
        self.joint_fc2 = nn.Linear(128, 64)
        self.joint_fc3 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dynamic_re, static_re, task_em, neighbor_matrix, r_mask, reputation_aware, action):
        if not isinstance(dynamic_re, torch.Tensor):
            dynamic_re_tensor = torch.tensor(dynamic_re, dtype=torch.float32)
        else:
            dynamic_re_tensor = dynamic_re.float()

        if not isinstance(static_re, torch.Tensor):
            static_re_tensor = torch.tensor(static_re, dtype=torch.float32)
        else:
            static_re_tensor = static_re.float()

        if not isinstance(task_em, torch.Tensor):
            task_em_tensor = torch.tensor(task_em, dtype=torch.float32)
        else:
            task_em_tensor = task_em.float()

        if not isinstance(neighbor_matrix, torch.Tensor):
            neighbor_matrix_tensor = torch.tensor(neighbor_matrix, dtype=torch.float32)
        else:
            neighbor_matrix_tensor = neighbor_matrix.float()

        if not isinstance(r_mask, torch.Tensor):
            r_mask_tensor = torch.tensor(r_mask, dtype=torch.float32)
        else:
            r_mask_tensor = r_mask.float()

        if not isinstance(reputation_aware, torch.Tensor):
            reputation_aware_tensor = torch.tensor(reputation_aware, dtype=torch.float32)
        else:
            reputation_aware_tensor = reputation_aware.float()


        # 处理动作输入 - action是一个列表，包含两个张量
        offload_action, server_action = action

        if not isinstance(offload_action, torch.Tensor):
            offload_action_tensor = torch.tensor(offload_action, dtype=torch.float32)
        else:
            offload_action_tensor = offload_action.float()

        if not isinstance(server_action, torch.Tensor):
            server_action_tensor = torch.tensor(server_action, dtype=torch.float32)
        else:
            server_action_tensor = server_action.float()

        dynamic_re_flat = dynamic_re_tensor.reshape(-1)
        neighbor_matrix_flat = neighbor_matrix_tensor.reshape(-1)
        reputation_aware_flat = reputation_aware_tensor.reshape(-1)
        offload_action_flat = offload_action_tensor.reshape(-1)
        server_action_flat = server_action_tensor.reshape(-1)

        # 合并状态特征
        state_input = torch.cat([
            dynamic_re_flat,
            static_re_tensor,
            task_em_tensor,
            neighbor_matrix_flat,
            r_mask_tensor,
            reputation_aware_flat
        ], dim=0)

        # 合并动作特征
        action_input = torch.cat([
            offload_action_flat,
            server_action_flat
        ], dim=0)

        # 状态编码
        state_x = F.relu(self.state_fc1(state_input))
        state_x = F.dropout(state_x, p=0.2, training=self.training)
        state_features = F.relu(self.state_fc2(state_x))

        # 动作编码
        action_features = F.relu(self.action_fc1(action_input))
        action_features = F.dropout(action_features, p=0.2, training=self.training)

        # 合并特征
        combined = torch.cat([state_features, action_features])

        # 最终层
        x = F.relu(self.joint_fc1(combined))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.joint_fc2(x))
        q_value = self.joint_fc3(x)

        return q_value