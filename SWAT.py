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

from re_util import DynamicNeighborMatrix, TaskOffloadingDataset, get_epoch_slot_data, calculate_transmission_energy
from re_agent import re_agent, swat_agent, create_robust_trust_mechanism


# 检查CUDA是否可用
cuda_enable = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ap.add_argument('--batch_size', type=float, default=16, help='学习的batchsize')
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
    log_file = "re_log/SWAT_episodes_{}_malicious_{}_device_{}_task_{}.txt".format(args.epochs,
                                                                                           args.malicious,
                                                                                           args.device_num,
                                                                                           args.task_arrival_prob)
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
    dataset_path = 're_dataset/episodes_{}_poisson-rate_{}_device-num_{}_dataset.pkl'.format(args.epochs,args.task_arrival_prob,args.device_num)

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
        vis_path = 're_dataset/episodes_{}_poisson-rate_{}_device-num_{}_visualization.png'.format(args.epochs,
                                                                                                   args.task_arrival_prob,
                                                                                                   args.device_num)
        dataset_generator.visualize_dataset(epoch=0, save_path=vis_path)

    # 从文件加载数据集
    loaded_dataset = TaskOffloadingDataset.load_dataset(dataset_path)

    # 获取PyTorch形式的数据集
    torch_dataset = dataset_generator.get_torch_dataset()

    # 初始化设备属性（包括是否为恶意设备）
    devices = {}
    for i in range(args.device_num):
        if i < args.device_num * args.malicious:  # 恶意设备
            malicious_type = 0
            devices.update({
                i: [
                    0,  # 恶意标志
                    random.randint(3, 7) / 10,  # 较低的安全级别
                    random.randint(8, 10) / 10,  # 能力级别
                ]
            })
        else:
            devices.update({
                i: [
                    1,  # 正常标志
                    random.randint(5, 10) / 10,  # 较高的安全级别
                    random.randint(8, 10) / 10,  # 能力级别
                ]
            })

    # 初始化邻居管理器
    neighbor_manager = DynamicNeighborMatrix(args)

    # 初始化智能体
    agent = swat_agent(args)
    agent.re_optimizer = None

    # 开始训练循环
    for epoch in range(args.epochs):
        # 初始化任务记录矩阵
        task_suc = np.zeros([args.device_num, args.device_num])
        task_fail = np.zeros([args.device_num, args.device_num])
        reward_buffer = []

        # 更新邻居矩阵
        neighbor_matrix = neighbor_manager.update_epoch(epoch=0)

        # 获取当前时隙的数据
        epoch_slot_data = get_epoch_slot_data(torch_dataset, epoch, 0)

        # 任务向量
        task_em = epoch_slot_data['task_arrivals']
        task_size = epoch_slot_data['task_sizes']

        # 计算静态安全向量
        static_re = [0 for i in range(args.device_num)]
        for i in range(args.device_num):
            co = 0
            for j in range(args.neighbors_num):
                co += devices[neighbor_matrix[i][j]][1] + devices[neighbor_matrix[i][j]][2]
            static_re[i] = args.se_ca * (devices[i][1] + devices[i][2]) + (1 - args.se_ca) * co / args.neighbors_num

        # 计算动态安全向量 (基于历史交互)
        dynamic_re = (task_suc + 1) / (task_suc + args.su_fa * task_fail + 2)

        # 设置当前状态
        current_state = (task_em.clone(), dynamic_re.copy(), static_re.copy(), neighbor_matrix.copy())

        # 初始化性能指标
        ave_reward = 0
        ave_fail = 0
        ave_en = 0
        pre_fail = 0

        # 时隙循环
        for t in range(args.T):
            fail_task = 0
            task_num = 0

            # 生成信誉矩阵 (使用鲁棒多维信任计算机制)
            with torch.no_grad():
                reputation_matrix = agent.re_net(dynamic_re, static_re, neighbor_matrix)
            r_d = torch.mean(reputation_matrix, dim=1)
            r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

            # 生成卸载决策
            agent.actor.eval()
            actions = agent.actor(reputation_matrix, task_em, neighbor_matrix, r_mask)

            # 计算奖励和统计任务执行结果
            e_reward = 0
            de_fail = 0
            ex_fail = 0

            for i in range(len(actions)):
                if torch.argmax(actions[i]) == 0:
                    if task_em[i]:  # 有任务但不卸载，视为失败
                        fail_task += 1
                        task_num += 1
                        de_fail += 1
                    continue
                else:
                    neighbor_index = torch.argmax(actions[i]) - 1
                    target_device_id = neighbor_matrix[i][neighbor_index]

                    # 计算能耗奖励 (与设备信誉相关)
                    e_reward -= (calculate_transmission_energy(neighbor_index + 1, task_size[i]) * (1 - r_d[target_device_id]))
                    task_num += 1

                    # 判断任务是否成功 (根据目标设备是否为恶意设备)
                    if devices[target_device_id][0] == 0:  # 恶意设备
                        ex_fail += 1
                        fail_task += 1
                        task_fail[i][target_device_id] += 1
                    else:
                        task_suc[i][target_device_id] += 1
            if task_num > 0:
                failure_rate = fail_task / task_num
                failure_penalty = -100 * np.exp(2.5 * failure_rate)
                energy_reward = e_reward
                reward = failure_penalty + energy_reward
            else:
                reward = e_reward

            # 奖励标准化处理
            reward_buffer.append(reward)
            if len(reward_buffer) > 1000:
                reward_buffer.pop(0)
            reward_mean = sum(reward_buffer) / len(reward_buffer)
            reward_std = math.sqrt(sum((r - reward_mean) ** 2 for r in reward_buffer) / len(reward_buffer)) + 1e-6
            normalized_reward = (reward - reward_mean) / reward_std

            # 更新统计值
            ave_reward += reward
            ave_en -= e_reward * 10
            pre_fail = fail_task / task_num if task_num > 0 else pre_fail
            ave_fail += fail_task / task_num if task_num > 0 else pre_fail

            # 更新邻居矩阵和状态
            neighbor_matrix = neighbor_manager.update_epoch(epoch=t + 1)
            epoch_slot_data = get_epoch_slot_data(torch_dataset, epoch, t + 1)
            task_em = epoch_slot_data['task_arrivals']
            task_size = epoch_slot_data['task_sizes']

            # 重新计算静态信誉
            for i in range(args.device_num):
                co = 0
                for j in range(args.neighbors_num):
                    co += devices[neighbor_matrix[i][j]][1] + devices[neighbor_matrix[i][j]][2]
                static_re[i] = args.se_ca * (devices[i][1] + devices[i][2]) + (1 - args.se_ca) * co / args.neighbors_num

            # 更新动态信誉
            dynamic_re = (task_suc + 1) / (task_suc + args.su_fa * task_fail + 2)

            # 设置下一状态
            next_state = (task_em.clone(), dynamic_re.copy(), static_re.copy(), neighbor_matrix.copy())

            # 判断是否为最后一个时间步
            done = (t == args.T - 1)

            # 将经验存入回放缓冲区
            agent.replay_buffer.push(current_state, actions.clone(), reward, next_state, done)

            # 更新当前状态
            current_state = next_state

        # 记录epoch性能
        logging.info("epoch:{} | 奖励： {} | 任务失败率: {} | 能耗:{}".format(
            epoch, ave_reward / args.T, ave_fail / args.T, ave_en / args.T
        ))

        # 训练智能体
        if len(agent.replay_buffer) >= args.batch_size:
            logging.info(f"Epoch {epoch} 完成，开始训练...")

            # 训练次数
            train_iterations = 1
            total_actor_loss = 0
            total_critic_loss = 0

            for _ in range(train_iterations):
                # 从回放缓冲区采样
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = agent.replay_buffer.sample(
                    int(args.batch_size))

                # 转换为张量
                reward_tensor = torch.FloatTensor(reward_batch)
                done_tensor = torch.FloatTensor(done_batch)

                # ==================== 训练Critic网络 ====================
                agent.critic.train()
                critic_values = []
                next_values = []

                for i in range(len(state_batch)):
                    with torch.no_grad():
                        # 提取状态信息
                        task_em, dynamic_re, static_re, neighbor_matrix = state_batch[i]
                        next_task_em, next_dynamic_re, next_static_re, next_neighbor_matrix = next_state_batch[i]

                        # 提取动作信息
                        current_action = action_batch[i].clone().detach()

                        # 使用鲁棒多维信任计算机制生成信誉矩阵
                        reputation_matrix = agent.re_net(dynamic_re, static_re, neighbor_matrix)
                        r_d = torch.mean(reputation_matrix, dim=1)
                        r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

                        # 下一状态的信誉矩阵
                        next_reputation_matrix = agent.re_net(next_dynamic_re, next_static_re, next_neighbor_matrix)
                        next_r_d = torch.mean(next_reputation_matrix, dim=1)
                        next_r_mask = torch.sigmoid((next_r_d - args.re_mask) * 10)

                        # 获取下一状态的最佳动作
                        next_actions = agent.actor(next_reputation_matrix, next_task_em, next_neighbor_matrix,
                                                   next_r_mask)

                    # 获取当前Q值
                    q_value = agent.critic(dynamic_re, static_re, task_em, neighbor_matrix, r_mask, current_action)

                    # 获取下一状态的Q值
                    with torch.no_grad():
                        next_q_value = agent.critic(next_dynamic_re, next_static_re, next_task_em, next_neighbor_matrix,
                                                    next_r_mask, next_actions)

                    critic_values.append(q_value)
                    next_values.append(next_q_value)

                # 转换为张量
                critic_values = torch.cat(critic_values)
                next_values = torch.cat(next_values)

                # 计算目标Q值 (TD目标)
                target_q_values = reward_tensor + (1 - done_tensor) * args.gamma * next_values

                # 计算TD误差
                critic_loss = F.smooth_l1_loss(critic_values, target_q_values)

                # 更新Critic
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)
                agent.critic_optimizer.step()
                total_critic_loss += critic_loss.item()

                # ==================== 训练Actor网络 ====================
                agent.actor.train()
                actor_loss = 0

                for i in range(len(state_batch)):
                    task_em, dynamic_re, static_re, neighbor_matrix = state_batch[i]

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
                    reputation_matrix = agent.re_net(dynamic_re, static_re, neighbor_matrix)

                    # 计算信誉掩码
                    r_d = torch.mean(reputation_matrix, dim=1)
                    r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

                    # 获取Actor的当前策略动作
                    current_actions = agent.actor(reputation_matrix, task_em, neighbor_matrix, r_mask)

                    # 获取当前策略的Q值
                    agent.critic.eval()
                    q_value = agent.critic(dynamic_re, static_re, task_em, neighbor_matrix, r_mask, current_actions)

                    # 策略损失 = -Q值 (我们希望最大化Q值，因此取负)
                    batch_policy_loss = - q_value.mean()
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

            logging.info(f"Epoch {epoch} 训练完成:")
            logging.info(f"  - Actor损失: {avg_actor_loss:.4f}")
            logging.info(f"  - Critic损失: {avg_critic_loss:.4f}")

            # 定期保存模型
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                save_path = f"re_model/SWAT_device_{args.device_num}_task_{args.task_arrival_prob}_malicious_{args.malicious}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                }, save_path)
                logging.info(f"模型保存至: {save_path}")