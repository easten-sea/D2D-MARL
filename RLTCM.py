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
from re_agent import RLTCM_agent

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    # Parameter parsing
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

    # Parameters for reward calculation
    ap.add_argument('--V', type=float, default=50, help='李雅普诺夫优化的漂移函数的惩罚项权重')
    ap.add_argument('--gamma', type=float, default=0.5, help='未来奖励的折扣率')

    # Parameters for learning
    ap.add_argument('--learning_rate', type=float, default=0.001, help='学习率，取值（0，1）')
    ap.add_argument('--batch_size', type=float, default=16, help='学习的batchsize')
    ap.add_argument('--tau', type=float, default=0.001, help='目标网络的软更新参数')
    ap.add_argument('--target_update_freq', type=int, default=5, help='目标网络的更新频率')

    # Malicious device ratio and reputation parameters
    ap.add_argument('--malicious', type=float, default=0.2, help='恶意设备占比')
    ap.add_argument('--se_ca', type=float, default=0.9, help='设备安全级别和能力级别参数')
    ap.add_argument('--su_fa', type=float, default=1, help='任务成功和失败参数')
    ap.add_argument('--re_mask', type=float, default=0.2, help='信誉屏蔽阈值')
    ap.add_argument('--re_ba', type=float, default=0.9, help='信誉矩阵参数')

    args = ap.parse_args()
    args.task_size_range = [4, 10]
    args.max_tolerance_range = [5, 15]  # 时隙
    args.max_hops_range = [1, 3]  # 跳数

    # Add device to args for easy access
    args.device = device

    # Configure logging
    log_file = "re_log/RLTCM_episodes_{}_malicious_{}_device_{}_task_{}.txt".format(args.epochs, args.malicious,
                                                                                    args.device_num,
                                                                                    args.task_arrival_prob)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        logging.info(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

    setup_seed(args.seed_num)

    dataset_generator = TaskOffloadingDataset(args)
    dataset_path = 're_dataset/episodes_{}_poisson-rate_{}_device-num_{}_dataset.pkl'.format(args.epochs,
                                                                                             args.task_arrival_prob,
                                                                                             args.device_num)

    if not os.path.exists(dataset_path):
        stats = dataset_generator.get_statistics()
        print("Dataset statistics:")
        for key, value in stats.items():
            if key != 'tasks_per_device':
                print(f"- {key}: {value}")
        dataset_generator.save_dataset(dataset_path)
        vis_path = 're_dataset/episodes_{}_poisson-rate_{}_device-num_{}_visualization.png'.format(args.epochs,
                                                                                                   args.task_arrival_prob,
                                                                                                   args.device_num)
        dataset_generator.visualize_dataset(epoch=0, save_path=vis_path)

    loaded_dataset = TaskOffloadingDataset.load_dataset(dataset_path)
    torch_dataset = dataset_generator.get_torch_dataset()

    for key in torch_dataset:
        if isinstance(torch_dataset[key], torch.Tensor):
            torch_dataset[key] = torch_dataset[key].to(device)

    devices = {}
    for i in range(args.device_num):
        if i < args.device_num * args.malicious:
            devices.update({
                i: [
                    0,
                    random.randint(3, 7) / 10,
                    random.randint(8, 10) / 10,
                ]
            })
        else:
            devices.update({
                i: [
                    1,
                    random.randint(5, 10) / 10,
                    random.randint(8, 10) / 10,
                ]
            })


    neighbor_manager = DynamicNeighborMatrix(args)
    agent = RLTCM_agent(args)
    agent.actor = agent.actor.to(device)
    agent.critic = agent.critic.to(device)

    for epoch in range(args.epochs):
        task_suc = np.zeros([args.device_num, args.device_num])
        task_fail = np.zeros([args.device_num, args.device_num])
        reward_buffer = []

        neighbor_matrix = neighbor_manager.update_epoch(epoch=0)
        neighbor_matrix_tensor = torch.tensor(neighbor_matrix, dtype=torch.long, device=device)

        epoch_slot_data = get_epoch_slot_data(torch_dataset, epoch, 0)

        task_em = epoch_slot_data['task_arrivals']
        task_size = epoch_slot_data['task_sizes']
        if not isinstance(task_em, torch.Tensor):
            task_em = torch.tensor(task_em, dtype=torch.float32, device=device)
        else:
            task_em = task_em.to(device)

        static_re = [0 for i in range(args.device_num)]
        for i in range(args.device_num):
            co = 0
            for j in range(args.neighbors_num):
                co += devices[neighbor_matrix[i][j]][1] + devices[neighbor_matrix[i][j]][2]
            static_re[i] = args.se_ca * (devices[i][1] + devices[i][2]) + (1 - args.se_ca) * co / args.neighbors_num

        static_re_tensor = torch.tensor(static_re, dtype=torch.float32, device=device)

        dynamic_re = (task_suc + 1) / (task_suc + args.su_fa * task_fail + 2)
        dynamic_re_tensor = torch.tensor(dynamic_re, dtype=torch.float32, device=device)
        current_state = (task_em.clone(), dynamic_re_tensor.clone(), static_re_tensor.clone(), neighbor_matrix_tensor.clone())

        ave_reward = 0
        ave_fail = 0
        ave_en = 0
        pre_fail = 0

        for t in range(args.T):
            fail_task = 0
            task_num = 0

            with torch.no_grad():
                r_data = agent.re_net.forward(dynamic_re_tensor, static_re_tensor, neighbor_matrix_tensor)
            reputation_matrix = r_data
            r_d = torch.mean(reputation_matrix, dim=1)
            r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

            agent.actor.eval()
            actions = agent.actor(reputation_matrix, task_em, neighbor_matrix_tensor, r_mask)

            e_reward = 0
            r_reward = 0
            de_fail = 0
            ex_fail = 0

            for i in range(len(actions)):
                if torch.argmax(actions[i]) == 0:
                    if task_em[i]:
                        fail_task += 1
                        task_num += 1
                        de_fail += 1
                    continue
                else:
                    neighbor_index = torch.argmax(actions[i]) - 1
                    neighbor_id = neighbor_matrix[i][neighbor_index.item()]
                    e_reward -= (calculate_transmission_energy(neighbor_index + 1, task_size[i]) * (1 - r_d[neighbor_id]))
                    task_num += 1
                    if devices[neighbor_id][0] == 0:
                        ex_fail += 1
                        fail_task += 1
                        task_fail[i][neighbor_id] += 1
                    else:
                        task_suc[i][neighbor_id] += 1

            if task_num > 0:
                failure_rate = fail_task / task_num
                failure_penalty = -100 * np.exp(2.5 * failure_rate)
                energy_reward = e_reward
                reward = failure_penalty + energy_reward
            else:
                reward = e_reward

            reward_buffer.append(reward)
            if len(reward_buffer) > 1000:
                reward_buffer.pop(0)
            reward_mean = sum(reward_buffer) / len(reward_buffer)
            reward_std = math.sqrt(sum((r - reward_mean) ** 2 for r in reward_buffer) / len(reward_buffer)) + 1e-6
            normalized_reward = (reward - reward_mean) / reward_std

            ave_reward += reward
            ave_en -= e_reward * 10
            pre_fail = fail_task / task_num if task_num > 0 else pre_fail
            ave_fail += fail_task / task_num if task_num > 0 else pre_fail
            neighbor_matrix = neighbor_manager.update_epoch(epoch=t + 1)
            neighbor_matrix_tensor = torch.tensor(neighbor_matrix, dtype=torch.long, device=device)
            epoch_slot_data = get_epoch_slot_data(torch_dataset, epoch, t + 1)
            task_em = epoch_slot_data['task_arrivals']
            task_size = epoch_slot_data['task_sizes']

            if not isinstance(task_em, torch.Tensor):
                task_em = torch.tensor(task_em, dtype=torch.float32, device=device)
            else:
                task_em = task_em.to(device)

            for i in range(args.device_num):
                co = 0
                for j in range(args.neighbors_num):
                    co += devices[neighbor_matrix[i][j]][1] + devices[neighbor_matrix[i][j]][2]
                static_re[i] = args.se_ca * (devices[i][1] + devices[i][2]) + (1 - args.se_ca) * co / args.neighbors_num

            static_re_tensor = torch.tensor(static_re, dtype=torch.float32, device=device)

            dynamic_re = (task_suc + 1) / (task_suc + args.su_fa * task_fail + 2)
            dynamic_re_tensor = torch.tensor(dynamic_re, dtype=torch.float32, device=device)

            next_state = (task_em.clone(), dynamic_re_tensor.clone(), static_re_tensor.clone(), neighbor_matrix_tensor.clone())

            done = (t == args.T - 1)
            agent.replay_buffer.push(current_state, actions.clone(), reward, next_state, done)
            current_state = next_state
            if t % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        logging.info("epoch:{} | 奖励： {} | 任务失败率: {} | 能耗:{}".format(epoch, ave_reward / args.T, ave_fail / args.T, ave_en / args.T))

        if len(agent.replay_buffer) >= args.batch_size:
            logging.info(f"Epoch {epoch} completed, starting training...")
            train_iterations = 1
            total_actor_loss = 0
            total_critic_loss = 0
            total_reputation_loss = 0

            for _ in range(train_iterations):
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = agent.replay_buffer.sample(int(args.batch_size))
                reward_tensor = torch.FloatTensor(reward_batch).to(device)
                done_tensor = torch.FloatTensor(done_batch).to(device)

                # ==================== Train Critic network ====================
                agent.critic.train()
                critic_values = []
                next_values = []

                for i in range(len(state_batch)):
                    with torch.no_grad():
                        task_em, dynamic_re, static_re, neighbor_matrix = state_batch[i]
                        next_task_em, next_dynamic_re, next_static_re, next_neighbor_matrix = next_state_batch[i]
                        if not isinstance(task_em, torch.Tensor):
                            task_em = torch.tensor(task_em, dtype=torch.float32, device=device)
                        elif task_em.device != device:
                            task_em = task_em.to(device)

                        if not isinstance(dynamic_re, torch.Tensor):
                            dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32, device=device)
                        elif dynamic_re.device != device:
                            dynamic_re = dynamic_re.to(device)

                        if not isinstance(static_re, torch.Tensor):
                            static_re = torch.tensor(static_re, dtype=torch.float32, device=device)
                        elif static_re.device != device:
                            static_re = static_re.to(device)

                        if not isinstance(neighbor_matrix, torch.Tensor):
                            neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.long, device=device)
                        elif neighbor_matrix.device != device:
                            neighbor_matrix = neighbor_matrix.to(device)

                        if not isinstance(next_task_em, torch.Tensor):
                            next_task_em = torch.tensor(next_task_em, dtype=torch.float32, device=device)
                        elif next_task_em.device != device:
                            next_task_em = next_task_em.to(device)

                        if not isinstance(next_dynamic_re, torch.Tensor):
                            next_dynamic_re = torch.tensor(next_dynamic_re, dtype=torch.float32, device=device)
                        elif next_dynamic_re.device != device:
                            next_dynamic_re = next_dynamic_re.to(device)

                        if not isinstance(next_static_re, torch.Tensor):
                            next_static_re = torch.tensor(next_static_re, dtype=torch.float32, device=device)
                        elif next_static_re.device != device:
                            next_static_re = next_static_re.to(device)

                        if not isinstance(next_neighbor_matrix, torch.Tensor):
                            next_neighbor_matrix = torch.tensor(next_neighbor_matrix, dtype=torch.long, device=device)
                        elif next_neighbor_matrix.device != device:
                            next_neighbor_matrix = next_neighbor_matrix.to(device)
                        current_action = action_batch[i].clone().detach()
                        if current_action.device != device:
                            current_action = current_action.to(device)

                        reputation_matrix = agent.re_net.forward(dynamic_re, static_re, neighbor_matrix)
                        r_d = torch.mean(reputation_matrix, dim=1)
                        r_mask = torch.sigmoid((r_d - args.re_mask) * 10)

                        next_r_data = agent.re_net.forward(next_dynamic_re, next_static_re, next_neighbor_matrix)
                        next_reputation_matrix = next_r_data
                        next_r_d = torch.mean(next_reputation_matrix, dim=1)
                        next_r_mask = torch.sigmoid((next_r_d - args.re_mask) * 10)
                        next_actions = agent.actor(next_reputation_matrix, next_task_em, next_neighbor_matrix,
                                                   next_r_mask)

                    q_value = agent.critic(dynamic_re, static_re, task_em, neighbor_matrix, r_mask, current_action)
                    with torch.no_grad():
                        next_q_value = agent.critic(next_dynamic_re, next_static_re, next_task_em, next_neighbor_matrix,
                                                    next_r_mask, next_actions)

                    critic_values.append(q_value)
                    next_values.append(next_q_value)
                critic_values = torch.cat(critic_values)
                next_values = torch.cat(next_values)
                target_q_values = reward_tensor + (1 - done_tensor) * args.gamma * next_values
                critic_loss = F.smooth_l1_loss(critic_values, target_q_values)
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)
                agent.critic_optimizer.step()
                total_critic_loss += critic_loss.item()

                agent.actor.train()
                actor_loss = 0

                for i in range(len(state_batch)):
                    task_em, dynamic_re, static_re, neighbor_matrix = state_batch[i]
                    if not isinstance(dynamic_re, torch.Tensor):
                        dynamic_re = torch.tensor(dynamic_re, dtype=torch.float32, device=device)
                    elif dynamic_re.device != device:
                        dynamic_re = dynamic_re.to(device)

                    if not isinstance(static_re, torch.Tensor):
                        static_re = torch.tensor(static_re, dtype=torch.float32, device=device)
                    elif static_re.device != device:
                        static_re = static_re.to(device)

                    if not isinstance(task_em, torch.Tensor):
                        task_em = torch.tensor(task_em, dtype=torch.float32, device=device)
                    elif task_em.device != device:
                        task_em = task_em.to(device)

                    if not isinstance(neighbor_matrix, torch.Tensor):
                        neighbor_matrix = torch.tensor(neighbor_matrix, dtype=torch.long, device=device)
                    elif neighbor_matrix.device != device:
                        neighbor_matrix = neighbor_matrix.to(device)
                    with torch.no_grad():
                        reputation_matrix = agent.re_net.forward(dynamic_re, static_re, neighbor_matrix)
                    r_d = torch.mean(reputation_matrix, dim=1)
                    r_mask = torch.sigmoid((r_d - args.re_mask) * 10)
                    current_actions = agent.actor(reputation_matrix, task_em, neighbor_matrix, r_mask)
                    agent.critic.eval()
                    q_value = agent.critic(dynamic_re, static_re, task_em, neighbor_matrix, r_mask, current_actions)
                    batch_policy_loss = -q_value.mean()
                    actor_loss += batch_policy_loss
                actor_loss = actor_loss / len(state_batch)

                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
                agent.actor_optimizer.step()
                total_actor_loss += actor_loss.item()

            avg_actor_loss = total_actor_loss / train_iterations
            avg_critic_loss = total_critic_loss / train_iterations
            avg_reputation_loss = total_reputation_loss / train_iterations

            logging.info(f"Epoch {epoch} training completed:")
            logging.info(f"  - Actor loss: {avg_actor_loss:.4f}")
            logging.info(f"  - Critic loss: {avg_critic_loss:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                save_path = f"re_model/RLTCM_model_device_{args.device_num}_task_{args.task_arrival_prob}_malicious_{args.malicious}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict()
                }, save_path)
                logging.info(f"Model saved to: {save_path}")

            if device.type == 'cuda':
                torch.cuda.empty_cache()
                logging.info(f"Memory usage after epoch: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")