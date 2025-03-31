import os.path

import torch
from sysmodel import task, environment, get_observation, ddpg_gen_observation, update_neighbors
from MARL import agent, setup_seed
import numpy as np
import argparse
import random
import math
from sysmodel import gen_observation
import time
from MARL import simulate_task_arrivals

def gen_dataset(args, episode_num):
    for episode in range(episode_num):
        global_index = 0
        for t in range(args.T - args.cost_time_max):
            for agent in range(args.device_num):
                task_arrivals = simulate_task_arrivals((args.poisson_rate + 0.1 * agent / args.device_num) / 2, args.T, args.seed_num + episode)
                if not os.path.exists('dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}/episode_{}'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max, episode)):
                    os.mkdir('dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}/episode_{}'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max, episode))
                with open('dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}/episode_{}/train_eposide_{}.txt'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max, episode, agent), 'a') as file:
                    file.write(str(t) + '\n')
                    for i in range(task_arrivals[t]):
                        ts = task(args, t, global_index)
                        while ts.ddl <= (t + math.ceil(ts.size / (args.cpu_max * args.T_slot))):
                            ts = task(args, t, global_index)
                        global_index += 1
                        file.write(str(ts.index) + '\t' + str(ts.size) + '\t' + str(ts.ddl) + '\t' + str(ts.hop) + '\n')

def re_gen_dataset(args, episode_num):
    for episode in range(episode_num):
        global_index = 0
        for t in range(args.T - args.cost_time_max):
            for agent in range(args.device_num):
                task_arrivals = simulate_task_arrivals((args.poisson_rate + 0.1 * agent / args.device_num) / 2, args.T, args.seed_num + episode)
                if not os.path.exists('re_dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}/episode_{}'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max, episode)):
                    os.mkdir('re_dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}/episode_{}'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max, episode))
                with open('re_dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}/episode_{}/train_eposide_{}.txt'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max, episode, agent), 'a') as file:
                    file.write(str(t) + '\n')
                    for i in range(task_arrivals[t]):
                        ts = task(args, t, global_index)
                        while ts.ddl <= (t + math.ceil(ts.size / (args.cpu_max * args.T_slot))):
                            ts = task(args, t, global_index)
                        global_index += 1
                        file.write(str(ts.index) + '\t' + str(ts.size) + '\t' + str(ts.ddl) + '\t' + str(ts.hop) + '\n')



def run_eposide(args, logging, epoch, agents, env, cpu_fres):
    logging.info("开始运行episode--{}".format(epoch))
    start_time = time.time()
    fail_task_num = 0
    new_ob, new_env_kno = gen_observation(args, agents, epoch, 0)
    total_rewards = [0.0 for i in agents]
    total_energy_cost = [0.0 for i in agents]
    get_rewards = 0.0
    get_energy_cost = 0.0
    rewards_num = 0
    energy_cost_num = 0
    p_backlog = 0
    q_backlog = 0
    for t in range(args.T + 100):
        time_offloadings = []
        observations = new_ob
        env_kno = new_env_kno
        mask = [0 for i in range(len(agents))]
        actions = [torch.zeros(args.act_em_size) for i in range(len(agents))]

        for agent in agents:
            #检查p队列先进先出的任务是否过期，即使所有设备最大的cpu功率也无法满足它的完成需求
            while(not agent.p.empty() and agent.p.queue[0].ddl <= (t + math.ceil(agent.p.queue[0].size / (args.cpu_max * args.T_slot)) )):
                task = agent.p.get()
                agent.p_backlog -= task.size
                fail_task_num += 1
                env.max_task_num += 1
                #存在任务太多导致，在第一次排队等待卸载的过程中已无法满足完成需求的情况
                if agent.tasks[task.index][-1] is not None:
                #任务失败，返回延迟奖励值为0，训练上一个卸载设备
                    true_delay_reward = torch.tensor([0.0])
                    pre_agent = agents[agent.tasks[task.index][0]]
                    d_r, _ = pre_agent.pred_r_module(agent.tasks[task.index][1],
                                                     agent.tasks[task.index][2],
                                                     pre_agent.tasks[task.index])
                    d_r_mean = (d_r + agent.tasks[task.index][-1]) / 2
                    loss = pre_agent.mse(d_r_mean, true_delay_reward)

                    pre_agent.r_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pre_agent.pred_r_module.parameters(), max_norm=1.0)
                    pre_agent.r_optimizer.step()

                    pre_agent.tasks.pop(task.index)
                agent.tasks.pop(task.index)


            # 如果p队列为空，则不需要进行调度
            if agent.p.empty():
                mask[agent.index] = 1
                continue

            #将embedding输入actor网络得到卸载决策
            output, cpu_fre = agent.actor(observations[agent.index])
            select_index = torch.argmax(output, dim=-1)
            select_action = agent.index
            if select_index != 0:
                select_action = agent.neighbors[select_index - 1]

            #统计本轮所有设备动作以更新环境
            time_offloadings.append([agent.index, select_action, agent.p.queue[0]])
            cpu_fres[agent.index] = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
            action = torch.from_numpy(np.array([agent.index])).float()
            action = torch.cat([action, output, cpu_fres[agent.index]])
            actions[agent.index] = action


        #根据本轮的观测结果和动作分别生成邻域环境知识
        act_kno = [torch.zeros(args.act_cognition_size) for i in range(len(agents))]
        rebuild_a = [0 for i in range(len(agents))]
        for agent in agents:
            a_neighbors = []
            for i in agent.neighbors:
                a_neighbors.append(actions[i].detach())
            act_kno[agent.index], rebuild_a[agent.index] = agent.act_cognition(actions[agent.index], a_neighbors)

        nei_kno = [i.detach() for i in act_kno]
        for agent in agents:
            #生成动作共识存储到replaybuffer中用于学习
            a_rebuild_loss = agent.mse(rebuild_a[agent.index], actions[agent.index].detach())
            kl_loss = 0.0
            for nei in agent.neighbors:
                if torch.all(act_kno[nei] == 0):
                    continue
                # 为了防止inplace操作，将邻居的邻域知识detach下来，避免纳入计算图中
                kl_loss += agent.kl(act_kno[agent.index].log(), nei_kno[nei])
            act_loss = kl_loss

            agent.act_optimizer.zero_grad()
            act_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.act_cognition.parameters(), max_norm=1.0)
            agent.act_optimizer.step()


        #执行动作并返回相应的奖励
        # env_kno的共享导致了在训练共识模块时，预测奖励模块出现inplace操作.所以通过detach()方法从原本计算图中取下新的向量
        e_k = [i.detach() for i in env_kno]
        new_ob, new_env_kno, rewards, energy_costs = env.env_update(observations, time_offloadings, cpu_fres, t, mask, e_k, actions, epoch)

        #将轨迹存储到buffer中
        done = 1
        for agent in agents:
            if not agent.p.empty():
                done = 0
        for i in range(len(agents)):
            # 不detach()会在后面产生原地操作的错误
            agents[i].replay_buffer.store(env_kno[i].detach(), actions[i].detach(), act_kno[i].detach(), rewards[i], new_env_kno[i].detach(), observations[i], new_ob[i], done)
        total_rewards += rewards
        total_energy_cost += energy_costs
        for i in rewards:
            if i != 1.0:
                get_rewards += i
                rewards_num += 1
        for i in energy_costs:
            if i != 0.0:
                get_energy_cost += i
                energy_cost_num += 1
        if t < args.T:
            for agent in agents:
                p_backlog += agent.p_backlog
                q_backlog += agent.q_backlog
    end_time = time.time()

    logging.info('平均奖励：{}'.format((get_rewards / rewards_num).item()))
    logging.info('平均能耗：{}'.format(sum(total_energy_cost) / args.T / args.device_num))
    logging.info('平均p队列积压：{}'.format(p_backlog / len(agents)/ args.T))
    logging.info('平均q队列积压：{}'.format(q_backlog / len(agents)/ args.T))
    logging.info("episode---{}---运行结束，失败任务数：{}，失败任务率：{}，花费时间:{}".format(epoch, fail_task_num, fail_task_num/ env.max_task_num , end_time - start_time))

    return sum(total_energy_cost) / args.T / args.device_num, (get_rewards / rewards_num).item(), fail_task_num, fail_task_num/ env.max_task_num






def ddpg_run_eposide(args, logging, epoch, agents, env, cpu_fres):
    logging.info("开始第--{}--轮运行".format(epoch))
    start_time = time.time()
    fail_task_num = 0
    new_ob = ddpg_gen_observation(args, agents, epoch, 0)
    total_rewards = [0.0 for i in agents]
    total_energy_cost = [0.0 for i in agents]
    get_rewards = 0.0
    get_energy_cost = 0.0
    rewards_num = 0
    energy_cost_num = 0
    for t in range(args.T):
        time_offloadings = []
        observations = new_ob
        mask = [0 for i in range(len(agents))]
        actions = [torch.zeros(args.act_em_size) for i in range(len(agents))]

        for agent in agents:
            #检查p队列先进先出的任务是否过期，即使所有设备最大的cpu功率也无法满足它的完成需求
            while(not agent.p.empty() and agent.p.queue[0].ddl <= (t + math.ceil(agent.p.queue[0].size / (args.cpu_max * args.T_slot)) )):
                task = agent.p.get()
                agent.p_backlog -= task.size
                fail_task_num += 1
                env.max_task_num += 1

            # 如果p队列为空，则不需要进行调度
            if agent.p.empty():
                mask[agent.index] = 1
                continue

            #将embedding输入actor网络得到卸载决策
            output, cpu_fre = agent.actor(observations[agent.index])
            select_index = torch.argmax(output, dim=-1)
            select_action = agent.index
            if select_index != 0:
                select_action = agent.neighbors[select_index - 1]

            #统计本轮所有设备动作以更新环境
            time_offloadings.append([agent.index, select_action, agent.p.queue[0]])
            cpu_fres[agent.index] = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
            #生成action的embedding以用于生成动作邻域认知和预测奖励
            action = torch.from_numpy(np.array([agent.index])).float()
            action = torch.cat([action, output, cpu_fres[agent.index]])
            actions[agent.index] = action

        #执行动作并返回相应的奖励
        new_ob, rewards, energy_costs = env.ddpg_env_update(observations, time_offloadings, cpu_fres, t, mask, actions, epoch)

        #将轨迹存储到buffer中
        done = 1
        for agent in agents:
            if not agent.p.empty():
                done = 0
        actions =[i.detach() for i in actions]
        for i in range(len(agents)):
            # 不detach()会在后面产生原地操作的错误
            agents[i].replay_buffer.store(observations, actions, rewards[i], new_ob, done)
        total_rewards += rewards
        total_energy_cost += energy_costs
        for i in rewards:
            if i != 1.0:
                get_rewards += i
                rewards_num += 1
        for i in energy_costs:
            if i != 0.0:
                get_energy_cost += i
                energy_cost_num += 1
    end_time = time.time()
    if isinstance((get_rewards / rewards_num), float):
        mean_reward = get_rewards / rewards_num
    else:
        mean_reward = (get_rewards / rewards_num).item()
    logging.info('平均奖励：{}'.format(mean_reward))
    logging.info('平均能耗：{}'.format(sum(total_energy_cost) / args.T / args.device_num))
    logging.info("episode---{}---运行结束，失败任务数：{}，失败任务率：{}，花费时间:{}".format(epoch, fail_task_num,
                                                                        fail_task_num / env.max_task_num,
                                                                        end_time - start_time))

    return sum(total_energy_cost) / args.T / args.device_num, mean_reward, fail_task_num, fail_task_num / env.max_task_num





def dqn_run_eposide(args, logging, epoch, agents, env, cpu_fres, random_ep = 0.01):
    logging.info("开始第--{}--轮运行".format(epoch))
    start_time = time.time()
    fail_task_num = 0
    new_ob = ddpg_gen_observation(args, agents, epoch, 0)
    total_rewards = [0.0 for i in agents]
    total_energy_cost = [0.0 for i in agents]
    get_rewards = 0.0
    get_energy_cost = 0.0
    rewards_num = 0
    energy_cost_num = 0
    for t in range(args.T + 100):
        time_offloadings = []
        observations = new_ob
        mask = [0 for i in range(len(agents))]
        actions = [torch.zeros(args.act_em_size) for i in range(len(agents))]
        outputs = [ -1 for i in range(len(agents))]

        for agent in agents:
            #检查p队列先进先出的任务是否过期，即使所有设备最大的cpu功率也无法满足它的完成需求
            while(not agent.p.empty() and agent.p.queue[0].ddl <= (t + math.ceil(agent.p.queue[0].size / (args.cpu_max * args.T_slot)) )):
                task = agent.p.get()
                agent.p_backlog -= task.size
                fail_task_num += 1
                env.max_task_num += 1

            # 如果p队列为空，则不需要进行调度
            if agent.p.empty():
                mask[agent.index] = 1
                continue

            #将embedding输入actor网络得到卸载决策
            output = agent.DQN(observations)
            if np.random.rand() < random_ep:
                select_index = np.random.randint(0, args.neighbors_num + 1)
                if select_index == 0:
                    select_action = agent.index
                else:
                    select_action = agent.neighbors[select_index - 1]
            else:
                select_index = torch.argmax(output, dim=-1)
                select_action = agent.index
                if select_index != 0:
                    select_action = agent.neighbors[select_index - 1]
            outputs[agent.index] = select_index

            #统计本轮所有设备动作以更新环境
            time_offloadings.append([agent.index, select_action, agent.p.queue[0]])
            action = torch.from_numpy(np.array([agent.index])).float()
            action = torch.cat([action, output])
            actions[agent.index] = action

        #执行动作并返回相应的奖励
        new_ob, rewards, energy_costs = env.dqn_env_update(observations, time_offloadings, t, mask, actions, epoch)

        #将轨迹存储到buffer中
        done = 1
        for agent in agents:
            if not agent.p.empty():
                done = 0
        for i in range(len(agents)):
            # 不detach()会在后面产生原地操作的错误
            agents[i].replay_buffer.store(observations, outputs[i], rewards[i], new_ob, done)
        total_rewards += rewards
        total_energy_cost += energy_costs
        for i in rewards:
            if i != 1.0:
                get_rewards += i
                rewards_num += 1
        for i in energy_costs:
            if i != 0.0:
                get_energy_cost += i
                energy_cost_num += 1
    end_time = time.time()
    logging.info('平均奖励：{}'.format((get_rewards / rewards_num).item()))
    logging.info('平均能耗：{}'.format(sum(total_energy_cost) / args.T / args.device_num))
    logging.info("episode---{}---运行结束，失败任务数：{}，失败任务率：{}，花费时间:{}".format(epoch, fail_task_num,
                                                                        fail_task_num / env.max_task_num ,
                                                                        end_time - start_time))

    return sum(total_energy_cost) / args.T / args.device_num, (get_rewards / rewards_num).item(), fail_task_num, fail_task_num / env.max_task_num


#
# if __name__ == '__main__':
#     gen_dataset(10, 100)