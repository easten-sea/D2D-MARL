import torch
from sysmodel import task, environment,get_observation, agent, update_neighbors
from MARL import xxx_agent, setup_seed
import numpy as np
import argparse
import random
import math
from run import run_eposide,gen_dataset, ddpg_run_eposide, dqn_run_eposide
from train import train_epoch, ddpg_train_epoch, dqn_train_epoch, coma_train_epoch
from local import local_run
from closest import closest_run
from DDPG import DDPG_agent
from DQN import  DQN_agent
import os
import time
import logging
import sys




cuda_enable = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='agrs for MARL')
    ap.add_argument('--alth', type=str, default='our', help='运行的算法')
    ap.add_argument('--epochs', type=int, default=2000, help='轨迹样本数')
    ap.add_argument('--device_num', type=int, default=10, help='设备数量')
    ap.add_argument('--neighbors_num', type=int, default=5, help='每个设备的邻居数量')
    ap.add_argument('--seed_num', type=int, default=10, help='随机数种子')
    ap.add_argument('--T', type=int, default=200, help='总时隙数')
    ap.add_argument('--task_em_size', type=int, default=4, help='任务embedding的大小')
    ap.add_argument('--act_em_size', type=int, default=8, help='动作embedding的大小')
    ap.add_argument('--env_cognition_size', type=int, default=128, help='环境邻域认知embedding的大小')
    ap.add_argument('--act_cognition_size', type=int, default=32, help='动作邻域认知embedding的大小')
    ap.add_argument('--hop_max', type=int, default=5, help='卸载跳数的最大值')
    ap.add_argument('--poisson_rate', type=float, default=0.3, help='任务生成的泊松值')


    ap.add_argument('--task_size_max', type=int, default=10, help='任务的最大值,单位为GB')
    ap.add_argument('--cpu_max', type=int, default=4, help='设备cpu频率最大值，每个设备再进一步设置自己的最大值以表示异构设备，单位为GHz')
    ap.add_argument('--k', type=float, default=62.5, help='能耗的计算能量效率参数')
    ap.add_argument('--T_slot', type=int, default=5, help='每个时隙的长度')
    ap.add_argument('--cost_time_max', type=int, default=10, help='任务最大容忍时长')
    ap.add_argument('--mean_gain', type=float, default=0.5, help='信道增益的平均增益')


    #用于计算奖励的参数
    ap.add_argument('--V', type=float, default=50, help='李雅普诺夫优化的漂移函数的惩罚项权重')
    ap.add_argument('--gamma', type=float, default=0.5, help='未来奖励的折扣率')

    #用于学习的参数
    ap.add_argument('--learning_rate', type=float, default=0.001, help='学习率，取值（0，1）')
    ap.add_argument('--batch_size', type=float, default=64, help='学习的batchsize')
    ap.add_argument('--tau', type=float, default=0.001, help='目标网络的软更新参数')
    ap.add_argument('--target_update_freq', type=int, default=5, help='目标网络的更新频率')

    args = ap.parse_args()

    # 配置日志输出
    log_file = "log/{}_episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}_output_log.txt".format(args.alth, args.epochs, args.poisson_rate, args.device_num, args.cost_time_max)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到控制台
            logging.FileHandler(log_file)  # 输出到文件
        ]
    )

    setup_seed(args.seed_num)

    if not os.path.exists('dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max)):
        os.mkdir('dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}'.format(args.epochs, args.poisson_rate, args.device_num, args.cost_time_max))
        gen_dataset(args, args.epochs)

    agents = []

    cpu_fres = [np.zeros(1) for i in range(args.device_num)]


    if args.alth == 'local':
        for i in range(args.device_num):
            agents.append(agent(args, i))

        total_energy_cost = [0.0 for i in agents]
        fail_task_num = 0
        fail_rate = 0.0
        te = 0.0
        for epoch in range(args.epochs):
            logging.info("开始运行episode--{}".format(epoch))
            start_time = time.time()

            energy_cost, fail_task, total_task_num = local_run(args, epoch, agents)
            te += energy_cost
            fail_task_num += fail_task
            fail_rate += fail_task / total_task_num
            end_time = time.time()
            logging.info('平均能耗：{}'.format(energy_cost))
            logging.info("episode---{}---运行结束，失败任务数：{}，失败任务率：{}，花费时间:{}".format(epoch, fail_task , fail_task / total_task_num, end_time - start_time))
        logging.info("平均总能耗： {} | 平均失败任务数： {} | 平均失败任务率：{}".format(te / args.epochs, fail_task_num / args.epochs, fail_rate / args.epochs))
    elif args.alth == 'closest':
        for i in range(args.device_num):
            agents.append(agent(args, i))

        total_energy_cost = [0.0 for i in agents]
        fail_task_num = 0
        fail_rate = 0.0
        te = 0.0
        for epoch in range(args.epochs):
            logging.info("开始运行episode--{}".format(epoch))
            start_time = time.time()

            energy_cost, fail_task, total_task_num = closest_run(args, epoch, agents)
            te += energy_cost

            fail_task_num += fail_task
            fail_rate += fail_task / total_task_num

            end_time = time.time()
            logging.info('平均能耗：{}'.format(energy_cost))
            logging.info("episode---{}---运行结束，失败任务数：{}，失败任务率：{}，花费时间:{}".format(epoch, fail_task , fail_task / total_task_num, end_time - start_time))
        logging.info("平均总能耗： {} | 平均失败任务数： {} | 平均失败任务率：{}".format(te / args.epochs, fail_task_num / args.epochs, fail_rate / args.epochs))
    elif args.alth == 'DDPG' or args.alth =='COMA':
        # print('here')
        for i in range(args.device_num):
            agents.append(DDPG_agent(args, i))
        env = environment(args, agents)
        fail_task = 0
        fail_rate = 0.0
        tr = 0.0
        te = 0.0
        for epoch in range(args.epochs):
            env.max_task_num = 0
            total_energy_cost, total_rewards, fail_task_num, fail = ddpg_run_eposide(args, logging, epoch, agents, env, cpu_fres)
            print(env.max_task_num)
            tr += total_rewards
            te += total_energy_cost
            fail_task += fail_task_num
            fail_rate += fail
            if args.alth == 'DDPG':
                ddpg_train_epoch(args, logging, epoch, agents, env)
            if args.alth =='COMA':
                # print(1111111111)
                coma_train_epoch(args, logging, epoch, agents, env)
        logging.info("平均总能耗： {} | 平均总奖励： {} | 平均失败任务数： {} | 平均任务失败率: {}".format(te / args.epochs, tr / args.epochs,
                                                                                fail_task / args.epochs,
                                                                                fail_rate / args.epochs))

    elif args.alth == 'DQN' or args.alth == 'D3QN':
        # print('here')
        for i in range(args.device_num):
            agents.append(DQN_agent(args, i))
        env = environment(args, agents)
        fail_task = 0
        fail_rate = 0.0
        tr = 0.0
        te = 0.0

        for epoch in range(args.epochs):
            env.max_task_num = 0
            total_energy_cost, total_rewards, fail_task_num, fail = dqn_run_eposide(args, logging, epoch, agents, env, cpu_fres)
            print(env.max_task_num)
            tr += total_rewards
            te += total_energy_cost
            fail_task += fail_task_num
            fail_rate += fail
            dqn_train_epoch(args, logging, epoch, agents, env)
        logging.info("平均总能耗： {} | 平均总奖励： {} | 平均失败任务数： {} | 平均任务失败率: {}".format(te / args.epochs, tr / args.epochs,
                                                                                fail_task / args.epochs,
                                                                                fail_rate / args.epochs))

    else:
        for i in range(args.device_num):
            agents.append(xxx_agent(args, i))
        env = environment(args, agents)
        fail_task = 0
        fail_rate = 0.0
        tr = 0.0
        te = 0.0

        for epoch in range(args.epochs):
            env.max_task_num = 0
            total_energy_cost, total_rewards, fail_task_num, fail = run_eposide(args, logging, epoch, agents, env, cpu_fres)
            print(env.max_task_num)
            tr += total_rewards
            te += total_energy_cost
            fail_task += fail_task_num
            fail_rate += fail

            train_epoch(args, logging, epoch, agents, env)
        logging.info("平均总能耗： {} | 平均总奖励： {} | 平均失败任务数： {} | 平均任务失败率: {}".format( te / args.epochs , tr / args.epochs , fail_task / args.epochs, fail_rate / args.epochs))



















