import random
import queue
import  numpy as np
import networkx as nx
import torch
import torch.nn as nn




def get_observation(agents, agent):
    observation = {
        "channel_gain": [0.0 for i in range(len(agent.neighbors))],
        "self_cpu": 0,
        "p_backlog": [0 for i in range(len(agent.neighbors) + 1)],
        "q_backlog": [0 for i in range(len(agent.neighbors) + 1)]
    }
    observation["channel_gain"] = agent.ch_gain
    observation["self_cpu"] = agent.cpu
    observation["q_backlog"][0] = agent.q_backlog
    for neighbor in agent.neighbors:
        observation["p_backlog"][agent.neighbors.index(neighbor) + 1] = agents[neighbor].p_backlog
        observation["q_backlog"][agent.neighbors.index(neighbor) + 1] = agents[neighbor].q_backlog

    return observation

def gen_observation(args, agents, epo, t):
    new_observations = [[torch.zeros(args.neighbors_num * 3 + 3), torch.zeros(args.task_em_size)] for i in range(args.device_num)]
    update_neighbors(args, agents)
    for agent in agents:
        # 观察由上一个时隙卸载后的邻居队列情况和自身cpu频率，这一个时隙的信道增益和自身队列情况组成
        agent.channel_update()
        # print(agent.ch_gain)
        observation = get_observation(agents, agent)
        # 任务随机抵达后得到新的p队列积压，第一个位置是agent本身的积压;系统进入预结束状态,任务不再抵达
        if t < args.T:
            agent.task_arrive(epo, t)
        observation["p_backlog"][0] = agent.p_backlog

        # 生成embedding
        e_embed = []
        for key, value in observation.items():
            if not isinstance(value, float):
                # 展平列表并添加到embedding
                e_embed.extend(value)
            else:
                e_embed.append(value)
        e_embed = torch.from_numpy(np.array(e_embed)).float()
        if agent.p.empty():
            k_embed = torch.zeros(args.task_em_size)
        else:
            k_embed = agent.p.queue[0].toembed()
        new_observations[agent.index] = [e_embed, k_embed]
    env_kno = [torch.zeros(args.env_cognition_size) for i in range(len(agents))]
    rebuild_e = [0 for i in range(len(agents))]
    for agent in agents:
        #生成环境邻域认知
        env_neighbors = []
        a_neighbors = []
        for i in agent.neighbors:
            env_neighbors.append(torch.cat([new_observations[i][0], new_observations[i][1]]))
        o = torch.cat([new_observations[agent.index][0], new_observations[agent.index][1]])
        env_kno[agent.index], rebuild_e[agent.index] = agent.env_cognition(o, env_neighbors)

    #直接将整个邻域知识detach，以避免访问已释放的env_kno中的tensor，造成需要维持计算图
    nei_kno = [i.detach() for i in env_kno]
    # 训练共识模块
    for agent in agents:
        o = torch.cat([new_observations[agent.index][0], new_observations[agent.index][1]])
        e_rebuild_loss = agent.mse(rebuild_e[agent.index], o)
        kl_loss = 0.0
        for nei in agent.neighbors:
            if torch.all(env_kno[nei] == 0):
                continue
            # 为了防止inplace操作，将邻居的邻域知识detach下来，避免纳入计算图中
            kl_loss += agent.kl(env_kno[agent.index].log(), nei_kno[nei])
        env_loss = kl_loss

        agent.env_optimizer.zero_grad()
        env_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.env_cognition.parameters(), max_norm=1.0)
        agent.env_optimizer.step()

    return new_observations, env_kno



def ddpg_gen_observation(args, agents, epo, t):
    new_observations = [[torch.zeros(args.neighbors_num * 3 + 3), torch.zeros(args.task_em_size)] for i in range(args.device_num)]
    update_neighbors(args, agents)
    for agent in agents:
        # 观察由上一个时隙卸载后的邻居队列情况和自身cpu频率，这一个时隙的信道增益和自身队列情况组成
        agent.channel_update()
        observation = get_observation(agents, agent)
        # 任务随机抵达后得到新的p队列积压，第一个位置是agent本身的积压;系统进入预结束状态,任务不再抵达
        if t < args.T:
            agent.task_arrive(epo, t)
        observation["p_backlog"][0] = agent.p_backlog

        # 生成embedding
        e_embed = []
        for key, value in observation.items():
            if not isinstance(value, float):
                # 展平列表并添加到embedding
                e_embed.extend(value)
            else:
                e_embed.append(value)
        e_embed = torch.from_numpy(np.array(e_embed)).float()
        if agent.p.empty():
            k_embed = torch.zeros(args.task_em_size)
        else:
            k_embed = agent.p.queue[0].toembed()
        new_observations[agent.index] = [e_embed, k_embed]

    return new_observations


def update_neighbors(args, agents):
    for agent in agents:
        agent.x = random.randint(0, 100)
        agent.y = random.randint(0, 100)
    #去重
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            while agents[i].x == agents[j].x or agents[i].y == agents[j].y:
                agents[j].x = random.randint(0, 100)
                agents[j].y = random.randint(0, 100)
    for agent in agents:
        dis = [0 for i in agents]
        for other_agent in agents:
                dis[other_agent.index] = ((agent.x - other_agent.x) **2 + (agent.y - other_agent.y) **2) ** (1 / 2)

        indices = np.argsort(dis)
        agent.neighbors = list(indices[ - args.neighbors_num:])
        agent.neighbors_dis = [dis[i] for i in agent.neighbors]

class task:
    _global_index = 0  # 用于跟踪全局索引

    def __init__(self, args, t, global_index):
        self.index = global_index
        self.size = random.randint(1, args.task_size_max)
        self.ddl = t + random.randint(args.cost_time_max - 10, args.cost_time_max)
        self.hop = 0

    def show(self):
        print("任务索引：" + str(self.index) + "任务大小：" + str(self.size) + "任务ddl：" + str(self.ddl) + "任务跳数：" + str(self.hop))

    def toembed(self):
        return torch.from_numpy(np.array([self.index, self.size, self.ddl, self.hop])).float()


class agent:
    def __init__(self, args, index):
        self.args = args
        self.index = index
        self.x = 0
        self.y = 0
        self.p = queue.Queue()
        self.p_backlog = 0
        self.q = queue.Queue()
        self.q_backlog = 0
        self.cpu = 0.0
        self.cpu_max = random.randint(1, args.cpu_max)
        self.cpu_min = 0.0
        self.neighbors = []
        self.neighbors_dis = []
        self.ch_gain = []
        self.e = []
        self.bandwidth = random.randint(5, 10) / 10 #对标处理频率，单位GHZ
        self.transpower = random.randint(50, 100) #传输功耗，对标最大处理功耗1000
        self.tasks = {}
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss()
        self.bce = nn.BCELoss()


        #生成样本时间数的高斯白噪声的值
        noise_var = 0.01
        epsilon = 1e-9  # 非零小值
        while True:
            white_noise = np.random.normal(0, np.sqrt(noise_var), args.T)
            white_noise = np.where(white_noise <= 0, epsilon, white_noise)  # 确保没有零或负值，负值也会导致reward出现nan,对数函数 np.log2 不接受负值或零作为输入
            if not np.any(np.isnan(white_noise)):
                break
        self.white_noise = white_noise


    def task_arrive(self, epi, t):
        #任务服从泊松分布到达
        with open('dataset/episodes_{}_poisson-rate_{}_device-num_{}_task-delay_{}/episode_{}/train_eposide_{}.txt'.format(self.args.epochs, self.args.poisson_rate, self.args.device_num, self.args.cost_time_max, epi, self.index), 'r') as file:
            # print(epi, t, self.index)
            for line in file.readlines():
                line = line.strip()
                th = line.split('\t')
                if len(th) == 1 and t >= 0:
                    t -= 1
                    continue
                if len(th) == 4 and t == -1:
                    # print(line)
                    ts = task(self.args, t, 0)
                    ts.index, ts.size, ts.ddl, ts.hop = int(th[0]), int(th[1]), int(th[2]), int(th[3])
                    self.p.put(ts)
                    self.p_backlog += ts.size
                    #用于更新任务状态以更新延迟奖励预测模块，第一个字段是任务上一个卸载的客户端索引，第二个字段是上一次的观测，第三个字段是上一次采取的操作，第四个字段是上一次卸载时刻卸载目的地的邻域认知embedding，第五个字段是上次的预测奖励值
                    self.tasks.update({ts.index: [-1, [torch.zeros(self.args.neighbors_num * 3 + 3), torch.zeros(self.args.task_em_size)], torch.zeros(self.args.task_em_size + 3), torch.zeros(self.args.env_cognition_size), None ]})
                if len(th) == 1 and t == -1:
                    break

    def channel_update(self, gamma = 0.01, seta = 4):
        self.ch_gain = [gamma * (i) ** (- seta) for i in self.neighbors_dis]
        # print(self.ch_gain)

class environment:
    def __init__(self, args, agents):
        self.agents = agents
        self.args = args
        self.reward_max = 10
        self.q_backlog = 0
        self.p_backlog = 0
        self.SNR_min = 0
        self.SNR_max = 0
        self.max_task_num = 0


    def env_update(self, observations, time_offloadings, cpu_fres, t, mask, env_kno, actions, epoch):
        delay_rewards = [0.0 for i in range(len(self.agents))]
        #执行卸载策略
        for action in time_offloadings:
            task = self.agents[action[0]].p.queue[0]

            # 检查向q队列卸载是否能够满足保证任务完成，并计算保证任务完成的最小cpu频率
            if action[0] == action[1]:
                cpu_min = (self.agents[action[0]].q_backlog + task.size)  / ((task.ddl - t) * self.args.T_slot)
                #如果最晚完成任务所需的功率依然大于设备的最大功率，那么任务无法保证在q队列中完成，本次卸载无效（应该需要做惩罚？
                if cpu_min > self.agents[action[0]].cpu_max:
                    continue
                #检查对于新进任务的最小需要满足功率是否是目前在q队列中的任务所需的最小满足功率
                ac_t = t
                flag = True
                if cpu_min > self.agents[action[0]].cpu_min:
                    self.agents[action[0]].cpu_min = cpu_min
                for task in self.agents[action[0]].q.queue:
                    ac_t += task.size / (cpu_min * self.args.T_slot)
                    if ac_t > task.ddl:
                        flag = False
                if flag:
                    self.agents[action[0]].cpu_min = cpu_min
                task = self.agents[action[0]].p.get()
                self.agents[action[0]].p_backlog -= task.size
                self.agents[action[0]].q.put(task)
                self.agents[action[0]].q_backlog += task.size

                delay_reward, pre_reward = self.agents[action[0]].pred_r_module(observations[action[0]], actions[action[0]].detach(), self.agents[action[0]].tasks[task.index])
                delay_rewards[action[0]] = delay_reward
                #向q队列进行卸载，此处应该返回任务成功的延迟奖励1，并且先唤醒训练上一个卸载的设备的奖励预测模块，再训练当前设备的模块
                true_delay_reward = torch.tensor([1.0])

                if self.agents[action[0]].tasks[task.index][-1] is not None:
                    pre_agent = self.agents[self.agents[action[0]].tasks[task.index][0]]
                    # print(pre_agent.index, action[0], task.index)
                    d_r, _ = pre_agent.pred_r_module(self.agents[action[0]].tasks[task.index][1],
                                                     self.agents[action[0]].tasks[task.index][2],
                                                     pre_agent.tasks[task.index])
                    d_r_mean = (d_r + self.agents[action[0]].tasks[task.index][-1]) / 2
                    loss = pre_agent.mse(d_r_mean, pre_reward.detach())
                    pre_agent.r_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pre_agent.pred_r_module.parameters(), max_norm=1.0)
                    pre_agent.r_optimizer.step()
                    pre_agent.tasks.pop(task.index)

                #训练本次卸载最终设备
                loss = self.agents[action[0]].mse(delay_reward, true_delay_reward)
                self.agents[action[0]].r_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[action[0]].pred_r_module.parameters(), max_norm=1.0)
                self.agents[action[0]].r_optimizer.step()
                self.agents[action[0]].tasks.pop(task.index)
            else:
                task = self.agents[action[0]].p.get()
                self.agents[action[0]].p_backlog -=task.size
                task.hop += 1
                self.agents[action[1]].p.put(task)
                self.agents[action[1]].p_backlog += task.size
                delay_reward, pre_reward = self.agents[action[0]].pred_r_module(observations[action[0]], actions[action[0]], self.agents[action[0]].tasks[task.index])
                delay_rewards[action[0]] = delay_reward
                if self.agents[action[0]].tasks[task.index][-1] is not None:
                    pre_agent = self.agents[self.agents[action[0]].tasks[task.index][0]]
                    d_r, _ = pre_agent.pred_r_module(self.agents[action[0]].tasks[task.index][1], self.agents[action[0]].tasks[task.index][2], pre_agent.tasks[task.index])
                    d_r_mean = (d_r + self.agents[action[0]].tasks[task.index][-1]) / 2
                    loss = pre_agent.mse(d_r_mean, pre_reward)
                    pre_agent.r_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pre_agent.pred_r_module.parameters(), max_norm=1.0)
                    pre_agent.r_optimizer.step()
                    pre_agent.tasks.pop(task.index)
                # 更新卸载目的地设备的tasks，以便再次卸载任务时能够获得相关信息。将设备action[1]的邻域认知存储到tasks中，以便当排队到处理该任务时进行预测
                # 第一个字段是任务上一个卸载的客户端索引，第二个字段是上一次的观测，第三个字段是上一次采取的操作，第四个字段是上一次卸载时刻卸载目的地的邻域认知embedding，第五个字段是上次的预测奖励值
                #添加detach防止训练reward prediction模块的时候出现需要保持计算图的需求
                self.agents[action[1]].tasks.update({task.index: [action[0], observations[action[0]], actions[action[0]].detach(), env_kno[action[1]].detach(),delay_reward.detach()]})


        #设置这一时隙的cpu频率并处理每个agent的q队列
        for i in range(len(cpu_fres)):
            if self.agents[i].q_backlog == 0:
                self.agents[i].cpu = 0.0
            else:
                self.agents[i].cpu = cpu_fres[i].item()

        for agent in self.agents:
            pro_task_size = agent.cpu * self.args.T_slot
            while pro_task_size != 0 and not agent.q.empty():
                if agent.q.queue[0].size > pro_task_size:
                    agent.q.queue[0].size -= pro_task_size
                    agent.q_backlog -= pro_task_size
                    pro_task_size = 0
                else:
                    pro_task_size -= agent.q.queue[0].size
                    agent.q_backlog -= agent.q.queue[0].size
                    if self.max_task_num < agent.q.queue[0].index:
                        self.max_task_num = agent.q.queue[0].index
                    agent.q.get()
                    self.max_task_num += 1


        #返回动作的即时奖励
        rewards = [1.0 for i in self.agents]
        energy_cost = [0.0 for i in self.agents]
        for agent in self.agents:
            energy_cost[agent.index] += self.args.k * (self.agents[agent.index].cpu) ** 3 * self.args.T_slot
        for action in time_offloadings:
            cpu_term = self.args.V * self.args.k * (self.agents[action[0]].cpu )**3 * self.args.T_slot - self.agents[action[0]].q_backlog * self.agents[action[0]].cpu
            if action[0] == action[1]:
                rewards[action[0]] =  - self.agents[action[0]].q_backlog  * action[2].size  - cpu_term
            else:
                transfer_term = 40 * self.args.V * self.agents[action[0]].transpower * action[2].size / ( self.agents[action[0]].bandwidth * np.log2(1 + self.agents[action[0]].transpower * self.agents[action[0]].ch_gain[self.agents[action[0]].neighbors.index(action[1])] / 0.00000000001 ))
                energy_cost[action[0]] += self.agents[action[0]].transpower * action[2].size / (self.agents[action[0]].bandwidth * np.log2(1 + self.agents[action[0]].transpower * self.agents[action[0]].ch_gain[self.agents[action[0]].neighbors.index(action[1])] / 0.00000000001 ))
                rewards[action[0]] =  - self.agents[action[1]].p_backlog   * action[2].size  - cpu_term - transfer_term

        #生成新的观测
        new_observations = [[torch.zeros(self.args.neighbors_num * 3 + 3), torch.zeros(self.args.task_em_size)] for i in
                            range(self.args.device_num)]
        e_k = [torch.zeros(self.args.env_cognition_size) for i in range(self.args.device_num)]
        if t != self.args.T - 1:
            new_observations, e_k = gen_observation(self.args, self.agents, epoch, t + 1)

        return new_observations, e_k, [ i * (10 + j) for i, j in zip(rewards, delay_rewards)], energy_cost






    def ddpg_env_update(self, observations, time_offloadings, cpu_fres, t, mask, actions, epoch):
        # 执行卸载策略
        for action in time_offloadings:
            task = self.agents[action[0]].p.queue[0]

            # 检查向q队列卸载是否能够满足保证任务完成，并计算保证任务完成的最小cpu频率
            if action[0] == action[1]:
                cpu_min = (self.agents[action[0]].q_backlog + task.size) / ((task.ddl - t) * self.args.T_slot)
                # 如果最晚完成任务所需的功率依然大于设备的最大功率，那么任务无法保证在q队列中完成，本次卸载无效（应该需要做惩罚？
                if cpu_min > self.agents[action[0]].cpu_max:
                    continue
                # 检查对于新进任务的最小需要满足功率是否是目前在q队列中的任务所需的最小满足功率
                ac_t = t
                flag = True
                if cpu_min > self.agents[action[0]].cpu_min:
                    self.agents[action[0]].cpu_min = cpu_min
                for task in self.agents[action[0]].q.queue:
                    ac_t += task.size / (cpu_min * self.args.T_slot)
                    if ac_t > task.ddl:
                        flag = False
                if flag:
                    self.agents[action[0]].cpu_min = cpu_min
                task = self.agents[action[0]].p.get()
                self.agents[action[0]].p_backlog -= task.size
                self.agents[action[0]].q.put(task)
                self.agents[action[0]].q_backlog += task.size

            else:
                task = self.agents[action[0]].p.get()
                self.agents[action[0]].p_backlog -= task.size
                task.hop += 1
                self.agents[action[1]].p.put(task)
                self.agents[action[1]].p_backlog += task.size

        # 设置这一时隙的cpu频率并处理每个agent的q队列
        for i in range(len(cpu_fres)):
            if self.agents[i].q_backlog == 0:
                self.agents[i].cpu = 0.0
            else:
                self.agents[i].cpu = cpu_fres[i].item()

        for agent in self.agents:
            pro_task_size = agent.cpu * self.args.T_slot
            while pro_task_size != 0 and not agent.q.empty():
                if agent.q.queue[0].size > pro_task_size:
                    agent.q.queue[0].size -= pro_task_size
                    agent.q_backlog -= pro_task_size
                    pro_task_size = 0
                else:
                    pro_task_size -= agent.q.queue[0].size
                    agent.q_backlog -= agent.q.queue[0].size
                    agent.q.get()
                    self.max_task_num += 1

        # 返回动作的即时奖励
        rewards = [1.0 for i in self.agents]
        energy_cost = [0.0 for i in self.agents]
        for agent in self.agents:
            energy_cost[agent.index] += self.args.k * (self.agents[agent.index].cpu) ** 3 * self.args.T_slot
        for action in time_offloadings:
            cpu_term = self.args.V * self.args.k * (self.agents[action[0]].cpu) ** 3 * self.args.T_slot - self.agents[
                action[0]].q_backlog * self.agents[action[0]].cpu
            if action[0] == action[1]:
                rewards[action[0]] = - self.agents[action[0]].q_backlog * action[2].size - cpu_term
            else:
                transfer_term = 40 * self.args.V * self.agents[action[0]].transpower * action[2].size / (
                            self.agents[action[0]].bandwidth * np.log2(
                        1 + self.agents[action[0]].transpower * self.agents[action[0]].ch_gain[
                            self.agents[action[0]].neighbors.index(action[1])] / 0.00000000001))
                energy_cost[action[0]] += self.agents[action[0]].transpower * action[2].size / (
                            self.agents[action[0]].bandwidth * np.log2(
                        1 + self.agents[action[0]].transpower * self.agents[action[0]].ch_gain[
                            self.agents[action[0]].neighbors.index(action[1])] / 0.00000000001))
                rewards[action[0]] = - self.agents[action[1]].p_backlog * action[2].size - cpu_term - transfer_term

        # 生成新的观测
        new_observations = [[torch.zeros(self.args.neighbors_num * 3 + 3), torch.zeros(self.args.task_em_size)] for i in
                            range(self.args.device_num)]
        if t != self.args.T - 1:
            new_observations = ddpg_gen_observation(self.args, self.agents, epoch, t + 1)

        return new_observations, [ i  for i in rewards], energy_cost




    def dqn_env_update(self, observations, time_offloadings, t, mask, actions, epoch):
        # 执行卸载策略
        for action in time_offloadings:
            task = self.agents[action[0]].p.queue[0]

            # 检查向q队列卸载是否能够满足保证任务完成，并计算保证任务完成的最小cpu频率
            if action[0] == action[1]:
                cpu_min = (self.agents[action[0]].q_backlog + task.size) / ((task.ddl - t) * self.args.T_slot)
                # 如果最晚完成任务所需的功率依然大于设备的最大功率，那么任务无法保证在q队列中完成，本次卸载无效（应该需要做惩罚？
                if cpu_min > self.agents[action[0]].cpu_max:
                    continue
                # 检查对于新进任务的最小需要满足功率是否是目前在q队列中的任务所需的最小满足功率
                ac_t = t
                flag = True
                if cpu_min > self.agents[action[0]].cpu_min:
                    self.agents[action[0]].cpu_min = cpu_min
                for task in self.agents[action[0]].q.queue:
                    ac_t += task.size / (cpu_min * self.args.T_slot)
                    if ac_t > task.ddl:
                        flag = False
                if flag:
                    self.agents[action[0]].cpu_min = cpu_min
                task = self.agents[action[0]].p.get()
                self.agents[action[0]].p_backlog -= task.size
                self.agents[action[0]].q.put(task)
                self.agents[action[0]].q_backlog += task.size

            else:
                task = self.agents[action[0]].p.get()
                self.agents[action[0]].p_backlog -= task.size
                task.hop += 1
                self.agents[action[1]].p.put(task)
                self.agents[action[1]].p_backlog += task.size

        # 设置这一时隙的cpu频率并处理每个agent的q队列
        for i in range(len(self.agents)):
            if self.agents[i].q_backlog == 0:
                self.agents[i].cpu = 0.0
            else:
                self.agents[i].cpu = self.agents[i].cpu_min

        for agent in self.agents:
            pro_task_size = agent.cpu * self.args.T_slot
            while pro_task_size != 0 and not agent.q.empty():
                if agent.q.queue[0].size > pro_task_size:
                    agent.q.queue[0].size -= pro_task_size
                    agent.q_backlog -= pro_task_size
                    pro_task_size = 0
                else:
                    pro_task_size -= agent.q.queue[0].size
                    agent.q_backlog -= agent.q.queue[0].size
                    agent.q.get()
                    self.max_task_num += 1

        # 返回动作的即时奖励
        rewards = [1.0 for i in self.agents]
        energy_cost = [0.0 for i in self.agents]
        for agent in self.agents:
            energy_cost[agent.index] += self.args.k * (self.agents[agent.index].cpu) ** 3 * self.args.T_slot
        for action in time_offloadings:
            cpu_term = self.args.V * self.args.k * (self.agents[action[0]].cpu) ** 3 * self.args.T_slot - self.agents[
                action[0]].q_backlog * self.agents[action[0]].cpu
            if action[0] == action[1]:
                rewards[action[0]] = - self.agents[action[0]].q_backlog * action[2].size - cpu_term
            else:
                transfer_term = 40 * self.args.V * self.agents[action[0]].transpower * action[2].size / (
                            self.agents[action[0]].bandwidth * np.log2(
                        1 + self.agents[action[0]].transpower * self.agents[action[0]].ch_gain[
                            self.agents[action[0]].neighbors.index(action[1])] / 0.00000000001))
                energy_cost[action[0]] += self.agents[action[0]].transpower * action[2].size / (
                            self.agents[action[0]].bandwidth * np.log2(
                        1 + self.agents[action[0]].transpower * self.agents[action[0]].ch_gain[
                            self.agents[action[0]].neighbors.index(action[1])] / 0.00000000001))
                rewards[action[0]] = - self.agents[action[1]].p_backlog * action[2].size - cpu_term - transfer_term

        # 生成新的观测
        new_observations = [[torch.zeros(self.args.neighbors_num * 3 + 3), torch.zeros(self.args.task_em_size)] for i in
                            range(self.args.device_num)]
        if t != self.args.T - 1:
            new_observations = ddpg_gen_observation(self.args, self.agents, epoch, t + 1)

        return new_observations, [ i  for i in rewards], energy_cost