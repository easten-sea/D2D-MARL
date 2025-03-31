
import numpy as np
from sysmodel import update_neighbors



def closest_run(args, epoch, agents):

    fail_task_num = 0
    total_task_num = 0
    total_energy_cost = [0.0 for i in agents]
    #多的时间用于处理任务
    for t in range(args.T + 100):
        update_neighbors(args, agents)
        for agent in agents:
            agent.channel_update()
            if t < args.T:
                agent.task_arrive(epoch, t)
            #代理的p队列任务必然卸载到q队列，只需检查最大cpu频率能否完成以及设计新的cpu频率
            if not agent.p.empty():
                task = agent.p.get()
                total_task_num += 1
                agent.p_backlog -= task.size
                #卸载的时候已经超时了（排队延迟
                if task.ddl <= t:
                    fail_task_num += 1
                    continue
                cpu_min = (agent.q_backlog + task.size) / ((task.ddl - t) * args.T_slot)
                # 如果最晚完成任务所需的功率依然大于设备的最大功率，那么任务无法保证在q队列中完成，向最近的邻居卸载
                if cpu_min > agent.cpu_max:
                    off_agent_index = agent.neighbors[agent.neighbors_dis.index(min(agent.neighbors_dis))]
                    agents[off_agent_index].p.put(task)
                    agents[off_agent_index].p_backlog += task.size
                    total_energy_cost[agent.index] += agents[agent.index].transpower * task.size / (agents[agent.index].bandwidth * np.log2(1 + agents[agent.index].transpower * agents[agent.index].ch_gain[agents[agent.index].neighbors.index(off_agent_index)] / 0.00000000001 ))
                    continue
                ac_t = t
                flag = True
                if cpu_min > agent.cpu_min:
                    agent.cpu_min = cpu_min
                for tk in agent.q.queue:
                    ac_t += tk.size / (cpu_min * args.T_slot)
                    if ac_t > tk.ddl:
                        flag = False
                if flag:
                    agent.cpu_min = cpu_min
                agent.q.put(task)
                agent.q_backlog += task.size

                agent.cpu = agent.cpu_min
                total_energy_cost[agent.index] += args.k * (agent.cpu)**3 * args.T_slot
                pro_task_size = agent.cpu * args.T_slot
                while pro_task_size != 0 and not agent.q.empty():
                    if agent.q.queue[0].size > pro_task_size:
                        agent.q.queue[0].size -= pro_task_size
                        agent.q_backlog -= pro_task_size
                        pro_task_size = 0
                    else:
                        pro_task_size -= agent.q.queue[0].size
                        agent.q_backlog -= agent.q.queue[0].size
                        agent.q.get()


    return sum(total_energy_cost) / args.T / args.device_num, fail_task_num, total_task_num