import torch
import numpy as np

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



def train_epoch(args, logging, epoch, agents, env):
    for agent in agents:
        if agent.replay_buffer.__len__() > args.batch_size:
            states, actions, act_cogns, rewards, next_states, observations, new_obs, dones = agent.replay_buffer.sample(int(args.batch_size))
        else:
            states, actions, act_cogns, rewards, next_states, observations, new_obs, dones = agent.replay_buffer.sample(agent.replay_buffer.__len__())
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = agent.critic(states, actions, act_cogns)

        with torch.no_grad():
            next_actions = []
            for i in new_obs:
                output, cpu_fre = agent.target_actor(i)
                cpu_fre = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
                action = torch.from_numpy(np.array([agent.index])).float()
                action = torch.cat([action, output, cpu_fre])
                next_actions.append(action)
            act_neis = [torch.zeros(args.act_cognition_size) for _ in range(len(next_actions))]
            future_rewards = agent.target_critic(next_states, next_actions, act_neis)
            target_q_values = rewards + (1 - dones) * args.gamma * future_rewards

        critic_loss = agent.mse(q_values, target_q_values)
        logging.info("agent-{} | critic loss: {}".format(agent.index, critic_loss.item()))
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        predicted_actions = []
        for i in observations:
            output, cpu_fre = agent.actor(i)
            cpu_fre = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
            action = torch.from_numpy(np.array([agent.index])).float()
            action = torch.cat([action, output, cpu_fre])
            predicted_actions.append(action)
        act_cogns = [i.detach() for i in act_cogns]
        states = [i.detach() for i in states]
        actor_loss = - agent.critic(states, predicted_actions, act_cogns).mean()
        logging.info("agent-{} | actor loss: {}".format(agent.index, actor_loss.item()))

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # 更新目标 Critic 和 Actor 网络
        if epoch % args.target_update_freq == 0:
            soft_update(agent.target_critic, agent.critic, args.tau)
            soft_update(agent.target_actor, agent.actor, args.tau)






def ddpg_train_epoch(args, logging, epoch, agents, env):
    for agent in agents:
        states, actions, rewards, next_states, dones = agent.replay_buffer.sample(int(args.batch_size))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = agent.critic(states, actions, agent.index)

        with torch.no_grad():
            next_actions = []
            for observations in next_states:
                actions = [torch.zeros(args.act_em_size) for i in range(len(agents))]
                for ag in agents:
                    output, cpu_fre = ag.target_actor(observations[ag.index])
                    cpu_fre = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
                    action = torch.from_numpy(np.array([ag.index])).float()
                    action = torch.cat([action, output, cpu_fre])
                    actions[ag.index] = action
                next_actions.append(actions)
            future_rewards = agent.target_critic(next_states, next_actions, agent.index)
            target_q_values = rewards + (1 - dones) * args.gamma * future_rewards

        critic_loss = agent.mse(q_values, target_q_values)
        logging.info("agent-{} | critic loss: {}".format(agent.index, critic_loss.item()))
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        predicted_actions = []
        for observations in states:
            actions = [torch.zeros(args.act_em_size) for i in range(len(agents))]
            for ag in agents:
                output, cpu_fre = agent.actor(observations[agent.index])
                cpu_fre = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
                action = torch.from_numpy(np.array([agent.index])).float()
                action = torch.cat([action, output, cpu_fre])
                actions[ag.index] = action
            predicted_actions.append(actions)
        actor_loss = - agent.critic(states, predicted_actions, agent.index).mean()
        logging.info("agent-{} | actor loss: {}".format(agent.index, actor_loss.item()))
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # 更新目标 Critic 和 Actor 网络
        if epoch % args.target_update_freq == 0:
            soft_update(agent.target_critic, agent.critic, args.tau)
            soft_update(agent.target_actor, agent.actor, args.tau)

def coma_train_epoch(args, logging, epoch, agents, env):
    critic = agents[0].critic
    target_critic = agents[0].target_critic
    critic_optimizer = agents[0].critic_optimizer
    for agent in agents:
        states, actions, rewards, next_states, dones = agent.replay_buffer.sample(int(args.batch_size * len(agent.replay_buffer.buffer)))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = critic(states, actions, agent.index)

        with torch.no_grad():
            next_actions = []
            for observations in next_states:
                actions = [torch.zeros(args.act_em_size) for i in range(len(agents))]
                for ag in agents:
                    output, cpu_fre = ag.target_actor(observations[ag.index])
                    cpu_fre = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
                    action = torch.from_numpy(np.array([ag.index])).float()
                    action = torch.cat([action, output, cpu_fre])
                    actions[ag.index] = action
                next_actions.append(actions)
            future_rewards = target_critic(next_states, next_actions, agent.index)
            target_q_values = rewards + (1 - dones) * args.gamma * future_rewards

        critic_loss = agent.mse(q_values, target_q_values)

        logging.info("agent-{} | critic loss: {}".format(agent.index, critic_loss.item()))
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()


        predicted_actions = []
        for observations in states:
            actions = [torch.zeros(args.act_em_size) for i in range(len(agents))]
            for ag in agents:
                output, cpu_fre = agent.actor(observations[agent.index])
                cpu_fre = agent.cpu_min + cpu_fre * (agent.cpu_max - agent.cpu_min)
                action = torch.from_numpy(np.array([agent.index])).float()
                action = torch.cat([action, output, cpu_fre])
                actions[ag.index] = action
            predicted_actions.append(actions)
        actor_loss = - critic(states, predicted_actions, agent.index).mean()
        logging.info("agent-{} | actor loss: {}".format(agent.index, actor_loss.item()))


        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        if epoch % args.target_update_freq == 0:
            soft_update(agent.target_actor, agent.actor, args.tau)

    if epoch % args.target_update_freq == 0:
        soft_update(target_critic, critic, args.tau)




def dqn_train_epoch(args, logging, epoch, agents, env):
    for agent in agents:
        states, actions, rewards, next_states, dones = agent.replay_buffer.sample(args.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = - (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = []
        for observations in states:
            output = agent.DQN(observations)
            action = torch.from_numpy(np.array([agent.index])).float()
            action = torch.cat([action, output])
            q_values.append(action)

        q_value = []
        for i in range(len(actions)):
            q_value.append(torch.stack(q_values)[i][actions[i]])
        q_value = torch.stack(q_value)

        next_q_values = []
        for observations in next_states:
            output = agent.target_DQN(observations)
            action = torch.from_numpy(np.array([agent.index])).float()
            action = torch.cat([action, output])
            next_q_values.append(action)

        next_q_values = torch.stack(next_q_values).max(1)[0]
        q_targets = rewards + args.gamma * next_q_values * (1 - dones)
        loss = agent.mse(q_value, q_targets.detach())

        logging.info("agent-{} | loss: {}".format(agent.index, loss.item()))

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # 更新目标 Critic 和 Actor 网络
        if epoch % args.target_update_freq == 0:
            soft_update(agent.target_DQN, agent.DQN, args.tau)