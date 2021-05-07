import os

import numpy as np

import torch
from buffer import ReplayBuffer
from ddpg import DDPGAgentTrainer
from network import Actor, Critic, TargetActor, TargetCritic
from torch.nn import functional as F
from torch.optim import Adam


class MADDPGAgentTrainer(DDPGAgentTrainer):
    def __init__(self, name, obs_shape_n, act_shape_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args

        # Train stuff
        self.pi = Actor(obs_shape_n[self.agent_index],
                        act_shape_n[self.agent_index]).to(args.device)
        self.Q = Critic(sum(obs_shape_n), sum(act_shape_n)).to(args.device)
        self.tgt_pi = TargetActor(self.pi)
        self.tgt_Q = TargetCritic(self.Q)
        self.pi_opt = Adam(self.pi.parameters(), lr=args.lr)
        self.Q_opt = Adam(self.Q.parameters(), lr=args.lr)
        self.grad_clipping = 0.5

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.min_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def save(self):
        torch.save(self.pi.state_dict(), f'{self.args.save_dir}{self.name}_actor.pth')
        torch.save(self.Q.state_dict(), f'{self.args.save_dir}{self.name}_critic.pth')
        torch.save(self.pi_opt.state_dict(), f'{self.args.save_dir}{self.name}_actor_optim.pth')
        torch.save(self.Q_opt.state_dict(), f'{self.args.save_dir}{self.name}_critic_optim.pth')

    def load(self, load_path):
        self.pi.load_state_dict(torch.load(f'{load_path}{self.name}_actor.pth'))
        self.Q.load_state_dict(torch.load(f'{load_path}{self.name}_critic.pth'))
        self.pi_opt.load_state_dict(torch.load(f'{load_path}{self.name}_actor_optim.pth'))
        self.Q_opt.load_state_dict(torch.load(f'{load_path}{self.name}_critic_optim.pth'))
        self.tgt_pi = TargetActor(self.pi)
        self.tgt_Q = TargetCritic(self.Q)

    def action(self, obs, train=False):
        obs = torch.Tensor(obs).to(self.args.device)
        act = self.pi(obs)
        if train:
            noise = torch.normal(0.,1., size=act.shape).to(self.args.device)
            act = torch.clamp(act + noise, -1.,1.)
        return act.detach().cpu().numpy()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        metrics = {}

        if len(self.replay_buffer) < self.min_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(
            self.args.batch_size)
        # collect replay sample from all agents
        tgt_actions = []
        actual_actions = []
        taken_actions = []
        obs_n = []
        obs_next_n = []
        index = self.replay_sample_index

        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        rew_v = torch.Tensor(rew).to(self.args.device).float().unsqueeze(1)
        dones_v = torch.Tensor(done).to(self.args.device).float()

        for agent in agents:
            obs, act, _, obs_next, _ = agent.replay_buffer.sample_index(index)
            obs_agent_state = torch.Tensor(obs).to(self.args.device).float()
            obs_agent_next_state = torch.Tensor(obs_next).to(self.args.device).float()
            act_agent = torch.Tensor(act).to(self.args.device).float()
            new_pi = agent.tgt_pi(obs_agent_next_state)
            if agent is self:
                act = agent.pi(obs_agent_state)
                actual_actions.append(act)
            else:
                actual_actions.append(act_agent)
            tgt_actions.append(new_pi)
            taken_actions.append(act_agent)
            obs_n.append(obs_agent_state)
            obs_next_n.append(obs_agent_next_state)

        tgt_actions_v = torch.cat(tgt_actions, dim=1).to(self.args.device)
        actual_actions_v = torch.cat(actual_actions, dim=1).to(self.args.device)
        taken_actions_nv = torch.cat(taken_actions, dim=1).to(self.args.device)
        obs_nv = torch.cat(obs_n, dim=1).to(self.args.device)
        obs_next_nv = torch.cat(obs_next_n, dim=1).to(self.args.device)

        # train critic
        self.Q_opt.zero_grad()
        Q_v = self.Q(obs_nv, taken_actions_nv)  # expected Q for S,A
        # Get an Bootstrap Action for S_next
        Q_next_v = self.tgt_Q(obs_next_nv, tgt_actions_v)  # Bootstrap Q_next
        Q_next_v[dones_v == 1.] = 0.0  # No bootstrap if transition is terminal
        # Calculate a reference Q value using the bootstrap Q
        Q_ref_v = rew_v + Q_next_v * self.args.gamma
        Q_loss_v = F.mse_loss(Q_v, Q_ref_v.detach())
        Q_loss_v.backward()
        self.Q_opt.step()
        metrics["train/loss_Q"] = Q_loss_v.cpu().detach().numpy()

        # train actor - Maximize Q value received over every S
        self.pi_opt.zero_grad()
        pi_loss_v = -self.Q(obs_nv, actual_actions_v)
        pi_loss_v = pi_loss_v.mean()
        pi_loss_v.backward()
        self.pi_opt.step()
        metrics["train/loss_pi"] = pi_loss_v.cpu().detach().numpy()

        # Sync target networks
        self.tgt_pi.sync(alpha=1 - 1e-3)
        self.tgt_Q.sync(alpha=1 - 1e-3)

        return [metrics["train/loss_Q"], metrics["train/loss_pi"], np.mean(Q_v.cpu().detach().numpy()), np.mean(rew), np.mean(Q_next_v.cpu().detach().numpy()), np.std(Q_v.cpu().detach().numpy()) ]
