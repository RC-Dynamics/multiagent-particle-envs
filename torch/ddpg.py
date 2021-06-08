import os

import numpy as np

import torch
from buffer import ReplayBuffer
from network import Actor, Critic, TargetActor, TargetCritic
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
from utils import gumbel_softmax, onehot_from_logits


class AgentTrainer(object):
    def __init__(self, name, obs_shape, act_space, args):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents):
        raise NotImplemented()


class DDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, obs_shape_n,
                 act_shape_n, agent_index, args, discrete=False):
        self.name = name
        self.agent_index = agent_index
        self.args = args
        self.discrete = discrete

        # Train stuff
        if isinstance(obs_shape_n, int):
            temp = [0 for _ in range(self.agent_index + 1)]
            temp[self.agent_index] = obs_shape_n
            obs_shape_n = temp
        if isinstance(act_shape_n, int):
            temp = [0 for _ in range(self.agent_index + 1)]
            temp[self.agent_index] = act_shape_n
            act_shape_n = temp
        self.pi = Actor(obs_shape_n[self.agent_index],
                        act_shape_n[self.agent_index]).to(args.device)
        self.Q = Critic(obs_shape_n[self.agent_index],
                        act_shape_n[self.agent_index]).to(args.device)
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
        torch.save(self.pi.state_dict(),
                   f'{self.args.save_dir}{self.name}_actor.pth')
        torch.save(self.Q.state_dict(),
                   f'{self.args.save_dir}{self.name}_critic.pth')
        torch.save(self.pi_opt.state_dict(),
                   f'{self.args.save_dir}{self.name}_actor_optim.pth')
        torch.save(self.Q_opt.state_dict(),
                   f'{self.args.save_dir}{self.name}_critic_optim.pth')

    def load(self, load_path):
        self.pi.load_state_dict(
            torch.load(f'{load_path}{self.name}_actor.pth')
        )
        self.Q.load_state_dict(
            torch.load(f'{load_path}{self.name}_critic.pth')
        )
        self.pi_opt.load_state_dict(
            torch.load(f'{load_path}{self.name}_actor_optim.pth')
        )
        self.Q_opt.load_state_dict(
            torch.load(f'{load_path}{self.name}_critic_optim.pth')
        )
        self.tgt_pi = TargetActor(self.pi)
        self.tgt_Q = TargetCritic(self.Q)

    def action(self, obs, train=False):
        obs = torch.Tensor(obs).to(self.args.device)
        act = self.pi(obs)
        if self.discrete:
            act = onehot_from_logits(act)
        else:
            act = torch.tanh(act)
            if train:
                noise = torch.normal(0., 1., size=act.shape).to(self.args.device)
                act = torch.clamp(act + noise, -1., 1.)
        return act.detach().cpu().numpy()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        metrics = {}
        # replay buffer is not large enough
        if len(self.replay_buffer) < self.min_replay_buffer_len:
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(
            self.args.batch_size)
        index = self.replay_sample_index
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        obs = torch.Tensor(obs).to(self.args.device).float()
        obs_next = torch.Tensor(obs_next).to(self.args.device).float()
        act = torch.Tensor(act).to(self.args.device).float()
        rew = torch.Tensor(rew).to(self.args.device).float().unsqueeze(1)
        dones = torch.Tensor(done).to(self.args.device).float()

        # train critic
        self.Q_opt.zero_grad()
        Q_v = self.Q(obs, act)  # expected Q for S,A
        A_next_v = self.tgt_pi(obs_next)  # Get an Bootstrap Action for S_next
        Q_next_v = self.tgt_Q(obs_next, A_next_v)  # Bootstrap Q_next
        Q_next_v[dones == 1.] = 0.0  # No bootstrap if transition is terminal
        # Calculate a reference Q value using the bootstrap Q
        Q_ref_v = rew + Q_next_v * self.args.gamma
        Q_loss_v = F.mse_loss(Q_v, Q_ref_v.detach())
        Q_loss_v.backward()
        self.Q_opt.step()
        metrics["train/loss_Q"] = Q_loss_v.cpu().detach().numpy()

        # train actor - Maximize Q value received over every S
        self.pi_opt.zero_grad()
        A_cur_v = self.pi(obs)
        if self.discrete:
            A_cur_v = gumbel_softmax(A_cur_v, hard=True)
        else:
            A_cur_v = torch.tanh(A_cur_v)
        pi_loss_v = -self.Q(obs, A_cur_v)
        pi_loss_v = pi_loss_v.mean()
        pi_loss_v.backward()
        self.pi_opt.step()
        metrics["train/loss_pi"] = pi_loss_v.cpu().detach().numpy()

        # Sync target networks
        self.tgt_pi.sync(alpha=1 - 1e-3)
        self.tgt_Q.sync(alpha=1 - 1e-3)

        return [metrics["train/loss_Q"], metrics["train/loss_pi"]]
