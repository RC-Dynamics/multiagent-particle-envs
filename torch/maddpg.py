import os

import numpy as np

import torch
from buffer import ReplayBuffer
from ddpg import DDPGAgentTrainer
from network import Actor, Critic, TargetActor, TargetCritic
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


def onehot_from_logits(logits):
    """
    Given batch of logits, return one-hot sample 
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


class MADDPGAgentTrainer(DDPGAgentTrainer):
    def __init__(self, name, obs_shape_n, act_shape_n, agent_index, args, local_q_func=False, discrete=True):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.discrete = discrete

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
        torch.save(self.pi.state_dict(),
                   f'{self.args.save_dir}/{self.name}_actor.pth')
        torch.save(self.Q.state_dict(),
                   f'{self.args.save_dir}/{self.name}_critic.pth')
        torch.save(self.pi_opt.state_dict(),
                   f'{self.args.save_dir}/{self.name}_actor_optim.pth')
        torch.save(self.Q_opt.state_dict(),
                   f'{self.args.save_dir}/{self.name}_critic_optim.pth')

    def load(self, load_path):
        self.pi.load_state_dict(torch.load(
            f'{load_path}/{self.name}_actor.pth'))
        self.Q.load_state_dict(torch.load(
            f'{load_path}/{self.name}_critic.pth'))
        self.pi_opt.load_state_dict(torch.load(
            f'{load_path}/{self.name}_actor_optim.pth'))
        self.Q_opt.load_state_dict(torch.load(
            f'{load_path}/{self.name}_critic_optim.pth'))
        self.tgt_pi = TargetActor(self.pi)
        self.tgt_Q = TargetCritic(self.Q)

    def action(self, obs, train=False):
        obs = torch.Tensor([obs]).to(self.args.device)
        act = self.pi(obs)
        if self.discrete:
            act = onehot_from_logits(act)
        else:
            act = torch.tanh(act)
        return act.detach().cpu().numpy().squeeze()

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
        obs_n = []
        obs_next_n = []
        act_n = []
        new_acts_n = []
        target_act_next_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(
                index)
            obs_v = torch.Tensor(obs).to(self.args.device).float()
            obs_next_v = torch.Tensor(obs_next).to(self.args.device).float()
            act_v = torch.Tensor(act).to(self.args.device).float()
            obs_n.append(obs_v)
            obs_next_n.append(obs_next_v)
            act_n.append(act_v)
            new_act = agents[i].pi(obs_v)
            if i == self.agent_index:
                my_act_logits = new_act
                if self.discrete:
                    new_act = gumbel_softmax(new_act, hard=True)
                else:
                    new_act = torch.tanh(new_act)
            else:
                if self.discrete:
                    new_act = onehot_from_logits(new_act)
                else:
                    new_act = torch.tanh(new_act)
            new_acts_n.append(new_act)
            tgt_res = agents[i].tgt_pi(obs_next_v)
            tgt_res = onehot_from_logits(tgt_res)
            target_act_next_n.append(tgt_res)
        _, _, rew, _, done = self.replay_buffer.sample_index(index)
        rew_v = torch.Tensor(rew).to(self.args.device).float().unsqueeze(1)
        dones_v = torch.Tensor(done).to(self.args.device).float()

        obs_nv = torch.cat(obs_n, dim=1).to(self.args.device)
        obs_next_nv = torch.cat(obs_next_n, dim=1).to(self.args.device)
        act_nv = torch.cat(act_n, dim=1).to(self.args.device)
        new_acts_nv = torch.cat(new_acts_n, dim=1).to(self.args.device)
        target_act_next_nv = torch.cat(
            target_act_next_n, dim=1).to(self.args.device)

        # train critic
        Q_v = self.Q(obs_nv, act_nv)  # expected Q for S,A
        # Get an Bootstrap Action for S_next
        Q_next_v = self.tgt_Q(
            obs_next_nv, target_act_next_nv)  # Bootstrap Q_next
        Q_next_v[dones_v == 1.] = 0.0  # No bootstrap if transition is terminal
        # Calculate a reference Q value using the bootstrap Q
        Q_ref_v = rew_v + Q_next_v * self.args.gamma
        self.Q_opt.zero_grad()
        Q_loss_v = torch.mean(torch.square(Q_ref_v.detach() - Q_v))
        Q_loss_v.backward()
        clip_grad_norm_(self.Q.parameters(), 0.5)
        self.Q_opt.step()
        metrics["train/loss_Q"] = Q_loss_v.cpu().detach().numpy()

        # train actor - Maximize Q value received over every S
        self.pi_opt.zero_grad()
        pi_loss_v = -self.Q(obs_nv, new_acts_nv)
        pi_loss_v = pi_loss_v.mean()
        pi_loss_v += (my_act_logits**2).mean() * 1e-3
        pi_loss_v.backward()
        clip_grad_norm_(self.pi.parameters(), 0.5)
        self.pi_opt.step()
        metrics["train/loss_pi"] = pi_loss_v.cpu().detach().numpy()

        # Sync target networks
        self.tgt_pi.sync(alpha=0.99)
        self.tgt_Q.sync(alpha=0.99)

        return [metrics["train/loss_Q"], metrics["train/loss_pi"], np.mean(Q_v.cpu().detach().numpy()), np.mean(rew), np.mean(Q_next_v.cpu().detach().numpy()), np.std(Q_v.cpu().detach().numpy())]
