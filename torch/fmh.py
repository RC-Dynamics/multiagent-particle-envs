import argparse
import pickle
import time
from gym.spaces import discrete

import numpy as np
import PIL
from gym import spaces

import wandb
from ddpg import DDPGAgentTrainer


def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    import multiagent.scenarios as scenarios
    from multiagent.environment import MultiAgentEnv

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)
    return env


def parse_args():
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str,
                        default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int,
                        default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int,
                        default=60000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float,
                        default=0.75, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=256,
                        help="number of units in the mlp")
    parser.add_argument("--num-adversaries", type=int,
                        default=0, help="number of adversaries")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None,
                        help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="tmp/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--device", type=str,
                        default="cpu", help="Torch device")

    return parser.parse_args()


def get_trainers(env, arglist):
    trainers = []
    trainer = DDPGAgentTrainer
    manager = trainer("agent_0",
                      env.observation_space[0].shape[0]*env.n,
                      env.n*2, 0, arglist, False)
    trainers.append(manager)
    for i in range(env.n):
        act_space = env.action_space[i]
        n = 0
        discrete = False
        if isinstance(act_space, spaces.Discrete):
            n = act_space.n
            discrete = True
        elif isinstance(act_space, spaces.Box):
            n = act_space.shape[0]
        else:
            raise ValueError("Not allowed action space")
        trainers.append(
            trainer(f"agent_{i+1}", 4, n, i+1, arglist, discrete)
        )
    return trainers


def train(arglist):
    wandb.init(project='multi_particle',
               name='FMH_' + arglist.exp_name,
               entity='robocin')
    # Create environment
    env = make_env(arglist.scenario)
    # Create agent trainers
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env, arglist)

    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    if arglist.display or arglist.restore:
        print('Loading previous state...')
        for agent in trainers:
            agent.load(arglist.load_dir)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    man_act_idx = 0
    t_start = time.time()

    frames = []
    gif_index = 0
    gif = True
    print('Starting iterations...')
    train = not arglist.display
    while True:
        metrics = {}
        # get action
        if man_act_idx % 8 == 0:
            manager_obs = np.asarray(obs_n).reshape(-1)
            man_action = trainers[0].action(manager_obs)
            objectives = man_action.reshape((-1, 2))

        workers_obs = list()
        for i, obs in enumerate(obs_n):
            pos = obs[2:4]
            objective = objectives[i]
            w_obs = np.concatenate((pos, objective))
            workers_obs.append(w_obs)
        action_n = [agent.action(obs, train)
                    for agent, obs in zip(trainers[1:], workers_obs)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        # env.render()
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        new_manager_obs = np.asarray(new_obs_n).reshape(-1)
        new_workers_obs = list()
        for i, n_obs in enumerate(new_obs_n):
            pos = n_obs[2:4]
            objective = objectives[i]
            w_obs = np.concatenate((pos, objective))
            new_workers_obs.append(w_obs)

        # collect experience
        man_rew = np.sum(rew_n)
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
        trainers[0].experience(manager_obs, man_action, man_rew,
                               new_manager_obs, done, terminal)
        for i, agent in enumerate(trainers[1:]):
            n_pos = new_obs_n[i][2:4]
            objective = objectives[i]
            rew = -np.linalg.norm(n_pos - objective)
            rew_n[i] = rew
            agent.experience(workers_obs[i], action_n[i],
                             rew, new_workers_obs[i], done, terminal)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            agent_rewards[i][-1] += rew

        if done or terminal:
            if gif:
                frame = env.render(mode='rgb_array')[0]
                frame = PIL.Image.fromarray(frame)
                frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
                frames.append(frame)
                gif_path = "{}/gif.gif".format(arglist.save_dir)
                frames[0].save(
                    fp=gif_path,
                    format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=10,
                    loop=0
                )
                metrics.update({"gif": wandb.Video(gif_path, fps=10, format="gif"),
                                "gif_index": gif_index})
                gif_index += 1
                frames.clear()
                gif = False
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            if len(episode_rewards) % 1000 == 0:
                gif = True

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display or gif:
            # time.sleep(0.1)
            frame = env.render(mode='rgb_array')[0]
            frame = PIL.Image.fromarray(frame)
            frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
            frames.append(frame)
            if arglist.display:
                continue

        # update all trainers, if not in display or benchmark mode
        loss = None
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)
            if loss:
                metrics.update({
                    "{}/q_loss".format(agent.name): loss[0],
                    "{}/p_loss".format(agent.name): loss[1],
                    "{}/mean(target_q)".format(agent.name): loss[2],
                    "{}/mean(rew)".format(agent.name): loss[3],
                    "{}/mean(target_q_next)".format(agent.name): loss[4],
                    "{}/std(target_q)".format(agent.name): loss[5]
                })

        # save model, display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            for agent in trainers:
                agent.save()
            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(
                        episode_rewards[-arglist.save_rate:]),
                    [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            metrics.update({
                "episodes": len(episode_rewards),
                "mean_ep_rw": np.mean(episode_rewards[-arglist.save_rate:])
            })
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(
                np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        if len(metrics):
            metrics.update({
                "train_step": train_step,
            })
            wandb.log(metrics)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
