#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy, load_feasibiltiy
from safe_rl.utils.logx import EpochLogger
import matplotlib.pyplot as plt
import os.path as osp
import json
plt.rcParams.update({'font.size': 16})

def collect_obs(env, args, bound=2.5):
    env.reset()
    config_dict = env.world_config_dict
    geoms_dict = config_dict['geoms']
    hazard_dict = {}
    for key, val in geoms_dict.items():
        hazard_dict.update({key:val['pos'].tolist()})
    x = np.linspace(-bound, bound, 100)
    y = np.linspace(-bound, bound, 100)
    X, Y = np.meshgrid(x, y)
    obs = []
    for i in range(100):
        for j in range(100):
            print(i, j)
            config_dict['robot_xy'] = np.array([X[i, j], Y[i, j]])
            env.world_config_dict = config_dict
            env.world.rebuild(config_dict, state=False)
            # env.render()
            obs.append(env.obs())
    np.save(osp.join(args.fpath, 'obs.npy'), np.array(obs))
    np.save(osp.join(args.fpath, 'config.npy'), config_dict, allow_pickle=True)
    with open(osp.join(args.fpath, 'world_config.json'), 'w') as f:
        json.dump(hazard_dict, f)


def visualize_region(env, args, get_feasibility, bound=2.5):
    obs = np.load(osp.join(args.fpath, 'obs.npy'))
    with open(osp.join(args.fpath, 'world_config.json')) as f:
        config_dict = json.load(f)
    multiplier, vc = get_feasibility(obs)
    def plot_region(feasibility, name):
        x = np.linspace(-bound, bound, 100)
        y = np.linspace(-bound, bound, 100)
        X, Y = np.meshgrid(x, y)
        feasibility = np.reshape(feasibility, X.shape)
        feasibility = feasibility + 0.2
        # feasibility = np.clip(feasibility, 0, np.inf)
        fig = plt.figure(figsize=[3, 3])
        ax = plt.axes()
        # ctf = ax.contourf(X, Y, feasibility, cmap='Accent')
        ax.contour(X, Y, feasibility, levels=0, color='red', linewidth=3)
        # plt.colorbar(ctf)
        plt.axis('equal')
        rect1 = plt.Rectangle((0, 0), 1, 1, fc=None,
                              ec='green', linewidth=3)
        h = [] # rect1
        l = [] # 'Feasible Region'
        for key, val in config_dict.items():
            if key.startswith('hazard'):
                c1 = plt.Circle((val[0], val[1]), radius=0.30, fill=False, ec='blue', linewidth=1.5)
                ax.add_patch(c1)
                if key[6] == '0':
                    h.append(c1)
                    l.append('Hazards')
            elif key.startswith('pillar'):
                c2 = plt.Circle((val[0], val[1]), radius=0.30, fill=False, ec='blue', linewidth=1.5)
                ax.add_patch(c2)
                if key[6] == '0':
                    h.append(c2)
                    l.append('Pillars')
            elif key.startswith('goal'):
                c3 = plt.Circle((val[0], val[1]), radius=0.30, fc='green', ec='green', alpha=0.4, linewidth=1.5)
                ax.add_patch(c3)
                h.append(c3)
                l.append('Goal')

        # plt.legend(h, l, loc='upper right', ncol=2)
        plt.tight_layout(pad=0.5)
        plt.savefig(osp.join(args.fpath, 'region-cn-{}.png'.format(name)))

    plot_region(vc, 'vc')
    plot_region(multiplier, 'mu')

def get_pic(env, args):
    with open(osp.join(args.fpath, 'world_config.json')) as f:
        config_dict = json.load(f)
    env.reset()
    world_dict = env.world_config_dict
    for key, val in world_dict['geoms'].items():
        for key1, val1 in config_dict.items():
            if key == key1:
                val.update({'pos':np.array(val1)})
    env.world.rebuild(world_dict)
    while True:
        env.render()
    a = 1

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    from safe_rl.utils.custom_env_utils import register_custom_env
    import gym
    register_custom_env()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='/home/mahaitong/PycharmProjects/safety-starter-agents/data/2021-12-31_ppo_dual_ascent_Safexp-CustomGoal2-v0/2021-12-31_23-43-36-ppo_dual_ascent_Safexp-CustomGoal2-v0_s0')
    parser.add_argument('--len', '-l', type=int, default=None)
    parser.add_argument('--episodes', '-n', type=int, default=5)
    parser.add_argument('--norender', '-nr', action='store_true', default=False)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    # env, get_action, sess = load_policy(args.fpath,
    #                                     args.itr if args.itr >=0 else 'last',
    #                                     args.deterministic)
    env, get_feasibility_indicator, sess = load_feasibiltiy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    # collect_obs(env, args)
    # run_policy(env, get_action, args.len, args.episodes, not(args.norender))
    visualize_region(env,args,get_feasibility_indicator)
    # get_pic(env, args)
