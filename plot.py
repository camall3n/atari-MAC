#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import argparse
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("directory", help='Directory where scores are saved')
parser.add_argument('-n','--n_trials', help='Number of trials to plot', type=int, default=None)
args = parser.parse_args()

dir_name, env = os.path.split(os.path.join(args.directory.rstrip("/")))

color = {
    'mac': 'black',
    'mac_16': 'black',
    'mac_320': 'xkcd:brown',
    'ac': 'xkcd:pink',
    'a2c': 'xkcd:red',
    'a2c_16': 'xkcd:red',
    'a2c_320': 'xkcd:magenta',
    'reinforce': 'xkcd:teal',
    'adv-reinforce': 'xkcd:blue',
}

def visualize(env, n_trials=None):
    alg_scores = {}
    env_dir = os.path.join(dir_name, env)
    algorithms = os.listdir(env_dir)
    for algorithm in algorithms:
        print("Algorithm: {}".format(algorithm))
        alg_dir = os.path.join(env_dir, algorithm)
        alg_scores[algorithm] = []
        for seed in os.listdir(alg_dir):
            if n_trials is None or int(seed) < n_trials:
                seed_dir = os.path.join(alg_dir, seed)
                score_file = os.path.join(seed_dir, "scores.txt")
                scores = np.loadtxt(score_file)
                # print(scores)
                alg_scores[algorithm].append(scores)
        max_ep = min([len(scores) for scores in alg_scores[algorithm]])
        alg_scores[algorithm] = np.asarray([scores[:max_ep] for scores in alg_scores[algorithm]])
        print("Trials:    {}".format(len(alg_scores[algorithm])))
        print("Timesteps: {}".format(max_ep))
        print()

    fig, ax = plt.subplots()
    for algorithm in algorithms:
        (N, max_ep) = alg_scores[algorithm].shape
        if n_trials != None and n_trials <= N:
            N = n_trials
        data = np.mean(alg_scores[algorithm][:N,:], axis=0)
        err  = np.std(alg_scores[algorithm][:N,:], axis=0) / np.sqrt(N)
        time = np.arange(0, max_ep)
        plt.plot(time, data, label=algorithm, alpha=1.0, color=color[algorithm])
        ax.fill_between(time, data-err, data+err, alpha=0.5, color=color[algorithm])

    plt.legend()
    if env == 'CartPole-v0':
        plt.ylim([0,250])
    elif env == 'LunarLander-v2':
        plt.ylim([-250,50])
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('{} ({} trials)'.format(env, N))

visualize(env, args.n_trials)
plt.show()
plt.close()
