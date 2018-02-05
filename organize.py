#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import argparse
import fnmatch
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("src", help='Directory to organize')
parser.add_argument("dest", help='Directory in which to save clean results')
parser.add_argument('--note', help='Manually add field (if missing)', type=str, default=None)
parser.add_argument('--num_cpus', help='Manually add field (if missing)', type=int, default=None)
parser.add_argument('--seed', help='Manually add field (if missing)', choices=['generate'], default=None)
args = parser.parse_args()

if not os.path.exists(args.dest):
    os.makedirs(args.dest)

for env in os.listdir(args.src):
    print(env)
    oldenv_dir = os.path.join(args.src, env)
    newenv_dir = os.path.join(args.dest, env)

    generated_seed = 0
    for trial in os.listdir(oldenv_dir):
        oldtrial_dir = os.path.join(oldenv_dir, trial)
        oldfile_name = os.path.join(oldtrial_dir, "log.txt")
        if os.path.exists(oldfile_name):
            with open(oldfile_name) as oldfile:
                # Extract params from header
                params = list(islice(oldfile, 2,16))
                params = [line.strip().split(": ") for line in params] # parse key/val pairs
                params = [p for p in params if len(p) == 2] # ignore keys with no value
                params = dict(params)

                # def checkField(field, default):
                #     if field not in params:
                #         if default is None:
                #             print(params)
                #             print("Warning: '{}' has no {}. Skipping.".format(oldfile_name, field))
                #             print("  You can manually set a backup value with '--{}'.".format(field))
                #             return False
                #         else:
                #             params[field] = default
                #             return True
                #     else:
                #         return True
                # if not checkField('note', args.note):
                #     continue
                # if not checkField('num_cpus', args.num_cpus):
                #     continue
                # if not checkField('seed', args.seed):
                #     continue
                algorithm = params['note'].lower()
                if 'aac' in algorithm:
                    algorithm = 'a2c'
                if 'mac' in algorithm:
                    algorithm = 'mac'
                n_cpus = int(params['num_cpus'])
                if params['seed'] == 'generate':
                    seed = generated_seed
                    generated_seed += 1
                else:
                    seed = int(params['seed'])

                # Open new file
                alg_str = '{}_{}'.format(algorithm, n_cpus)
                newtrial_dir = os.path.join(newenv_dir, alg_str)
                seed_dir = os.path.join(newtrial_dir, '{:03d}'.format(seed))
                if not os.path.exists(seed_dir):
                    os.makedirs(seed_dir)
                newfile_name = os.path.join(seed_dir, "scores.txt")
                with open(newfile_name, 'w') as newfile:

                    # Copy scores from oldfile to newfile
                    for line in oldfile:
                        if 'eval_avg_score' in line:
                            # Line will look like this:
                            # | eval_avg_score  | 655.859  |\n
                            # We just want score: 655.859
                            score = re.split(r'[ |\n]+', line.strip())[2]
                            newfile.write('{} '.format(score))
                    print('  {}  ->  {}'.format(oldfile_name, newfile_name))
