#!/usr/bin/env python
import os, logging, gym, datetime
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_frames, seed, nsteps, policy, lrschedule, num_cpu, model_path, pg_coef=1.0):
    num_timesteps = int(num_frames / 4 * 1.1)
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank, isTraining=True):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)), allow_early_resets=(not isTraining))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env, episode_life=isTraining, clip_rewards=isTraining)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i, isTraining=True) for i in range(num_cpu)])
    eval_env = SubprocVecEnv([make_env(num_cpu+i, isTraining=False) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, eval_env, seed, nsteps=nsteps, total_timesteps=num_timesteps, pg_coef=pg_coef, lrschedule=lrschedule, model_path=model_path)
    eval_env.close()
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=200)
    parser.add_argument('--logdir', help='Log directory', type=str, default="logs")
    parser.add_argument('--note', help='A short note to add to the log file', type=str, default="")
    parser.add_argument('--model_path', help='Path to pre-trained model', type=str, default="")
    parser.add_argument('--num_cpus', help='Number of CPUs (i.e. number of parallel enviornments)', type=int, default=16)
    parser.add_argument('--nsteps', help='Number of steps for each rollout', type=int, default=5)
    parser.add_argument('--pg_coef', help='Coefficient for policy gradient loss (affects policy learning rate)', type=float, default=1.0)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    logdir = os.path.join(args.logdir, args.env, timestamp)
    logger.reset()
    logger.configure(logdir)
    logger.log("")
    for arg in vars(args):
        logger.log("{}: {}".format(arg, getattr(args,arg)))
    logger.log("")
    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed, nsteps=args.nsteps,
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=args.num_cpus, model_path=args.model_path, pg_coef=args.pg_coef)

if __name__ == '__main__':
    main()
