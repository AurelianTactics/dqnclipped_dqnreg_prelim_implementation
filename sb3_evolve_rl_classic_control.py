import gym
import numpy as np
import torch as th
import time
import argparse

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', default='CartPole-v0')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--total_timesteps', default=20000, type=int)
parser.add_argument('--loss_type', default=0, type=int)
parser.add_argument('--dqn_reg_loss_weight', default=0.1, type=float)
args = parser.parse_args()
env_name = args.env
total_timesteps = args.total_timesteps
seed = args.seed
loss_type = args.loss_type
dqn_reg_loss_weight = args.dqn_reg_loss_weight

time_int = int(time.time())
model_save_name = "saved_models/dqn_{}_{}_{}_{}".format(env_name, loss_type, seed, time_int)

env = gym.make(env_name)
env = Monitor(env)
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256])
tensorboard_log = "logs/dqn_evolve_rl_classic_control"

# hyperparams
buffer_size = max(total_timesteps // 100, 500)
learning_starts = max(total_timesteps // 1000, 100)
train_freq = 1
target_update_interval = 100
exploration_fraction = (learning_starts + 1000)/total_timesteps

# evaluations parameters
eval_env = gym.make(env_name)
eval_env = Monitor(eval_env)
eval_freq = max(1000, total_timesteps//20)
eval_log_path = "eval_logs/dqn_evolve_rl_eval_{}_{}_{}_{}".format(env_name, loss_type, seed, time_int)
eval_callback = EvalCallback(eval_env, log_path=eval_log_path, eval_freq=eval_freq, deterministic=True, render=False, n_eval_episodes=5)

if env_name == 'MountainCar-v0':
    buffer_size = 10000  # max(total_timesteps // 100, 500)
    learning_starts = 1000  # max(total_timesteps // 1000, 100)
    learning_rate = 4e-3
    batch_size = 128
    gamma = 0.98
    train_freq = 16
    target_update_interval = 600
    gradient_steps = 8
    exploration_fraction = 0.2  # (learning_starts + 1000)/total_timesteps
    exploration_final_eps = 0.07

    model = DQN(MlpPolicy, env, policy_kwargs=policy_kwargs, target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction,
                buffer_size=buffer_size, train_freq=train_freq, learning_starts=learning_starts, seed=seed,
                tensorboard_log=tensorboard_log, verbose=1,
                loss_type=loss_type, dqn_reg_loss_weight=dqn_reg_loss_weight,
                batch_size=batch_size, learning_rate=learning_rate, gamma=gamma, gradient_steps=gradient_steps,
                exploration_final_eps=exploration_final_eps)
else:
    model = DQN(MlpPolicy, env, policy_kwargs=policy_kwargs, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction,
                buffer_size=buffer_size, train_freq=train_freq, learning_starts=learning_starts, seed=seed, tensorboard_log=tensorboard_log, verbose=1,
                loss_type=loss_type, dqn_reg_loss_weight=dqn_reg_loss_weight)

model.learn(total_timesteps=total_timesteps, log_interval=100, callback=eval_callback)

model.save(model_save_name)
