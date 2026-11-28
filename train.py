import time
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from wrapper import make_env
import argparse
import os

SEED = 55

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    return parser.parse_args()

def make_vec(n_envs=8, n_stack=4, env_kwargs=None):
	venv = make_vec_env(make_env, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
	return VecFrameStack(venv, n_stack=n_stack)

def benchmark(venv, n_iters=200):
	#Step with random actions. Return (steps_per_sec, episode_rewards)
	venv.reset()
	start = time.time()
	total_rewards = []
	for _ in range(n_iters):
		actions = []
		for _ in range(venv.num_envs):
			actions.append(venv.action_space.sample())

		obs, rewards, dones, infos = venv.step(actions)
		for i in range(venv.num_envs):
			if dones[i]:  # env i finished
				total_rewards.append(infos[i]['episode']['r'])


	elapsed = time.time() - start
	steps_per_sec = n_iters * venv.num_envs / elapsed
	return steps_per_sec, total_rewards

def train(name, venv, eval_venv, total_timesteps=600_000):
	os.makedirs('models/final', exist_ok=True)

	model = PPO('CnnPolicy',
	            venv,
	            n_steps=128,
	            batch_size=128,
	            n_epochs=4,
	            gamma=0.99,
	            ent_coef=0.02,
	            learning_rate=3e-4,
	            device='cuda',
	            verbose=1,
	            tensorboard_log='runs',
	            seed=SEED)

	checkpoint_callback = CheckpointCallback(save_freq=25_000 // venv.num_envs,
	                                         save_path='models/checkpoints',
	                                         name_prefix=name)

	eval_callback = EvalCallback(eval_venv,
	                             best_model_save_path='models/best/' + name,
	                             log_path='models/best/' + name,
	                             eval_freq=25_000 // venv.num_envs,
	                             n_eval_episodes=3,
	                             deterministic=False)
 
	record_callback = RecordCallback()


	model.learn(total_timesteps=total_timesteps, log_interval=10, tb_log_name=name, callback=[checkpoint_callback, eval_callback, record_callback])
	model.save('models/final/' + name)
	return model

class RecordCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        
    def _on_step():
        
        
        infos = self.locals['infos']
        actions = self.locals['actions']
        dones = self.locals['dones']
        
        self.logger.()
        print(infos)
        print(actions)
        print(dines)
        
        return True

if __name__ == '__main__':
	args = parse_args()
	venv = make_vec()
	eval_venv = make_vec(n_envs=1)

	train(args.name, venv, eval_venv)
	venv.close()
	eval_venv.close()
