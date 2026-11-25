import time
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from wrapper import make_env

VERSIONNAME = 'explore_v4'
SEED = 55

def make_vec(n_envs=4, n_stack=4):
	venv = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
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

def train(venv, eval_venv, total_timesteps=200_000):
	model = PPO('CnnPolicy',
	            venv,
	            n_steps=128,
	            batch_size=128,
	            n_epochs=4,
	            gamma=0.99,
	            ent_coef=0.02,
	            learning_rate=3e-4,
	            device='cpu',
	            verbose=1,
	            tensorboard_log='runs',
	            seed=SEED)

	checkpoint_callback = CheckpointCallback(save_freq=25_000 // venv.num_envs,
	                                         save_path='models/checkpoints',
	                                         name_prefix=VERSIONNAME)

	eval_callback = EvalCallback(eval_venv,
	                             best_model_save_path='models/best/' + VERSIONNAME,
	                             log_path='models/best/' + VERSIONNAME,
	                             eval_freq=25_000 // venv.num_envs,
	                             n_eval_episodes=3,
	                             deterministic=False)

	model.learn(total_timesteps=total_timesteps, log_interval=10, tb_log_name=VERSIONNAME, callback=[checkpoint_callback, eval_callback])
	model.save('models/ppo_test')
	return model

if __name__ == '__main__':
	venv = make_vec()
	eval_venv = make_vec(n_envs=1)
	# print('obs shape:', obs.shape)
	# steps_per_sec, ep_rewards = benchmark(venv)
	# print(f'steps/sec: {steps_per_sec}')
	# print(f'episode rewards: {ep_rewards}')
	train(venv, eval_venv)
	venv.close()
	eval_venv.close()
