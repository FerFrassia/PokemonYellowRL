from stable_baselines3 import PPO
from train import make_vec
import imageio as imageio
import json
import os

CHECKPOINT = 'models/best/explore_v4/best_model.zip'
VIDEOFOLDERNAME = 'videos'

def create_video():
	model = PPO.load(CHECKPOINT, device='cpu')
	venv = make_vec(n_envs=1)

	obs = venv.reset()
	frames = []
	log = []
	while (True):
		action, _ = model.predict(obs, deterministic=False)
		obs, rewards, dones, infos = venv.step(action)
		log.append({'player_x': int(infos[0]['player_x']),
		            'player_y': int(infos[0]['player_y']),
		            'current_map': int(infos[0]['current_map']),
		            'action': int(action[0]),
		            'new_maps': int(infos[0]['new_maps']),
		            'exp_gained': int(infos[0]['exp_gained'])})
		if dones[0]:
			break

		envFrames = venv.get_images()
		frames.append(envFrames[0])

	os.makedirs("./" + VIDEOFOLDERNAME, exist_ok=True)
	with open("best.json", "w", encoding="utf-8") as f:
		json.dump(log, f, indent=2)
	imageio.mimsave(os.path.join(VIDEOFOLDERNAME, 'best.mp4'), frames, fps=10)

	venv.close()

if __name__ == '__main__':
	create_video()


