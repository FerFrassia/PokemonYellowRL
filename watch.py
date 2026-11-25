from stable_baselines3 import PPO
from train import make_vec
import imageio as imageio
import json
import os
import numpy as np
import cv2
from wrapper import PokemonYellowEnv

VERSIONNUMBER = 'v6'
CHECKPOINT = 'models/best/explore_' + VERSIONNUMBER + '/best_model.zip'
VIDEOFOLDERNAME = 'videos'
VIDEOPATH = os.path.join(VIDEOFOLDERNAME, VERSIONNUMBER, VERSIONNUMBER + '.mp4')
JSONPATH = os.path.join(VIDEOFOLDERNAME, VERSIONNUMBER, VERSIONNUMBER + '.json')
NUM_ENVS = 24
ROWS = 4
COLS = 6

def create_video():
	assert ROWS * COLS == NUM_ENVS
	model = PPO.load(CHECKPOINT, device='cpu')
	venv = make_vec(n_envs=NUM_ENVS)
	try:
		obs = venv.reset()
		log = []
		os.makedirs(os.path.join(VIDEOFOLDERNAME, VERSIONNUMBER), exist_ok=True)
		battle_steps = [0 for _ in range(NUM_ENVS)]
		battle_starts = [0 for _ in range(NUM_ENVS)]
		was_in_battle = [False for _ in range(NUM_ENVS)]
		actions_in_battle = [{name: 0 for name in PokemonYellowEnv.ACTIONS} for _ in range(NUM_ENVS)]

		with imageio.get_writer(VIDEOPATH, fps=10) as writer:
			while (True):
				actions, _ = model.predict(obs)
				obs, rewards, dones, infos = venv.step(actions)

				for i in range(NUM_ENVS):
					if infos[i]['is_in_battle']:
						battle_steps[i] += 1
						actions_in_battle[i][PokemonYellowEnv.ACTIONS[int(actions[i])]] += 1

						if not was_in_battle[i]:
							was_in_battle[i] = True
							battle_starts[i] += 1
					else:
						was_in_battle[i] = False

					if dones[i]:
						log.append({'env': i,
									'player_x': int(infos[i]['player_x']),
		            				'player_y': int(infos[i]['player_y']),
		            				'current_map': int(infos[i]['current_map']),
									'reward': float(infos[i]['episode']['r']),
		            				'new_maps': int(infos[i]['new_maps']),
		            				'exp_gained': int(infos[i]['exp_gained']),
									'amount_of_visited_tiles': int(infos[i]['amount_of_visited_tiles']),
								   	'battle_steps': battle_steps[i],
									'battle_starts': battle_starts[i],
									'actions_in_battle': actions_in_battle[i]})

				envFrames = venv.get_images()
				for i in range(NUM_ENVS):
					cv2.putText(envFrames[i], str(i), (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
					h, w = envFrames[i].shape[:2]
					cv2.rectangle(envFrames[i], (0, 0), (w-1, h-1), (0, 0, 0), 1)

				rows = [np.hstack(envFrames[r*COLS:(r+1)*COLS]) for r in range(ROWS)]
				grid = np.vstack(rows)

				writer.append_data(grid)
				if len(log) >= NUM_ENVS:
					break

			with open(JSONPATH, "w", encoding="utf-8") as f:
				json.dump(log, f, indent=2)

	finally:
		venv.close()

if __name__ == '__main__':
	create_video()


