from stable_baselines3 import PPO
from train import make_vec
import imageio as imageio
import json
import os
import numpy as np
import cv2
from wrapper import PokemonYellowEnv
import argparse


VIDEOFOLDERNAME = 'videos'
NUM_ENVS = 24
ROWS = 4
COLS = 6
LIVE_SCALE = 4
LIVE_DELAY_MS = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--live', action='store_true')
    return parser.parse_args()

def checkpoint_path(name):
   return os.path.join('models', 'best', name, 'best_model.zip')

def create_video(name):
   assert ROWS * COLS == NUM_ENVS
   VIDEOPATH = os.path.join(VIDEOFOLDERNAME, name, name + '.mp4')
   JSONPATH = os.path.join(VIDEOFOLDERNAME, name, name + '.json')

   model = PPO.load(checkpoint_path(name), device='cpu')
   venv = make_vec(n_envs=NUM_ENVS)
   try:
      obs = venv.reset()
      log = []
      os.makedirs(os.path.join(VIDEOFOLDERNAME, name), exist_ok=True)
      battle_steps = [0 for _ in range(NUM_ENVS)]
      battle_starts = [0 for _ in range(NUM_ENVS)]
      was_in_battle = [False for _ in range(NUM_ENVS)]
      actions_in_battle = [{action_name: 0 for action_name in PokemonYellowEnv.ACTIONS} for _ in range(NUM_ENVS)]

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

def watch_live_game(name):
   model = PPO.load(checkpoint_path(name), device='cpu')
   venv = make_vec(n_envs=1, env_kwargs={'max_steps': None})
   try:
      obs = venv.reset()

      while (True):
         actions, _ = model.predict(obs)
         obs, rewards, dones, infos = venv.step(actions)

         frame = venv.get_images()[0]
         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
         frame = cv2.resize(frame, None, fx=LIVE_SCALE, fy=LIVE_SCALE, interpolation=cv2.INTER_NEAREST)
         cv2.imshow('Pokemon Yellow - ' + name, frame)

         if cv2.waitKey(LIVE_DELAY_MS) & 0xFF == ord('q'):
            break

   finally:
      venv.close()
      cv2.destroyAllWindows()


if __name__ == '__main__':
   args = parse_args()
   if args.live:
      watch_live_game(args.name)
   else:
      create_video(args.name)