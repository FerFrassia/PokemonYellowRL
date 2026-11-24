from pathlib import Path
import numpy as np
import stable_retro as retro
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import cv2

retro.data.add_custom_integration(str(Path(__file__).parent / "integrations"))


def make_env():
	env = retro.make("PokemonYellow-GameBoy",
	                 inttype=retro.data.Integrations.CUSTOM_ONLY,
	                 render_mode="rgb_array")
	return ScreenPreprocess(PokemonYellowEnv(env))

class Movement:
	frame = 0
	x = 0
	y = 0
	direction = 'DOWN'
	deltaFrame = 0
	deltaX = 0
	deltaY = 0

	def __init__(self, frame, x, y, direction, deltaFrame, deltaX, deltaY):
		self.frame = frame
		self.x = x
		self.y = y
		self.direction = direction
		self.deltaFrame = deltaFrame
		self.deltaX = deltaX
		self.deltaY = deltaY

	def __str__(self):
		return f'[{self.direction}]: deltaFrame={self.deltaFrame}, deltaX={self.deltaX}, deltaY={self.deltaY}, frame={self.frame}, x={self.x}, y={self.y}'


def experimentWithMovements():
	env.reset()
	noop = np.zeros(9, dtype=np.int8)
	_, _, _, _, info = env.step(noop)

	prevFrame = 0
	prevX = info['player_x']
	prevY = info['player_y']

	movements = []
	directions = ['LEFT', 'LEFT', 'LEFT', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'LEFT']
	directionIndex = 0
	for currentFrame in range(1, 360):
		obs, reward, terminated, truncated, info = env.step(press(directions[directionIndex]))
		if info['player_x'] != prevX or info['player_y'] != prevY:  # there was a movement

			# save movement
			currentMovement = Movement(frame=currentFrame, x=info['player_x'], y=info['player_y'],
			                           direction=directions[directionIndex], deltaFrame=currentFrame - prevFrame,
			                           deltaX=info['player_x'] - prevX, deltaY=info['player_y'] - prevY)
			movements.append(currentMovement)

			# update values
			prevFrame = currentFrame
			prevX = info['player_x']
			prevY = info['player_y']
			directionIndex = (directionIndex + 1) % len(directions)

		# #no-ops to wait while the UI is moving the player
		# for _ in range(30):
		# 	noop = np.zeros(9, dtype=np.int8)
		# 	_, _, _, _, info = env.step(noop)

	print('stuck on direction:', directions[directionIndex])
	print('final position:', info['player_x'], info['player_y'])

	for movement in movements:
		print(movement)

class PokemonYellowEnv(gym.Wrapper):
	ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START']
	MAX_STEPS = 50

	def __init__(self, env, hold=2, release=16):
		super().__init__(env)

		self.hold = hold
		self.release = release
		self.buttons = env.unwrapped.buttons
		self.visited_tiles = {}
		self.step_counter = 0
		self.action_space = gym.spaces.Discrete(len(self.ACTIONS))

	def _press(self, name):
		a = np.zeros(9, dtype=np.int8)
		a[self.buttons.index(name)] = 1
		return a

	def _visit(self, info):
		"""Remember the current tile. Returns True if it was new."""
		tiles = self.visited_tiles.setdefault(info['current_map'], set())
		pos = (info['player_x'], info['player_y'])
		is_new = pos not in tiles
		tiles.add(pos)
		return is_new

	def _do_action(self, name):
		for _ in range(self.hold):
			self.env.step(self._press(name))
		for _ in range(self.release):
			obs, _, terminated, _, info = self.env.step(np.zeros(9, dtype=np.int8)) #no-ops
		return obs, info, terminated

	def reset(self, **kwargs):
		self.env.reset(**kwargs)
		self.visited_tiles = {}
		self.step_counter = 0
		obs, _, _, _, info = self.env.step(np.zeros(9, dtype=np.int8))  # no-op to fill info
		self._visit(info) #the start tile isn't a discovery
		info['amount_of_visited_tiles'] = 1
		return obs, info

	def step(self, action):
		obs, info, terminated = self._do_action(self.ACTIONS[action])
		if self._visit(info):
			reward = 1.0
		else:
			reward = 0.0

		self.step_counter += 1
		info['amount_of_visited_tiles'] = sum(len(s) for s in self.visited_tiles.values())

		truncated = self.step_counter >= self.MAX_STEPS

		return obs, reward, terminated, truncated, info

class ScreenPreprocess(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		self.observation_space = gym.spaces.Box(0, 255, shape=(72, 80, 1), dtype=np.uint8)

	def observation(self, obs):
		resized = cv2.resize(obs,  # original image
		                    (80, 72),
		                    interpolation=cv2.INTER_AREA)
		gray_resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
		return gray_resized[..., None]

if __name__ == '__main__':
	check_env(make_env())

	env = make_env()
	obs, info = env.reset()
	total_reward = 0.0
	for i in range(1000):
		obs, r, term, trunc, info = env.step(env.action_space.sample())
		total_reward += r
		if trunc:
			print('truncated at global step', i + 1, '| tiles this episode:', info['amount_of_visited_tiles'], '| reward:', total_reward)
			obs, info = env.reset()
			total_reward = 0

