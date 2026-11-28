<h1 align="center">PokemonYellowRL</h1>
<p align="center">Agent that plays Pokémon Yellow for the Game Boy Color using PPO</p>

<p align="center">
  <img src="assets/demo.gif" alt="The trained agent playing" width="480">
</p>

## Installation

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## ROM and start state

The ROM and the save states are not included in the repository. Put your own legally obtained copy of Pokémon Yellow in `integrations/PokemonYellow-GameBoy/` as `rom.gb`.

Then create the start state the agent trains from: play the intro yourself, press S after getting Pikachu, and save it as `start`.

```bash
python play.py --save-as start
```

## Play the game yourself

```bash
python play.py --state start --save-as <save_name>
```

Both flags are optional. `--state` loads a saved state (without it the game starts from the beginning), and `--save-as` sets the name used when you press S to save the current state. States are saved in `integrations/PokemonYellow-GameBoy/`.

Controls: arrow keys, X (A), Z (B), Enter (Start), S (save state), I (print the game's RAM variables), Esc (quit).

## Train your own agent

```bash
python train.py --name <version_name>
```

Checkpoints are saved to `models/checkpoints/` and the best model to `models/best/<version_name>/`. Follow the training curves with `tensorboard --logdir runs`.

## Watch the trained agent play

```bash
python watch.py --name <version_name>
```

Plays 24 episodes of the best model at once and writes a grid video and a JSON with per-episode stats to `videos/<version_name>/`.