import numpy as np
import pyglet
from pyglet.window import key

import argparse
from envs import ROOT, save_state, make_play_env

MAP = {"UP": key.UP, "DOWN": key.DOWN, "LEFT": key.LEFT, "RIGHT": key.RIGHT,
       "A": key.X, "B": key.Z, "START": key.ENTER, "SELECT": key.BACKSPACE}
SCALE = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state')
    parser.add_argument('--save-as')
    return parser.parse_args()

def play(args):
    env = make_play_env(args.state)
    obs, info = env.reset()
    buttons = env.unwrapped.buttons
    print("buttons: ", buttons)
    print(f"loaded: state: {args.state}, save_as: {args.save_as}")
    H, W = obs.shape[:2]
    window = pyglet.window.Window(W * SCALE, H * SCALE, "Pokemon Yellow")
    pressed = set()
    state = {"obs": obs, "info": info}

    def tick(dt):
        action = np.array([1 if (b in MAP and MAP[b] in pressed) else 0 for b in buttons],
                          dtype=np.int8)
        state["obs"], _, term, trunc, state["info"] = env.step(action)

    @window.event
    def on_draw():
        window.clear()
        o = state["obs"]
        img = pyglet.image.ImageData(W, H, "RGB", o.tobytes(), pitch=-W * 3)
        fw, fh = window.get_framebuffer_size()
        img.blit(0, 0, width=fw, height=fh)

    @window.event
    def on_key_press(sym, mods):
        pressed.add(sym)
        if sym == key.S:
            save_state(env, args.save_as)
        elif sym == key.I:
            print(state["info"])
        elif sym == key.ESCAPE:
            window.close()

    @window.event
    def on_key_release(sym, mods):
        pressed.discard(sym)

    pyglet.clock.schedule_interval(tick, 1 / 60)
    try:
        pyglet.app.run()
    finally:
        env.close()

if __name__ == '__main__':
    play(parse_args())