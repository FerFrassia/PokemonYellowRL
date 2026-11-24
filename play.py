from pathlib import Path
import gzip
import numpy as np
import pyglet
from pyglet.window import key
import stable_retro as retro

ROOT = Path(__file__).parent
retro.data.add_custom_integration(str(ROOT / "integrations"))
env = retro.make("PokemonYellow-GameBoy", state=retro.State.NONE,
                 inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode="rgb_array")
obs, info = env.reset()
buttons = env.unwrapped.buttons
print("buttons:", buttons)

MAP = {"UP": key.UP, "DOWN": key.DOWN, "LEFT": key.LEFT, "RIGHT": key.RIGHT,
       "A": key.X, "B": key.Z, "START": key.ENTER, "SELECT": key.BACKSPACE}
SCALE = 4
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
        path = ROOT / "integrations" / "PokemonYellow-GameBoy" / "start.state"
        with gzip.open(path, "wb") as f:
            f.write(env.unwrapped.em.get_state())
        print("saved", path)
    elif sym == key.I:
        print(state["info"])
    elif sym == key.ESCAPE:
        window.close()

@window.event
def on_key_release(sym, mods):
    pressed.discard(sym)

pyglet.clock.schedule_interval(tick, 1 / 60)
pyglet.app.run()
env.close()
