import gzip
import stable_retro as retro
from pathlib import Path

ROOT = Path(__file__).parent

def save_state(env, name):
    if name is not None:
        path = ROOT / "integrations" / "PokemonYellow-GameBoy" / (name + ".state")
        with gzip.open(path, "wb") as f:
            f.write(env.unwrapped.em.get_state())
        print("saved", path)
    else:
        print("warning: didn't save because there's no save-as state")
        
def make_play_env(state_name):
    retro.data.add_custom_integration(str(ROOT / "integrations"))
    return retro.make("PokemonYellow-GameBoy", state=state_name or retro.State.NONE,
                 inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode="rgb_array")