from pathlib import Path
import stable_retro as retro

retro.data.add_custom_integration(str(Path(__file__).parent / "integrations"))
env = retro.make("PokemonYellow-GameBoy",
                 inttype=retro.data.Integrations.CUSTOM_ONLY,
                 render_mode="rgb_array")   # uses the default "start" state
obs, info = env.reset()
print("after reset:", info)

noop = env.action_space.sample() * 0
for _ in range(120):
    obs, *_ , info = env.step(noop)
print("after 120 idle frames:", info)
env.close()
