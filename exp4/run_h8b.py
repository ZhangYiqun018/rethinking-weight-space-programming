"""Supplement: uniform mass scaling is a no-op in this engine (see RESULTS.md). Body-only mass, and wider friction/mass ranges."""
import numpy as np, jax, jax.numpy as jnp
from common import *
env, state, ep, st = make_env()
N = 256
def batch(s, n): return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s)
B, E = load_actor("base"), load_actor("edited")
rolls = {"base": make_rollout(env, ep, actor_policy(B)), "edited": make_rollout(env, ep, actor_policy(E)),
         "openloop": make_rollout(env, ep, openloop_policy(16, 2, 1))}
Z = jnp.zeros(679)
def run(name, s): 
    oc, ln, _, _ = rolls[name](jax.random.split(jax.random.PRNGKey(0), N), batch(s, N), Z); return summarize(oc, ln)
def body_mass(s, k):
    return s.replace(polygon=s.polygon.replace(inverse_mass=s.polygon.inverse_mass.at[BODY].divide(k),
                                               inverse_inertia=s.polygon.inverse_inertia.at[BODY].divide(k)))
R = {"body_mass_x": {}, "wheel_friction_wide_x": {}, "car_mass_uniform_wide_x": {}}
for k in [0.5, 0.8, 0.9, 1.1, 1.2, 2, 4]:
    R["body_mass_x"][str(k)] = {n: run(n, body_mass(state, k)) for n in rolls}; print("bodymass", k, {n: v["success"] for n, v in R["body_mass_x"][str(k)].items()}, flush=True)
for k in [0.25, 0.5, 2, 4]:
    R["wheel_friction_wide_x"][str(k)] = {n: run(n, scale_wheel_friction(state, k)) for n in rolls}; print("fric", k, {n: v["success"] for n, v in R["wheel_friction_wide_x"][str(k)].items()}, flush=True)
for k in [0.5, 2, 4]:
    R["car_mass_uniform_wide_x"][str(k)] = {n: run(n, scale_car_mass(state, k)) for n in rolls}; print("mass", k, {n: v["success"] for n, v in R["car_mass_uniform_wide_x"][str(k)].items()}, flush=True)
save("h8b_supplement.json", R)
