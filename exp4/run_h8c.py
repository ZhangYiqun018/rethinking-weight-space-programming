"""Fine wheel-friction sweep (16 episodes/point) to locate where each policy starts/stops succeeding."""
import numpy as np, jax, jax.numpy as jnp
from common import *
env, state, ep, st = make_env()
n = 16
def batch(s): return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s)
rolls = {"base": make_rollout(env, ep, actor_policy(load_actor("base"))), "edited": make_rollout(env, ep, actor_policy(load_actor("edited"))),
         "openloop": make_rollout(env, ep, openloop_policy(16, 2, 1))}
R = {}
for k in [0.5, 0.6, 0.7, 0.75, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3, 3.5, 4]:
    s = scale_wheel_friction(state, k)
    R[str(k)] = {}
    for nm, r in rolls.items():
        oc, ln, _, _ = r(jax.random.split(jax.random.PRNGKey(0), n), batch(s), jnp.zeros(679))
        R[str(k)][nm] = summarize(oc, ln)
    print(k, {nm: v["success"] for nm, v in R[str(k)].items()}, flush=True)
save("h8c_friction_sweep.json", R)
