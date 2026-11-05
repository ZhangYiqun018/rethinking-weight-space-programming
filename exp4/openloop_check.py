import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from common import make_env, make_rollout, openloop_policy, actor_policy, load_actor
env, state, ep, _ = make_env()
b = lambda s, n: jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s)
keys = jax.random.split(jax.random.PRNGKey(123), 8)
for name, pol in [("openloop rev16", openloop_policy(16, 2, 1)), ("openloop rev0", openloop_policy(0, 2, 1)),
                  ("drive-only right", openloop_policy(0, 1, 2)), ("edited", actor_policy(load_actor("edited"))),
                  ("base", actor_policy(load_actor("base")))]:
    oc, ln, acts, _ = make_rollout(env, ep, pol)(keys, b(state, 8), jnp.zeros(679))
    print(f"{name:18s} outcome={np.asarray(oc).tolist()} len={np.asarray(ln).tolist()}")
