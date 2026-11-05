"""H1 reproduction + determinism check."""
import hashlib, time
import numpy as np, jax, jax.numpy as jnp
from common import *

env, state, ep, st = make_env()
out = {"config": {"level": "l/h2_car_ramp", "obs": "SYMBOLIC_FLAT", "action": "MULTI_DISCRETE", "env_size": "l",
                  "frame_skip": st.frame_skip, "max_timesteps": T, "dense_reward_scale": 0.2,
                  "num_polygons/circles/joints/thrusters": [st.num_polygons, st.num_circles, st.num_joints, st.num_thrusters]}}
def batch(n): return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), state)

for name, n, seed in [("base", 256, 1), ("edited", 1024, 2)]:
    P = load_actor(name)
    roll = make_rollout(env, ep, actor_policy(P))
    t0 = time.time()
    outc, ln, acts, obss = roll(jax.random.split(jax.random.PRNGKey(seed), n), batch(n), jnp.zeros(679))
    outc, ln, acts, obss = map(np.asarray, (outc, ln, acts, obss))
    r = summarize(outc, ln); r["seconds"] = time.time() - t0
    # determinism: hash action sequences (actions up to episode length), and of the action entry 0 only
    def h(i, cols=slice(None)): return hashlib.sha256(acts[i, :ln[i], cols].astype(np.int8).tobytes()).hexdigest()
    r["distinct_action_sequences_all6"] = len({h(i) for i in range(n)})
    r["distinct_state_trajectories_obs_hash"] = len({hashlib.sha256(obss[i, :ln[i]].tobytes()).hexdigest() for i in range(n)})
    r["distinct_action_sequences_entry0"] = len({h(i, slice(0, 1)) for i in range(n)})
    # initial obs identical across episodes?
    r["initial_obs_identical_across_episodes"] = bool((obss[:, 0] == obss[0, 0]).all())
    r["episode_lengths_distinct"] = sorted(set(map(int, ln)))[:20]
    # first step at which the sequences diverge from the modal sequence (entry 0)
    from collections import Counter
    c = Counter(h(i, slice(0, 1)) for i in range(n)); r["top_entry0_sequence_counts"] = c.most_common(5) and [v for _, v in c.most_common(5)]
    out[name] = r
    print(name, r)

# reset-state check using different keys
o1, s1 = env.reset(jax.random.PRNGKey(0), ep, override_reset_state=state)
o2, s2 = env.reset(jax.random.PRNGKey(99), ep, override_reset_state=state)
out["reset_identical_for_different_keys"] = bool(all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(s1), jax.tree.leaves(s2))))
print(out["reset_identical_for_different_keys"])
# also: argmax episode for both
for name in ["base", "edited"]:
    P = load_actor(name)
    roll = make_rollout(env, ep, lambda k, o, t, aux, P=P: (argmax_action(logits_fn(P, o)), None))
    oc, l, *_ = roll(jax.random.split(jax.random.PRNGKey(0), 1), batch(1), jnp.zeros(679))
    out[name + "_argmax_outcome"] = [int(oc[0]), int(l[0])]
save("h1_determinism.json", out)
