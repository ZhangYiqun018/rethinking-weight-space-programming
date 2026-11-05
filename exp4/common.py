"""Shared pieces for exp4: Kinetix env for l/h2_car_ramp, actor forward pass, vectorised rollouts."""
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from kinetix.environment import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util.saving import load_from_json_file

UP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream", "lm_understand-logs", "exp4_fix_car", "networks") + os.sep
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
T = 256
GROUPS = [3, 3, 3, 3, 2, 2]  # action entries: 4 motor bindings (3 values), 2 thruster bindings (2 values)
BODY, WHEEL_G, WHEEL_O = 5, 0, 1  # polygon idx of car body; circle idx of green wheel (shape 12) / other wheel (13)


def make_env(dense_reward_scale=0.2):
    state, static, env_params = load_from_json_file("l/h2_car_ramp")
    env_params = env_params.replace(dense_reward_scale=dense_reward_scale, max_timesteps=T)
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None,
                           env_params=env_params, static_env_params=static, auto_reset=False)
    return env, state, env_params, static


def load_actor(which):
    d = np.load(UP + ("ppo_final_seed0.npz" if which == "base" else "edited_actor.npz"), allow_pickle=False)
    if which == "base":
        p = "params/GeneralActorCriticRNN_0/"
        L = [(d["params/Dense_0/kernel"], d["params/Dense_0/bias"])]
        for i in range(5):
            q = p + f"DenseAndActivation_{i}/Dense_0/"
            L.append((d[q + "kernel"], d[q + "bias"]))
        L.append((d[p + "Dense_0/kernel"], d[p + "Dense_0/bias"]))
    else:
        L = [(d["embed.W"], d["embed.b"])] + [(d[f"actor{i}.W"], d[f"actor{i}.b"]) for i in range(5)] + \
            [(d["actor_head.W"], d["actor_head.b"])]
    return [(jnp.asarray(W, jnp.float32), jnp.asarray(b, jnp.float32)) for W, b in L]


def logits_fn(params, obs):
    x = jnp.clip(jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)
    h = jax.nn.relu(x @ params[0][0] + params[0][1])
    for W, b in params[1:6]:
        h = jnp.tanh(h @ W + b)
    return h @ params[6][0] + params[6][1]


def sample_action(logits, key):
    outs, o = [], 0
    for i, g in enumerate(GROUPS):
        outs.append(jax.random.categorical(jax.random.fold_in(key, i), logits[o:o + g]))
        o += g
    return jnp.stack(outs).astype(jnp.int32)


def argmax_action(logits):
    outs, o = [], 0
    for g in GROUPS:
        outs.append(jnp.argmax(logits[o:o + g])); o += g
    return jnp.stack(outs).astype(jnp.int32)


def clean_obs(obs):
    return jnp.clip(jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)


def make_rollout(env, env_params, policy):
    """policy(key, obs, t, aux) -> (action[6], logits[16]). Returns jitted fn(keys, states, aux) over a batch.
    Outputs per episode: outcome code (1 success, -1 red, 0 timeout/invalid), length, actions[T,6], obs[T,679]."""

    def one(key, state0, aux):
        obs0 = env.get_obs(state0)

        def body(c, t):
            key, state, obs, done, outcome, length = c
            key, k1, k2 = jax.random.split(key, 3)
            a, _ = policy(k1, obs, t, aux)
            nobs, nstate, r, d, info = env.step_env(k2, state, a, env_params)
            nobs = clean_obs(nobs)
            live = ~done
            succ = info["GoalR"] & live
            red = (r < -0.5) & live & ~succ
            outcome = jnp.where(live, jnp.where(succ, 1, jnp.where(red, -1, 0)), outcome)
            length = length + live.astype(jnp.int32)
            rec = (a, obs)
            # freeze after done
            state = jax.tree.map(lambda n, o: jnp.where(done, o, n), nstate, state)
            obs = jnp.where(done, obs, nobs)
            done = done | d
            return (key, state, obs, done, outcome, length), rec

        c0 = (key, state0, clean_obs(obs0), jnp.array(False), jnp.array(0), jnp.array(0))
        (_, _, _, _, outcome, length), (acts, obss) = jax.lax.scan(body, c0, jnp.arange(T))
        return outcome, length, acts, obss

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def actor_policy(params):
    """Sampled-from-softmax policy. aux = [679] vector of absolute observation-noise sigma (zeros for none)."""
    def pol(key, obs, t, aux):
        k = jax.random.fold_in(key, 7)
        obs = obs + aux * jax.random.normal(k, obs.shape)
        lg = logits_fn(params, obs)
        return sample_action(lg, key), lg
    return pol


def openloop_policy(n_rev, rev_val, drive_val):
    def pol(key, obs, t, aux):
        v = jnp.where(t < n_rev, rev_val, drive_val)
        return jnp.array([0, 0, 0, 0, 0, 0], jnp.int32).at[0].set(v), jnp.zeros(16)
    return pol


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0, c - h)), float(min(1, c + h))]


def summarize(outcome, length):
    o = np.asarray(outcome); n = len(o)
    k = int((o == 1).sum())
    return {"n": n, "success": k, "rate": k / n, "wilson95": wilson(k, n),
            "red": int((o == -1).sum()), "timeout_or_other": int((o == 0).sum()),
            "mean_len": float(np.mean(length))}


def shift_x(state, d):
    """Shift the whole car (body + both wheels) by d in x."""
    return state.replace(
        polygon=state.polygon.replace(position=state.polygon.position.at[BODY, 0].add(d)),
        circle=state.circle.replace(position=state.circle.position.at[:2, 0].add(d)))


def scale_wheel_friction(state, s):
    return state.replace(circle=state.circle.replace(friction=state.circle.friction.at[:2].multiply(s)))


def scale_car_mass(state, s):
    """Mass x s for body and both wheels (inverse mass and inverse inertia divided by s)."""
    return state.replace(
        polygon=state.polygon.replace(
            inverse_mass=state.polygon.inverse_mass.at[BODY].divide(s),
            inverse_inertia=state.polygon.inverse_inertia.at[BODY].divide(s)),
        circle=state.circle.replace(
            inverse_mass=state.circle.inverse_mass.at[:2].divide(s),
            inverse_inertia=state.circle.inverse_inertia.at[:2].divide(s)))


def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1)
