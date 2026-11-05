"""PPO baseline for Exp5 (REPRO_PLAN.md Phase 2c, H7), trained in JAX on pgx's MinAtar, scored on official MinAtar.

The actor is the same 600 -> 32 -> 32 -> 6 tanh network as in Exp5 and reads the same 6-channel input
(eval_minatar.CHANNELS). The critic is separate (256-256 ReLU); only the actor is compared. pgx's MinAtar keeps
sticky actions (0.1); we use its full 6-action set so the logits mean the same actions as in official MinAtar.
Final scores always come from eval_minatar.py on the official 1.0.15 games, never from pgx.

    python ppo_pgx.py --game breakout --steps 100_000_000 --lr 5e-4 --num-envs 64 --seed 0 --out runs/x
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_minatar import CHANNELS  # noqa: E402

# official MinAtar channel order per game (Env.channels in minatar 1.0.15); pgx keeps the same order
OFFICIAL = {
    "breakout": ["paddle", "ball", "trail", "brick"],
    "seaquest": ["sub_front", "sub_back", "friendly_bullet", "trail", "enemy_bullet", "enemy_fish", "enemy_sub",
                 "oxygen_guage", "diver_guage", "diver"],
    "asterix": ["player", "enemy", "trail", "gold"],
}


def channel_matrix(game):
    """(C_official, 6) 0/1 matrix: our channel c is the max over the official channels mapped to it."""
    m = np.zeros((len(OFFICIAL[game]), 6), dtype=np.float32)
    for ours, theirs in CHANNELS[game].items():
        for name in theirs:
            m[OFFICIAL[game].index(name), ours] = 1.0
    return jnp.asarray(m)


def to_input(obs, cmat):
    x = jnp.max(obs.astype(jnp.float32)[..., :, None] * cmat, axis=-2)  # (..., 10, 10, 6)
    return x.reshape(*x.shape[:-3], 600)


def init_mlp(key, sizes, scale_last):
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        gain = scale_last if i == len(sizes) - 2 else np.sqrt(2)
        w = jax.nn.initializers.orthogonal(gain)(k, (a, b))
        params.append((w, jnp.zeros(b)))
    return params


def actor(params, x):
    (w1, b1), (w2, b2), (w3, b3) = params
    return jnp.tanh(jnp.tanh(x @ w1 + b1) @ w2 + b2) @ w3 + b3


def critic(params, x):
    (w1, b1), (w2, b2), (w3, b3) = params
    return (jax.nn.relu(jax.nn.relu(x @ w1 + b1) @ w2 + b2) @ w3 + b3)[..., 0]


def make_train(a):
    cls = {"breakout": "MinAtarBreakout", "seaquest": "MinAtarSeaquest", "asterix": "MinAtarAsterix"}[a.game]
    env = getattr(importlib.import_module(f"pgx_minatar.{a.game}"), cls)(use_minimal_action_set=False,
                                                                         sticky_action_prob=0.1)
    cmat = channel_matrix(a.game)
    n_updates = a.steps // (a.num_envs * a.num_steps)
    mb_size = a.num_envs * a.num_steps // a.num_minibatches
    sched = optax.linear_schedule(a.lr, 0.0, n_updates * a.update_epochs * a.num_minibatches)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched if a.anneal else a.lr, eps=1e-5))
    step_fn = jax.vmap(env.step)
    init_fn = jax.vmap(env.init)

    def train(key):
        key, ka, kc, ke = jax.random.split(key, 4)
        params = {"actor": init_mlp(ka, [600, 32, 32, 6], 0.01), "critic": init_mlp(kc, [600, 256, 256, 1], 1.0)}
        opt = tx.init(params)
        state = init_fn(jax.random.split(ke, a.num_envs))
        ep_ret = jnp.zeros(a.num_envs)

        def env_step(carry, _):
            params, state, ep_ret, key = carry
            key, k_act, k_step, k_reset = jax.random.split(key, 4)
            x = to_input(state.observation, cmat)
            logits = actor(params["actor"], x)
            action = jax.random.categorical(k_act, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(a.num_envs), action]
            value = critic(params["critic"], x)
            nstate = step_fn(state, action, jax.random.split(k_step, a.num_envs))
            reward = nstate.rewards[:, 0]
            done = nstate.terminated | nstate.truncated
            ep_ret = ep_ret + reward
            finished = jnp.where(done, ep_ret, jnp.nan)
            reset = init_fn(jax.random.split(k_reset, a.num_envs))
            nstate = jax.tree_util.tree_map(lambda r, s: jnp.where(done.reshape(-1, *([1] * (s.ndim - 1))), r, s),
                                            reset, nstate)
            ep_ret = jnp.where(done, 0.0, ep_ret)
            return (params, nstate, ep_ret, key), (x, action, logp, value, reward, done, finished)

        def update(carry, _):
            params, opt, state, ep_ret, key = carry
            (params, state, ep_ret, key), traj = jax.lax.scan(env_step, (params, state, ep_ret, key), None,
                                                              a.num_steps)
            x, action, logp, value, reward, done, finished = traj
            last_v = critic(params["critic"], to_input(state.observation, cmat))

            def gae(carry, t):
                adv, next_v = carry
                r, d, v = t
                delta = r + a.gamma * next_v * (1 - d) - v
                adv = delta + a.gamma * a.gae_lambda * (1 - d) * adv
                return (adv, v), adv
            _, advs = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), (reward, done.astype(jnp.float32), value),
                                   reverse=True)
            targets = advs + value
            batch = jax.tree_util.tree_map(lambda z: z.reshape(-1, *z.shape[2:]), (x, action, logp, advs, targets))

            def loss_fn(p, mb):
                bx, ba, blogp, badv, btgt = mb
                logits = actor(p["actor"], bx)
                lp_all = jax.nn.log_softmax(logits)
                lp = lp_all[jnp.arange(ba.shape[0]), ba]
                ratio = jnp.exp(lp - blogp)
                nadv = (badv - badv.mean()) / (badv.std() + 1e-8)
                pg = -jnp.minimum(ratio * nadv, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * nadv).mean()
                vl = 0.5 * ((critic(p["critic"], bx) - btgt) ** 2).mean()
                ent = -(jnp.exp(lp_all) * lp_all).sum(-1).mean()
                return pg + a.vf_coef * vl - a.ent_coef * ent

            def epoch(carry, _):
                params, opt, key = carry
                key, k = jax.random.split(key)
                perm = jax.random.permutation(k, a.num_envs * a.num_steps)
                mbs = jax.tree_util.tree_map(lambda z: z[perm].reshape(a.num_minibatches, mb_size, *z.shape[1:]), batch)

                def mb_step(carry, mb):
                    params, opt = carry
                    grads = jax.grad(loss_fn)(params, mb)
                    upd, opt = tx.update(grads, opt, params)
                    return (optax.apply_updates(params, upd), opt), None
                (params, opt), _ = jax.lax.scan(mb_step, (params, opt), mbs)
                return (params, opt, key), None
            (params, opt, key), _ = jax.lax.scan(epoch, (params, opt, key), None, a.update_epochs)
            return (params, opt, state, ep_ret, key), jnp.nanmean(finished)

        (params, *_), returns = jax.lax.scan(update, (params, opt, state, ep_ret, key), None, n_updates)
        return params, returns
    return train, n_updates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=sorted(OFFICIAL))
    ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--anneal", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=128)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--num-minibatches", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train, n_updates = make_train(a)
    t0 = time.time()
    params, returns = jax.jit(train)(jax.random.PRNGKey(a.seed))
    returns = np.asarray(returns)
    wall = time.time() - t0
    (w1, b1), (w2, b2), (w3, b3) = params["actor"]
    np.savez(os.path.join(a.out, "actor.npz"), **{
        "actor/dense1/kernel": np.asarray(w1), "actor/dense1/bias": np.asarray(b1),
        "actor/dense2/kernel": np.asarray(w2), "actor/dense2/bias": np.asarray(b2),
        "actor/logits/kernel": np.asarray(w3), "actor/logits/bias": np.asarray(b3)})
    json.dump({"args": vars(a), "updates": n_updates, "wall_s": wall,
               "train_return_last10pct": float(np.nanmean(returns[-max(1, len(returns) // 10):])),
               "train_returns": [None if np.isnan(r) else float(r) for r in returns[:: max(1, len(returns) // 200)]]},
              open(os.path.join(a.out, "train.json"), "w"), indent=1)
    print(f"{a.game} lr={a.lr} envs={a.num_envs} seed={a.seed}: train return (last 10%) "
          f"{np.nanmean(returns[-max(1, len(returns) // 10):]):.2f}, {wall:.0f}s")


if __name__ == "__main__":
    main()
