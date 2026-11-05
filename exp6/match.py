"""Independent temperature-one match runner; never imports upstream code."""
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from pgx._src.games.go import Game

ROOT = Path(__file__).resolve().parent
NETWORKS = ROOT.parent / "upstream/lm_understand-logs/exp6_go_from_scratch/networks"
GAME = Game(size=9, komi=7.5, history_length=1, max_termination_steps=162)


def network(name):
    directory = NETWORKS / name
    weights = []
    for layer in [f"L{i}" for i in range(1, 7)] + ["out"]:
        w, b = [np.load(directory / f"{layer}.{suffix}.npy", allow_pickle=False)
                for suffix in ("W", "b")]
        assert w.shape == ((243 if layer == "L1" else 512),
                           82 if layer == "out" else 512)
        assert b.shape == (w.shape[1],)
        assert np.isfinite(w).all() and np.isfinite(b).all()
        weights.append((jnp.asarray(w), jnp.asarray(b)))

    def policy(observation, legal):
        del legal
        h = observation.reshape((-1, 243)).astype(jnp.float32)
        for w, b in weights[:-1]:
            h = jax.nn.relu(h @ w + b)
        return h @ weights[-1][0] + weights[-1][1]
    return policy


def random_policy(observation, legal):
    del observation
    return jnp.zeros(legal.shape, dtype=jnp.float32)


def wilson(wins, n):
    z = 1.959963984540054
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - half), float(center + half)]


def play(player_a, player_b, games=2000, seed=31337, batch_size=250):
    assert games % batch_size == 0 and batch_size % 2 == 0
    a_black = jnp.arange(batch_size) < batch_size // 2

    @jax.jit
    def batch(key):
        states = jax.vmap(lambda _: GAME.init())(jnp.arange(batch_size))
        done = jnp.zeros(batch_size, bool)

        def step(carry):
            states, done, key = carry
            legal = jax.vmap(GAME.legal_action_mask)(states)
            own_sign = jnp.where(states.color == 0, 1, -1)
            board = states.board * own_sign[:, None]
            obs = jnp.stack((board > 0, board < 0,
                             jnp.broadcast_to(states.color[:, None] == 1, board.shape)), -1)
            a_turn = (states.color == 0) == a_black
            logits = jnp.where(a_turn[:, None], player_a(obs, legal), player_b(obs, legal))
            key, action_key = jax.random.split(key)
            actions = jax.random.categorical(action_key, jnp.where(legal, logits, -jnp.inf))
            updated = jax.vmap(GAME.step)(states, actions)
            states = jax.tree.map(
                lambda old, new: jnp.where(done.reshape((batch_size,) + (1,) * (old.ndim - 1)), old, new),
                states, updated)
            return states, done | jax.vmap(GAME.is_terminal)(states), key

        states, _, _ = jax.lax.while_loop(lambda c: ~c[1].all(), step, (states, done, key))
        reward = jax.vmap(GAME.rewards)(states)
        won = jnp.where(a_black, reward[:, 0], reward[:, 1]) > 0
        return won, states.step_count, states.consecutive_pass_count, states.is_psk

    started = time.monotonic()
    records = []
    for i in range(games // batch_size):
        output = jax.device_get(batch(jax.random.fold_in(jax.random.PRNGKey(seed), i)))
        won, lengths, passes, repetitions = output
        records.extend({"a_black": k < batch_size // 2, "a_won": bool(won[k]),
                        "length": int(lengths[k]), "repetition": bool(repetitions[k]),
                        "two_passes": bool(passes[k] >= 2)}
                       for k in range(batch_size))
        print(f"  {len(records)}/{games}, {time.monotonic()-started:.1f}s", flush=True)
    wins = sum(r["a_won"] for r in records)
    return dict(games=games, seed=seed, batch_size=batch_size, wins=wins,
                score=wins / games, wilson95=wilson(wins, games),
                score_as_black=sum(r["a_won"] and r["a_black"] for r in records) / (games / 2),
                score_as_white=sum(r["a_won"] and not r["a_black"] for r in records) / (games / 2),
                mean_length=float(np.mean([r["length"] for r in records])),
                seconds=time.monotonic() - started, records=records)
