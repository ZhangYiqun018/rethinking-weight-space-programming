"""Measure blinded-study ground truth. Never place these outputs in subject work."""
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("readwt_common", Path(__file__).with_name("common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
M, R = C.engine()
import jax
import jax.numpy as jnp
import numpy as np

OUT = C.ROOT / "truth"
ADJ = R.ADJ.astype(jnp.float32)
EYE = jnp.eye(81, dtype=bool)


def move_features(board, legal):
    """Return legal capture/rescue/self-atari/eye features for all 82 actions."""
    members, neighbors, libs = R.groups(board)
    own = (members & (board[None, :] == 1)).any(axis=1)
    opp = (members & (board[None, :] == -1)).any(axis=1)
    touched = neighbors.T
    killed = touched & opp[None, :] & (libs[None, :] == 1)
    removed = (killed.astype(jnp.float32) @ members.astype(jnp.float32)) > 0
    captured = removed.any(axis=1)
    joined = touched & own[None, :]
    merged = (joined.astype(jnp.float32) @ members.astype(jnp.float32) > 0) | EYE
    empty_after = ((board[None, :] == 0) | removed) & ~EYE
    merged_neighbors = merged.astype(jnp.float32) @ ADJ > 0
    new_libs = (merged_neighbors & empty_after).sum(axis=1)
    # A capture can save an atari group without joining it: include that case.
    unmerged_libs = empty_after.astype(jnp.float32) @ neighbors.T.astype(jnp.float32)
    after_libs = jnp.where(joined, new_libs[:, None], unmerged_libs)
    escape = (own[None, :] & (libs[None, :] == 1) & (after_libs >= 2)).any(axis=1)
    self_atari = (new_libs == 1) & ~captured
    eye = R.eye_mask(board) & (legal[:81].sum() > 1)
    return jnp.pad(jnp.stack([captured, escape, self_atari, eye], axis=1),
                   ((0, 1), (0, 0))) & legal[:, None]


def tally(features, action, own_move_number):
    """Counts: moves, capture opportunities/hits, escape opportunities/hits,
    self-atari moves, eye-fill moves, early pass events."""
    selected = features[action]
    return jnp.array([1, features[:, 0].any(), selected[0],
                      features[:, 1].any(), selected[1], selected[2], selected[3],
                      (action == 81) & (own_move_number < 30)], dtype=jnp.int32)


def measure(policy, seed, games=1000, batch_size=100):
    a_black = jnp.arange(batch_size) < batch_size // 2

    @jax.jit
    def batch(key):
        states = jax.vmap(lambda _: M.GAME.init())(jnp.arange(batch_size))
        done = jnp.zeros(batch_size, bool)
        counts = jnp.zeros((batch_size, 8), jnp.int32)

        def step(carry):
            states, done, counts, key = carry
            legal = jax.vmap(M.GAME.legal_action_mask)(states)
            board = jnp.sign(states.board) * jnp.where(states.color == 0, 1, -1)[:, None]
            obs = jnp.stack((board == 1, board == -1,
                             jnp.broadcast_to(states.color[:, None] == 1, board.shape)), -1)
            a_turn = (states.color == 0) == a_black
            logits = jnp.where(a_turn[:, None], policy(obs, legal), 0.0)
            key, action_key = jax.random.split(key)
            actions = jax.random.categorical(action_key, jnp.where(legal, logits, -jnp.inf))
            features = jax.vmap(move_features)(board, legal)
            added = jax.vmap(tally)(features, actions, counts[:, 0] + 1)
            counts += jnp.where((a_turn & ~done)[:, None], added, 0)
            updated = jax.vmap(M.GAME.step)(states, actions)
            states = jax.tree.map(lambda old, new: jnp.where(
                done.reshape((batch_size,) + (1,) * (old.ndim - 1)), old, new), states, updated)
            return states, done | jax.vmap(M.GAME.is_terminal)(states), counts, key

        states, _, counts, _ = jax.lax.while_loop(lambda c: ~c[1].all(), step,
                                                 (states, done, counts, key))
        return counts, states.step_count

    records = []
    for i in range(games // batch_size):
        counts, lengths = jax.device_get(batch(jax.random.fold_in(jax.random.PRNGKey(seed), i)))
        for k in range(batch_size):
            records.append({"a_black": k < batch_size // 2, "counts": counts[k].tolist(),
                            "length": int(lengths[k])})
        print(f"statistics seed {seed}: {len(records)}/{games}", flush=True)
    return summarize(records, seed, batch_size)


def summarize(records, seed, batch_size):
    games = len(records)
    c = np.array([r["counts"] for r in records]).sum(axis=0)
    return {
        "capture_rate": float(c[2] / c[1]) if c[1] else None,
        "atari_escape_rate": float(c[4] / c[3]) if c[3] else None,
        "self_atari_rate": float(c[5] / c[0]),
        "eye_fill_rate": float(c[6] / c[0]),
        "pass_before_60_rate": float(np.mean([r["counts"][7] > 0 for r in records])),
        "mean_game_length": float(np.mean([r["length"] for r in records])),
        "statistics_provenance": {"games": games, "seed": seed, "batch_size": batch_size,
                                  "count_columns": ["own_moves", "capture_opportunities", "captures",
                                                    "escape_opportunities", "escapes", "self_atari",
                                                    "eye_fills", "early_pass_events"],
                                  "totals": c.tolist(), "records": records}}


def first_move(policy):
    logits = policy(jnp.zeros((1, 81, 3), bool), jnp.ones((1, 82), bool))[0]
    probs = np.asarray(jax.nn.softmax(logits)).astype(float)
    assert np.isclose(probs.sum(), 1, atol=1e-6)
    top = sorted(range(81), key=lambda i: (-probs[i], i))[:3]
    return {"probabilities": probs.tolist(),
            "top3": [{"row": i // 9, "col": i % 9, "probability": float(probs[i])} for i in top],
            "p_tengen": float(probs[40])}


def saved_match(name, a, b, seed):
    path = OUT / f"match_{name}.json"
    if not path.exists():
        path.write_text(json.dumps(M.play(a, b, seed=seed), indent=2) + "\n")
    return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=True)
    policies = {role: M.network(name) for role, name in C.ROLES.items()}
    for index, (role, policy) in enumerate(policies.items()):
        path = OUT / f"{role}.json"
        if path.exists():
            continue
        result = measure(policy, 35000 + index)
        result["first_move"] = first_move(policy)
        result["role"] = role
        result["sources"] = {}
        for opponent in ["random", "R1"]:
            source = (C.EXP6 / "results" /
                      (f"eval_{role}_vs_random.json" if opponent == "random" else
                       f"eval_R1_vs_{'s0' if role == 'ppo' else role}.json"))
            if source.exists():
                match = json.loads(source.read_text())
                value = match["score"] if opponent == "random" else 1 - match["score"]
                provenance = {"path": str(source), "inverted": opponent == "R1",
                              "games": match["games"], "seed": match["seed"]}
            else:
                match = saved_match(f"{role}_vs_{opponent}", policy,
                                    M.random_policy if opponent == "random" else R.r1, 35100 + index)
                value = match["score"]
                provenance = {"path": str(OUT / f"match_{role}_vs_{opponent}.json"),
                              "inverted": False, "games": match["games"], "seed": match["seed"]}
            result[f"winrate_vs_{opponent}"] = value
            result["sources"][f"winrate_vs_{opponent}"] = provenance
        path.write_text(json.dumps(result, indent=2) + "\n")
    pairs = {}
    for i, (a, b) in enumerate([("b0", "b50"), ("b0", "ppo"), ("b50", "ppo")]):
        source = C.EXP6 / "results" / f"gate_{a}_vs_s0.json"
        if b == "ppo" and source.exists():
            match = json.loads(source.read_text())
        else:
            source = OUT / f"match_{a}_vs_{b}.json"
            match = saved_match(f"{a}_vs_{b}", policies[a], policies[b], 35200 + i)
        pairs[f"{a}_vs_{b}"] = {"probability": match["score"], "games": match["games"],
                                "seed": match["seed"], "source": str(source)}
    (OUT / "pairwise.json").write_text(json.dumps(pairs, indent=2) + "\n")


if __name__ == "__main__":
    main()
