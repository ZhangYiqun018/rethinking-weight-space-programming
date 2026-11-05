"""Check vectorized tactical scores against an independent Python flood fill."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pgx._src.games.go import Game

spec = importlib.util.spec_from_file_location("rules", Path(__file__).with_name("rules.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
adj = [[j for j in range(81) if abs(i // 9 - j // 9) + abs(i % 9 - j % 9) == 1]
       for i in range(81)]


def groups(board):
    unseen = set(np.flatnonzero(board))
    out = []
    while unseen:
        first = min(unseen)
        group, frontier = {first}, [first]
        while frontier:
            i = frontier.pop()
            for j in adj[i]:
                if board[j] == board[first] and j not in group:
                    group.add(j)
                    frontier.append(j)
        unseen -= group
        liberties = {j for i in group for j in adj[i] if board[j] == 0}
        out.append((int(board[first]), group, liberties))
    return out


def score(board, move):
    old = groups(board)
    after = board.copy()
    after[move] = 1
    captures = 0
    rescued = False
    for color, stones, libs in old:
        if color == -1 and libs == {move}:
            after[list(stones)] = 0
            captures += len(stones)
        if color == 1 and libs == {move}:
            rescued = True
    new = groups(after)
    own_libs = next(libs for color, stones, libs in new if move in stones)
    threatened = False
    for color, stones, libs in old:
        if color != -1 or len(libs) < 2 or not (stones & set(adj[move])):
            continue
        for c, s, l in new:
            if c == -1 and s == stones and len(l) == 1:
                threatened = True
    distance = min(move // 9, move % 9, 8 - move // 9, 8 - move % 9)
    return (100 * captures + 10 * (rescued and len(own_libs) >= 2) +
            threatened - 20 * (len(own_libs) == 1 and captures == 0) +
            0.03 * (distance in [2, 3]) +
            0.02 * (any(board[j] == 1 for j in adj[move]) and np.count_nonzero(board) < 40))


def main():
    game = Game(size=9, history_length=1)
    step = jax.jit(game.step)
    legal_fn = jax.jit(game.legal_action_mask)
    scores_fn = jax.jit(R.tactical_scores)
    rng = np.random.default_rng(9182)
    checked = 0
    features = set()
    for _ in range(5):
        state = game.init()
        for turn in range(150):
            legal = np.asarray(legal_fn(state))
            board = np.sign(np.asarray(state.board)) * (1 if turn % 2 == 0 else -1)
            if turn % 7 == 0:
                actual = np.asarray(scores_fn(jnp.asarray(board)))
                for move in np.flatnonzero(legal[:81]):
                    reference = score(board, int(move))
                    np.testing.assert_allclose(actual[move], reference, atol=0.001, rtol=1e-6)
                    checked += 1
                    if reference >= 100:
                        features.add("capture")
                    elif reference >= 10:
                        features.add("rescue")
                    elif reference >= 1:
                        features.add("threat")
                    elif reference < 0:
                        features.add("self_atari")
                obs = np.stack((board == 1, board == -1, np.full(81, turn % 2 == 1)), -1)[None]
                for player in (R.r0, R.r1):
                    logits = np.asarray(player(jnp.asarray(obs), jnp.asarray(legal[None])))[0]
                    assert np.isfinite(logits).any()
                    assert not np.isfinite(logits[~legal]).any()
                    eligible = legal[:81] & ~np.asarray(R.eye_mask(jnp.asarray(board)))
                    assert np.isfinite(logits[81]) == (not eligible.any())
            candidates = np.flatnonzero(legal[:81])
            state = step(state, int(rng.choice(candidates)) if len(candidates) else 81)
            if bool(game.is_terminal(state)):
                break
    assert features == {"capture", "rescue", "threat", "self_atari"}, features
    # Interior eye permits one enemy diagonal; edge eye permits none.
    board = np.zeros(81, int)
    board[adj[40]] = 1
    board[30] = -1
    assert bool(R.eye_mask(jnp.asarray(board))[40])
    board[32] = -1
    assert not bool(R.eye_mask(jnp.asarray(board))[40])
    board = np.zeros(81, int)
    board[adj[0]] = 1
    assert bool(R.eye_mask(jnp.asarray(board))[0])
    board[10] = -1
    assert not bool(R.eye_mask(jnp.asarray(board))[0])
    for player in (R.r0, R.r1):
        logits = np.asarray(player(jnp.zeros((1, 81, 3), bool),
                                   jnp.asarray([[False] * 81 + [True]])))[0]
        assert np.isneginf(logits[:81]).all() and logits[81] == 0
    state = step(step(game.init(), 81), 81)
    assert bool(game.is_terminal(state))
    np.testing.assert_array_equal(game.rewards(state), [-1, 1])
    print(f"PASS: {checked} legal-move scores agree with independent flood fill; "
          f"features={sorted(features)}; masks, eyes, pass fallback and komi checked.")


if __name__ == "__main__":
    main()
