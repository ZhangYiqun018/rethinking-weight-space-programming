"""Hand-built fixtures for every measured statistic and its denominator."""
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("study_truth", Path(__file__).with_name("truth.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
import jax
import jax.numpy as jnp
import numpy as np


def fixture(own=(), opponent=(), moves=None):
    board = np.zeros(81, np.int32)
    board[list(own)] = 1
    board[list(opponent)] = -1
    legal = np.append(board == 0, True)
    if moves is not None:
        legal[:] = False
        legal[list(moves) + [81]] = True
    return board, legal


def main():
    features = jax.jit(T.move_features)
    b, legal = fixture([31, 39, 49], [40], [41])
    f = features(b, legal)
    assert bool(f[41, 0]) and not bool(f[81, 0])
    np.testing.assert_array_equal(T.tally(f, 41, 1)[:3], [1, 1, 1])
    np.testing.assert_array_equal(T.tally(f, 81, 1)[:3], [1, 1, 0])
    b, legal = fixture([40], [31, 39, 49], [41])
    f = features(b, legal)
    assert bool(f[41, 1]) and not bool(f[81, 1])
    np.testing.assert_array_equal(T.tally(f, 41, 1)[3:5], [1, 1])
    # Capture at 32 frees point 31, saving group 40 without joining it.
    b, legal = fixture([40, 22, 30], [31, 39, 49], [32])
    f = features(b, legal)
    assert bool(f[32, 0]) and bool(f[32, 1])
    b, legal = fixture([], [31, 39, 49], [40])
    f = features(b, legal)
    assert bool(f[40, 2]) and not bool(f[40, 0])
    assert int(T.tally(f, 40, 1)[5]) == 1
    b, legal = fixture([31, 39, 41, 49], [], [40, 0])
    f = features(b, legal)
    assert bool(f[40, 3]) and int(T.tally(f, 40, 1)[6]) == 1
    legal[0] = False
    assert not bool(features(b, legal)[40, 3])
    # All-pass features are false; the early-pass boundary is own moves 29/30.
    f = jnp.zeros((82, 4), bool)
    assert int(T.tally(f, 81, 29)[7]) == 1
    assert int(T.tally(f, 81, 30)[7]) == 0
    assert int(T.tally(f, 0, 1)[7]) == 0
    np.testing.assert_array_equal(T.tally(f, 81, 30), [1, 0, 0, 0, 0, 0, 0, 0])
    first = T.first_move(lambda obs, legal: jnp.zeros((1, 82)))
    np.testing.assert_allclose(first["probabilities"], 1 / 82, rtol=1e-6)
    assert abs(first["p_tengen"] - 1 / 82) < 1e-7
    assert [(p["row"], p["col"]) for p in first["top3"]] == [(0, 0), (0, 1), (0, 2)]
    # Deterministic pass/pass games validate length, game-level early passes,
    # color allocation, and zero win rate of a fixed Black passer.
    state = T.M.GAME.init()
    state = T.M.GAME.step(T.M.GAME.step(state, 81), 81)
    assert int(state.step_count) == 2
    np.testing.assert_array_equal(T.M.GAME.rewards(state), [-1, 1])
    stats = T.summarize([
        {"a_black": True, "length": 10, "counts": [5, 2, 1, 2, 1, 1, 0, 2]},
        {"a_black": False, "length": 20, "counts": [10, 3, 2, 2, 0, 0, 1, 0]},
    ], 123, 2)
    expected = {"capture_rate": 3 / 5, "atari_escape_rate": 1 / 4,
                "self_atari_rate": 1 / 15, "eye_fill_rate": 1 / 15,
                "pass_before_60_rate": 1 / 2, "mean_game_length": 15}
    for key, value in expected.items():
        assert stats[key] == value, (key, stats[key])
    zero = T.summarize([{"a_black": True, "length": 2,
                         "counts": [1, 0, 0, 0, 0, 0, 0, 1]}], 0, 1)
    assert zero["capture_rate"] is None and zero["atari_escape_rate"] is None
    print("PASS: capture and escape opportunity denominators; extension and nonjoining capture rescue; "
          "self-atari; conditional eye-fill; pass/own-move 29-versus-30 boundary; "
          "empty-board probabilities/top3/tengen; length, color rewards and rate inversion.")


if __name__ == "__main__":
    main()
