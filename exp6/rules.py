"""Fixed, preregistered one-ply Go heuristics using only board observations."""
import jax
import jax.numpy as jnp
import numpy as np

POINTS = np.arange(81)
ROW, COL = POINTS // 9, POINTS % 9
ADJ = jnp.asarray((abs(ROW[:, None] - ROW) + abs(COL[:, None] - COL)) == 1)
DIAG = jnp.asarray((abs(ROW[:, None] - ROW) == 1) & (abs(COL[:, None] - COL) == 1))
EDGE = jnp.asarray((ROW == 0) | (ROW == 8) | (COL == 0) | (COL == 8))
LINE = jnp.asarray(np.isin(np.minimum.reduce([ROW, COL, 8 - ROW, 8 - COL]), [2, 3]))
INDEX = jnp.arange(81)


def eye_mask(board):
    surrounded = jnp.all(~ADJ | (board[None, :] == 1), axis=1)
    bad_diagonals = jnp.sum(DIAG & (board[None, :] == -1), axis=1)
    return (board == 0) & surrounded & (bad_diagonals <= jnp.where(EDGE, 0, 1))


def finish(board, legal, scores):
    eligible = legal[:81] & ~eye_mask(board)
    moves = jnp.where(eligible, scores, -jnp.inf)
    return jnp.append(moves, jnp.where(eligible.any(), -jnp.inf, 0.0))


def board_of(obs):
    return obs[:, 0].astype(jnp.int32) - obs[:, 1].astype(jnp.int32)


def groups(board):
    """Label same-color components; derive unique liberties by adjacency union."""
    same = ADJ & (board[:, None] == board[None, :]) & (board[:, None] != 0)
    initial = jnp.where(board != 0, INDEX, 81)

    def propagate(carry):
        labels, _ = carry
        updated = jnp.minimum(labels, jnp.min(jnp.where(same, labels[None, :], 81), axis=1))
        return updated, jnp.any(updated != labels)

    labels, _ = jax.lax.while_loop(lambda c: c[1], propagate, (initial, True))
    members = (INDEX[:, None] == labels[None, :]) & (board[None, :] != 0)
    neighbors = (members.astype(jnp.float32) @ ADJ.astype(jnp.float32)) > 0
    liberties = neighbors & (board[None, :] == 0)
    return members, neighbors, liberties.sum(axis=1)


def tactical_scores(board):
    members, neighbors, liberty_count = groups(board)
    own = (members & (board[None, :] == 1)).any(axis=1)
    opponent = (members & (board[None, :] == -1)).any(axis=1)
    # Rows below are candidate moves, columns are original groups or points.
    touched = neighbors.T
    killed = touched & opponent[None, :] & (liberty_count[None, :] == 1)
    removed = (killed.astype(jnp.float32) @ members.astype(jnp.float32)) > 0
    captures = removed.sum(axis=1)
    joined = touched & own[None, :]
    new_group = ((joined.astype(jnp.float32) @ members.astype(jnp.float32)) > 0) | jnp.eye(81, dtype=bool)
    empty_after = ((board[None, :] == 0) | removed) & ~jnp.eye(81, dtype=bool)
    new_neighbors = (new_group.astype(jnp.float32) @ ADJ.astype(jnp.float32)) > 0
    new_liberties = (new_neighbors & empty_after).sum(axis=1)
    rescue = (joined & (liberty_count[None, :] == 1)).any(axis=1) & (new_liberties >= 2)
    enemy_liberties_after = empty_after.astype(jnp.float32) @ neighbors.T.astype(jnp.float32)
    threaten = (touched & opponent[None, :] & ~killed &
                (liberty_count[None, :] >= 2) & (enemy_liberties_after == 1)).any(axis=1)
    self_atari = (new_liberties == 1) & (captures == 0)
    early_friend = (ADJ & (board[None, :] == 1)).any(axis=1) & ((board != 0).sum() < 40)
    return (100 * captures + 10 * rescue + threaten - 20 * self_atari +
            0.03 * LINE + 0.02 * early_friend)


@jax.jit
def r0(observation, legal):
    return jax.vmap(lambda obs, mask: finish(board_of(obs), mask, jnp.zeros(81)))(observation, legal)


@jax.jit
def r1(observation, legal):
    def policy(obs, mask):
        board = board_of(obs)
        return finish(board, mask, 10 * tactical_scores(board))
    return jax.vmap(policy)(observation, legal)
