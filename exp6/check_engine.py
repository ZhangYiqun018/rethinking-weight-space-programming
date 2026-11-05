"""Check the match adapter against pgx's public environment API."""
import jax
import jax.numpy as jnp
import numpy as np
import pgx
from pgx._src.games.go import Game


def main():
    public = pgx.make("go_9x9")
    direct = Game(size=9, komi=7.5, history_length=1, max_termination_steps=162)
    public_step = jax.jit(public.step)
    direct_step = jax.jit(direct.step)
    legal_fn = jax.jit(direct.legal_action_mask)
    reward_fn = jax.jit(direct.rewards)
    rng = np.random.default_rng(5521)
    positions = 0
    endings = set()
    for game in range(6):
        p = public.init(jax.random.PRNGKey(game))
        d = direct.init()
        while True:
            legal = np.asarray(legal_fn(d))
            np.testing.assert_array_equal(p.legal_action_mask, legal)
            sign = 1 if int(d.color) == 0 else -1
            board = np.asarray(d.board) * sign
            obs = np.stack((board > 0, board < 0, np.full(81, int(d.color) == 1)), -1)
            np.testing.assert_array_equal(np.asarray(p.observation)[..., [0, 1, 16]].reshape(81, 3), obs)
            np.testing.assert_array_equal(p._x.board, d.board)
            assert int(p.current_player) == int(p._player_order[int(d.color)])
            positions += 1
            # First trajectory tests two passes; others deliberately reach late play.
            choices = np.flatnonzero(legal[:81])
            action = 81 if game == 0 or not len(choices) else int(rng.choice(choices))
            p = public_step(p, jnp.int32(action))
            d = direct_step(d, jnp.int32(action))
            assert bool(p.terminated) == bool(direct.is_terminal(d))
            np.testing.assert_array_equal(np.asarray(p.rewards)[np.asarray(p._player_order)],
                                          reward_fn(d))
            if bool(p.terminated):
                endings.add("two_passes" if int(d.consecutive_pass_count) >= 2 else
                            "repetition" if bool(d.is_psk) else "move_limit")
                break
    assert "two_passes" in endings and "move_limit" in endings
    print(f"PASS: {positions} public-pgx/direct-engine positions agree on observations, "
          f"boards, legal masks, player/color mapping, termination and rewards; endings={sorted(endings)}.")


if __name__ == "__main__":
    main()
