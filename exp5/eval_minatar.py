"""Score MinAtar policies the way Exp5 describes, with our own observation wrapper (REPRO_PLAN.md Phase 1b / 2c).

Protocol from the authors' prompt: official MinAtar 1.0.15 games, sticky actions 0.1, actions sampled from the
softmax of the logits, episodes cut at 10,000 steps, mean score over 1,000 episodes.

The 600-number input is rebuilt from the channel description in each prompt ("THE INPUT"): index
(row * 10 + column) * 6 + channel, channels player / danger / reward / projectile / trail / gauge.

    python eval_minatar.py --game breakout --policy net:networks/breakout/hand.npz --episodes 1000
    python eval_minatar.py --game breakout --policy heuristic
"""
import argparse
import copy
import json
import multiprocessing as mp

import numpy as np
from minatar import Environment

# Our channel -> MinAtar channels, per game, as stated in the prompts.
CHANNELS = {
    "breakout": {0: ["paddle"], 2: ["brick"], 3: ["ball"], 4: ["trail"]},
    "seaquest": {0: ["sub_front", "sub_back"], 1: ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 2: ["enemy_fish", "enemy_sub", "diver"], 3: ["friendly_bullet"], 4: ["trail", "sub_back"],
                 5: ["oxygen_guage", "diver_guage"]},
    "asterix": {0: ["player"], 1: ["enemy"], 2: ["gold"], 4: ["trail"]},
}
MAX_STEPS = 10_000


def observation(env, game):
    state = env.state()
    idx = env.env.channels
    x = np.zeros((10, 10, 6), dtype=np.float32)
    for ours, theirs in CHANNELS[game].items():
        x[:, :, ours] = np.max(np.stack([state[:, :, idx[c]] for c in theirs]), axis=0)
    return x.reshape(-1)


class Net:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.w = [z[f"actor/{n}/{p}"] for n in ("dense1", "dense2", "logits") for p in ("kernel", "bias")]

    def logits(self, x, env=None):
        w1, b1, w2, b2, w3, b3 = self.w
        return np.tanh(np.tanh(x @ w1 + b1) @ w2 + b2) @ w3 + b3


class Heuristic:
    """Breakout only, no network: move the paddle under the column where the ball will reach the paddle row.

    keep_direction=False aims at the landing column, which reverses the ball's horizontal direction on every catch.
    keep_direction=True aims at the column the ball is in one row above the paddle; MinAtar then bounces it straight
    back up and the horizontal direction is kept (the insight in Astra's Breakout design).
    """

    def __init__(self, keep_direction=False):
        self.keep_direction = keep_direction

    def logits(self, x, env):
        g = env.env
        target = g.ball_x
        if g.ball_dir in (2, 3):  # falling: follow MinAtar's wall rule (clamp, then flip) down to the paddle row
            x_pos, dx, cols = g.ball_x, (1 if g.ball_dir == 2 else -1), {g.ball_y: g.ball_x}
            for y_pos in range(g.ball_y + 1, 10):
                x_pos += dx
                if x_pos < 0 or x_pos > 9:
                    x_pos, dx = min(max(x_pos, 0), 9), -dx
                cols[y_pos] = x_pos
            # the paddle moves before the ball in act(), so it gets one move per row still to fall
            moves_left = 9 - g.ball_y
            keep = cols.get(8, cols[9])
            target = keep if self.keep_direction and abs(keep - g.pos) <= moves_left else cols[9]
        out = np.full(6, -50.0)
        out[1 if target < g.pos else (3 if target > g.pos else 0)] = 50.0
        return out


class SimulatorHeuristic:
    """Breakout only, an upper reference for hand-written programs: it copies the game state, simulates the ball
    (bricks, walls and ceiling included) until it next reaches the paddle row, and moves the paddle there early,
    preferring the catch column that keeps the ball's horizontal direction. It reads the simulator, which no
    network can, so it is a ceiling for "a program that knows the rules", not a fair competitor."""

    def logits(self, x, env):
        g = env.env
        sim = copy.deepcopy(g)
        sim.random = np.random.RandomState(0)
        keep = land = None
        for step in range(1, 200):
            if sim.ball_y == 8 and sim.ball_dir in (2, 3):
                keep = sim.ball_x
                land = min(max(sim.ball_x + (1 if sim.ball_dir == 2 else -1), 0), 9)
                break
            sim.pos = sim.ball_x  # keep the simulated paddle under the ball so a bounce cannot end the copy
            sim.act(0)
        target = g.ball_x if keep is None else (keep if abs(keep - g.pos) <= step else land)
        out = np.full(6, -50.0)
        out[1 if target < g.pos else (3 if target > g.pos else 0)] = 50.0
        return out


def make_policy(spec):
    if spec.startswith("net:"):
        return Net(spec[4:])
    if spec == "random":
        return None
    if spec == "heuristic":
        return Heuristic()
    if spec == "heuristic_keepdir":
        return Heuristic(keep_direction=True)
    if spec == "heuristic_sim":
        return SimulatorHeuristic()
    raise ValueError(spec)


def episode(args):
    game, spec, seed = args
    policy = make_policy(spec)
    rng = np.random.default_rng(seed)
    env = Environment(game, sticky_action_prob=0.1)
    env.random = np.random.RandomState(seed)
    env.env.random = np.random.RandomState(seed + 1_000_003)
    env.reset()
    total, steps = 0.0, 0
    while steps < MAX_STEPS:
        if policy is None:
            a = int(rng.integers(6))
        else:
            z = policy.logits(observation(env, game), env)
            p = np.exp(z - z.max())
            a = int(rng.choice(6, p=p / p.sum()))
        r, done = env.act(a)
        total += r
        steps += 1
        if done:
            break
    return total, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=sorted(CHANNELS))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    jobs = [(a.game, a.policy, a.seed * 1_000_000 + i) for i in range(a.episodes)]
    with mp.Pool() as pool:
        res = pool.map(episode, jobs, chunksize=4)
    scores = np.array([r[0] for r in res])
    lengths = np.array([r[1] for r in res])
    out = {"game": a.game, "policy": a.policy, "episodes": a.episodes, "mean": float(scores.mean()),
           "se": float(scores.std(ddof=1) / np.sqrt(len(scores))), "max": float(scores.max()),
           "mean_length": float(lengths.mean()), "cut_at_max_steps": float((lengths >= MAX_STEPS).mean())}
    print(json.dumps(out))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
