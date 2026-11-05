"""Exp1 no-LLM baseline (H6): name the game from a policy network's weights with three fixed rules from game mechanics.

Rules and thresholds were written down (2026-11-04) before this script was first run. They were not tuned
afterwards. They are not blind: the author knew which network is which game.

Probe: random observations in the prompt's format: channel 3 = wall border; every inner cell of channels
0, 1, 2 and 4 is on with probability 0.05. Logits come from the actor weights only; the game is never run.
  V = std over probes of (logit_up - logit_down), the decision signal along the vertical axis
  H = std over probes of (logit_left - logit_right), the same along the horizontal axis
Rules:
  R0  max(V, H) < 0.05                          -> unknown (no input-dependent decision at all)
  R1  V > 3 H  -> vertical game:  |mean(up - down)| / V > 1 -> freeway (always heads up), else pong
  R2  H > 3 V  -> horizontal game: breakout
  R3  otherwise (both axes matter)              -> pacman (maze)

    python -I stats_baseline.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_prompts import load_actor, transform  # noqa: E402

NO_OP, UP, DOWN, LEFT, RIGHT = range(5)
AXIS_RATIO, FREEWAY_RATIO, MIN_SIGNAL = 3.0, 1.0, 0.05


def probes(n=4000, p=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 10, 10, 5))
    inner = np.zeros((10, 10), bool)
    inner[1:9, 1:9] = True
    for ch in (0, 1, 2, 4):
        x[:, :, :, ch] = (rng.random((n, 10, 10)) < p) & inner
    x[:, :, :, 3] = ~inner
    return x.reshape(n, 500)


def logits(w, x):
    w1, b1, w2, b2, w3, b3 = w
    return np.tanh(np.tanh(x @ w1 + b1) @ w2 + b2) @ w3 + b3


def classify(w, x):
    z = logits(w, x)
    ud, lr = z[:, UP] - z[:, DOWN], z[:, LEFT] - z[:, RIGHT]
    V, H = ud.std(), lr.std()
    feats = {"V": round(float(V), 3), "H": round(float(H), 3), "up_bias": round(float(abs(ud.mean()) / max(V, 1e-9)), 2)}
    if max(V, H) < MIN_SIGNAL:
        return "unknown", feats
    if V > AXIS_RATIO * H:
        return ("freeway" if feats["up_bias"] > FREEWAY_RATIO else "pong"), feats
    if H > AXIS_RATIO * V:
        return "breakout", feats
    return "pacman", feats


def main():
    key = json.load(open(os.path.join(HERE, "networks", "answer_key.json")))
    x = probes()
    print(f"{'net':9s} {'truth':9s} {'condition':9s} {'guess':9s} ok  features")
    hits = {}
    for net in sorted(key):
        truth = key[net] if isinstance(key[net], str) else key[net].get("game")
        for cond in ("orig", "perm", "shuffle", "randinit"):  # orig = anon = swap12 here: the script never reads names
            guess, f = classify(transform(load_actor(os.path.join(HERE, "..", "upstream", "lm_understand-logs", "exp1_which_game", "networks", f"{net}.npz")), cond, 0), x)
            ok = guess == ("unknown" if cond == "randinit" else truth)
            hits.setdefault(cond, []).append(ok)
            print(f"{net:9s} {truth:9s} {cond:9s} {guess:9s} {int(ok)}   {f}")
    print({c: f"{sum(v)}/{len(v)}" for c, v in hits.items()})


if __name__ == "__main__":
    main()
