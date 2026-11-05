"""Build Exp1 ("which game?") prompts for the controlled re-run (REPRO_PLAN.md Phase 3b, H5/H6).

The weights section is generated from a network's .npz in the authors' format (3 decimals, one line per row), so
the 'orig' condition reproduces their prompt byte for byte (checked by --verify). Every condition changes exactly
one thing:

  orig      the authors' prompt
  anon      channel meanings removed ("channel 2: meaning not given"); the wall border is still stated
  swap12    the descriptions of channels 1 (moving objects) and 2 (reward targets) swapped, weights untouched
  perm      hidden units of both layers permuted (the network computes exactly the same function)
  shuffle   W1's rows shuffled (destroys all spatial and channel structure; the right answer is "cannot tell")
  randinit  an untrained network with the training init (orthogonal, gain sqrt 2, output gain 0.01)

    python make_prompts.py --verify                 # orig == authors' prompts
    python make_prompts.py --out prompts            # all conditions for all four networks
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "..", "upstream", "lm_understand-logs", "exp1_which_game")
CHANNEL_LINES = [
    "  channel 0  agent: the cells the player controls",
    "  channel 1  moving objects: things other than the agent that move",
    "  channel 2  reward targets: things that give reward",
    "  channel 3  obstacles: cells the agent cannot enter or that end the episode (the wall border is always here)",
    "  channel 4  previous frame: the cells that were in channel 0 or channel 1 on the previous step",
]
CONDITIONS = ["orig", "anon", "swap12", "perm", "shuffle", "randinit", "anonshuffle"]


def fmt_rows(a):
    return "\n".join(" ".join(f"{x:.3f}".replace("-0.000", "0.000") for x in row) for row in np.atleast_2d(a))


def weights_text(w):
    w1, b1, w2, b2, w3, b3 = w
    return "\n".join([
        "W1 (layer 1 weights), shape 500 x 32: one line per row (row i = input i), 32 values per line", fmt_rows(w1), "",
        "b1 (layer 1 biases), shape 32:", fmt_rows(b1), "",
        "W2 (layer 2 weights), shape 32 x 32: one line per row", fmt_rows(w2), "",
        "b2 (layer 2 biases), shape 32:", fmt_rows(b2), "",
        "W3 (output layer weights), shape 32 x 5: one line per row", fmt_rows(w3), "",
        "b3 (output layer biases), shape 5:", fmt_rows(b3)]) + "\n"


def load_actor(path):
    z = np.load(path, allow_pickle=False)
    return [z[f"actor/{n}/{p}"].astype(np.float64) for n in ("dense1", "dense2", "logits") for p in ("kernel", "bias")]


def header(condition):
    lines = open(os.path.join(UPSTREAM, "runs", "breakout", "prompt.txt")).read().split("W1 (layer 1 weights)", 1)[0].rstrip("\n").split("\n")
    if condition in ("anon", "anonshuffle"):  # anonshuffle = anon header + shuffle weights
        for k, line in enumerate(CHANNEL_LINES):
            lines[lines.index(line)] = f"  channel {k}  meaning not given" + (
                " (the wall border is always here)" if k == 3 else "")
    if condition == "swap12":
        i1, i2 = lines.index(CHANNEL_LINES[1]), lines.index(CHANNEL_LINES[2])
        lines[i1] = CHANNEL_LINES[2].replace("channel 2", "channel 1")
        lines[i2] = CHANNEL_LINES[1].replace("channel 1", "channel 2")
    return "\n".join(lines).rstrip("\n")


def transform(w, condition, seed):
    rng = np.random.default_rng(seed)
    w1, b1, w2, b2, w3, b3 = [a.copy() for a in w]
    if condition == "perm":
        p1, p2 = rng.permutation(32), rng.permutation(32)
        w1, b1 = w1[:, p1], b1[p1]
        w2, b2 = w2[p1][:, p2], b2[p2]
        w3 = w3[p2]
    elif condition in ("shuffle", "anonshuffle"):
        w1 = w1[rng.permutation(500)]
    elif condition == "randinit":
        def ortho(shape, gain):
            q, r = np.linalg.qr(rng.standard_normal((max(shape), min(shape))))
            q = q * np.sign(np.diag(r))
            return gain * (q if shape[0] >= shape[1] else q.T)[:shape[0], :shape[1]]
        w1, w2, w3 = ortho((500, 32), np.sqrt(2)), ortho((32, 32), np.sqrt(2)), ortho((32, 5), 0.01)
        b1, b2, b3 = np.zeros(32), np.zeros(32), np.zeros(5)
    return [w1, b1, w2, b2, w3, b3]


def build(net_id, condition, seed=0):
    w = transform(load_actor(os.path.join(UPSTREAM, "networks", f"{net_id}.npz")), condition, seed)
    return header(condition) + "\n\n" + weights_text(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    key = json.load(open(os.path.join(HERE, "networks", "answer_key.json")))
    if a.verify:
        for net_id, info in key.items():
            theirs = open(os.path.join(UPSTREAM, "runs", info["game"], "prompt.txt")).read()
            ours = build(net_id, "orig")
            same = ours == theirs
            print(net_id, info["game"], "identical" if same else f"DIFFERENT ({len(ours)} vs {len(theirs)} chars)")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for net_id in key:
            for c in CONDITIONS:
                with open(os.path.join(a.out, f"{net_id}__{c}.txt"), "w") as f:
                    f.write(build(net_id, c))
        print(f"{len(key) * len(CONDITIONS)} prompts -> {a.out}")


if __name__ == "__main__":
    main()
