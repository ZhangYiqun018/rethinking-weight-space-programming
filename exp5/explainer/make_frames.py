#!/usr/bin/env python3
"""Generate rollout frames of Astra's hand-written Breakout network (hand.npz).

Reuses observation() and Net from exp5/eval_minatar.py by path.
Writes blog/figs/breakout_explainer/frames.json (< 1 MB).
Run with:
    exp5/astra_write/.venv/bin/python -I exp5/explainer/make_frames.py
"""
import importlib.util
import json
import os

import numpy as np
from minatar import Environment

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_PATH = os.path.join(ROOT, "exp5", "eval_minatar.py")
HAND_NPZ = os.path.join(
    ROOT,
    "upstream",
    "lm_understand-logs",
    "exp5_write_from_scratch",
    "networks",
    "breakout",
    "hand.npz",
)
OUT_PATH = os.path.join(
    ROOT, "blog", "figs", "breakout_explainer", "frames.json"
)


def load_eval_minatar():
    spec = importlib.util.spec_from_file_location("eval_minatar", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r3(arr):
    return [round(float(v), 3) for v in arr]


def main():
    eval_minatar = load_eval_minatar()
    policy = eval_minatar.Net(HAND_NPZ)
    w1, b1, w2, b2, w3, b3 = policy.w

    seed = 0
    num_record = 380  # 380 consecutive steps: clears full first rack (30 bricks) + refills rack 2

    rng = np.random.default_rng(seed)
    env = Environment("breakout", sticky_action_prob=0.1)
    env.random = np.random.RandomState(seed)
    env.env.random = np.random.RandomState(seed + 1_000_003)
    env.reset()

    raw_steps = []
    cum_score = 0.0
    step = 0

    while step < eval_minatar.MAX_STEPS:
        g = env.env
        x = eval_minatar.observation(env, "breakout")
        h1 = np.tanh(x @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        logits = h2 @ w3 + b3
        p = np.exp(logits - logits.max())
        probs = p / p.sum()
        sampled_action = int(rng.choice(6, p=probs))
        greedy_action = int(np.argmax(logits))

        paddle_col = int(g.pos)
        ball_row, ball_col = int(g.ball_y), int(g.ball_x)
        trail_row, trail_col = int(g.last_y), int(g.last_x)
        ball_dir = int(g.ball_dir)
        brick_grid = g.brick_map.astype(int).tolist()

        r, done = env.act(sampled_action)
        cum_score += float(r)

        # In MinAtar breakout.py, new_y == 9 is reached inside act() iff before act()
        # ball_y == 8 and ball_dir in (2, 3). Regardless of whether the paddle catches
        # the ball at ball_x (row-8 col) or new_x (row-9 col) or misses, g.ball_x after
        # act() is set to new_x (the exact column where the ball reaches row 9).
        reached_row9 = (ball_row == 8) and (ball_dir in (2, 3))
        row9_col_at_step = int(g.ball_x) if reached_row9 else None
        row8_col_at_step = ball_col if reached_row9 else None

        # Check wall bounce on this step
        horiz_flipped = (ball_dir in (0, 3) and int(g.ball_dir) in (1, 2)) or (
            ball_dir in (1, 2) and int(g.ball_dir) in (0, 3)
        )
        wall_bounce = horiz_flipped and (ball_col in (0, 9) or int(g.ball_x) in (0, 9))
        paddle_catch = reached_row9 and (not done) and (int(g.ball_y) == 8)

        if step < num_record + 200:
            raw_steps.append(
                {
                    "step": step,
                    "paddle_col": paddle_col,
                    "ball": [ball_row, ball_col],
                    "trail": [trail_row, trail_col],
                    "ball_dir": ball_dir,
                    "brick_grid": brick_grid if step < num_record else None,
                    "h1": r3(h1) if step < num_record else None,
                    "h2": r3(h2) if step < num_record else None,
                    "logits": r3(logits) if step < num_record else None,
                    "action": sampled_action,
                    "greedy_action": greedy_action,
                    "reward": round(float(r), 3),
                    "score": int(round(cum_score)),
                    "reached_row9": reached_row9,
                    "row9_col_at_step": row9_col_at_step,
                    "row8_col_at_step": row8_col_at_step,
                    "paddle_catch": paddle_catch,
                    "wall_bounce": wall_bounce,
                }
            )
        step += 1
        if done:
            break

    episode_steps = step
    episode_score = cum_score

    # Populate ground-truth future landing_col (row 9) and catch_col (row 8) for each recorded step
    frames = []
    catches_count = 0
    wall_bounces_count = 0
    brick_hits_count = 0
    rack_refill_step = None

    for i in range(num_record):
        s = raw_steps[i]
        landing_col = None
        catch_col = None
        landing_step = None
        for j in range(i, len(raw_steps)):
            if raw_steps[j]["reached_row9"]:
                landing_col = raw_steps[j]["row9_col_at_step"]
                catch_col = raw_steps[j]["row8_col_at_step"]
                landing_step = raw_steps[j]["step"]
                break

        if s["paddle_catch"]:
            catches_count += 1
        if s["wall_bounce"]:
            wall_bounces_count += 1
        if s["reward"] > 0:
            brick_hits_count += 1
        if (
            i > 0
            and sum(sum(r) for r in raw_steps[i - 1]["brick_grid"]) == 0
            and sum(sum(r) for r in s["brick_grid"]) == 30
        ):
            rack_refill_step = i

        frames.append(
            {
                "step": s["step"],
                "paddle_col": s["paddle_col"],
                "ball": s["ball"],
                "trail": s["trail"],
                "ball_dir": s["ball_dir"],
                "landing_col": landing_col,
                "catch_col": catch_col,
                "landing_step": landing_step,
                "brick_grid": s["brick_grid"],
                "h1": s["h1"],
                "h2": s["h2"],
                "logits": s["logits"],
                "action": s["action"],
                "greedy_action": s["greedy_action"],
                "reward": s["reward"],
                "score": s["score"],
                "paddle_catch": s["paddle_catch"],
                "wall_bounce": s["wall_bounce"],
            }
        )

    out_data = {
        "game": "breakout",
        "seed": seed,
        "sticky_action_prob": 0.1,
        "num_frames": num_record,
        "episode_score": episode_score,
        "episode_steps": episode_steps,
        "recorded_score": frames[-1]["score"],
        "summary": {
            "catches": catches_count,
            "wall_bounces": wall_bounces_count,
            "brick_hits": brick_hits_count,
            "rack_refill_step": rack_refill_step,
        },
        "action_names": ["no-op", "left", "up", "right", "down", "fire"],
        "frames": frames,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB, {num_record} frames)")
    print(
        f"Full episode (seed={seed}): score={episode_score}, steps={episode_steps}; "
        f"recorded 0..{num_record - 1}: score={frames[-1]['score']}, "
        f"catches={catches_count}, wall_bounces={wall_bounces_count}, "
        f"brick_hits={brick_hits_count}, rack_refill_step={rack_refill_step}"
    )


if __name__ == "__main__":
    main()
