"""Acceptance check: parsing the authors' answer.md must reproduce their hand.npz and statement count exactly."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from edits_parser import apply_reply, to_npz_arrays

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, "..", "..", "upstream", "lm_understand-logs", "exp5_write_from_scratch")


def check(game):
    run = os.path.join(UP, "runs", game)
    res = apply_reply(open(os.path.join(run, "reply", "answer.md"), encoding="utf-8").read())
    rep = json.load(open(os.path.join(run, "apply.json")))["edit_report"]
    ref = np.load(os.path.join(UP, "networks", game, "hand.npz"), allow_pickle=False)
    mine = to_npz_arrays(res)
    same_keys = sorted(mine) == sorted(ref.files)
    eq = same_keys and all(np.array_equal(mine[k], ref[k]) and mine[k].dtype == ref[k].dtype for k in ref.files)
    count_ok = res.statements == rep["statements"] and res.applied == rep["applied"]
    print(f"{game:9s} arrays_equal={eq} statements={res.statements}/{rep['statements']} "
          f"applied={res.applied}/{rep['applied']} skipped={res.skipped} errors={list(res.errors)[:2]}")
    return eq and count_ok


if __name__ == "__main__":
    oks = [check(g) for g in ("breakout", "seaquest", "asterix")]
    print(f"{sum(oks)}/3 games pass")
    sys.exit(0 if all(oks) else 1)
