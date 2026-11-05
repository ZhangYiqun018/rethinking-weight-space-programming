"""Scripted steering baseline, step 2: write candidate checkpoints from directions.npz.

Edit per target layer L, on the selected neurons J (columns of down_proj):
    W[:, J] <- W[:, J] - k * v (v^T W[:, J])
k = 1 removes v from those neurons' writes; k > 1 writes against it. That is the same form as the edit under audit
(rank one, on 64 columns of layers 15 and 19); only v, J and k come from the script.

Variants (all with the same layers, the same number of columns and the same k):
    steer      v and J from directions.npz
    randdir    a random unit direction orthogonal to v, same J      (is it the direction?)
    randcols   v, but 64 random columns                             (is it the neurons?)

    python make_edit.py <base_dir> <directions.npz> <out_dir> --variant steer --k 2
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("directions")
    ap.add_argument("out_dir")
    ap.add_argument("--variant", choices=["steer", "randdir", "randcols"], default="steer")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.directions, allow_pickle=False)
    rng = np.random.default_rng(a.seed)
    index = json.load(open(os.path.join(a.base_dir, "model.safetensors.index.json")))["weight_map"]
    shutil.copytree(a.base_dir, a.out_dir, ignore=shutil.ignore_patterns("*.safetensors", ".cache"), dirs_exist_ok=True)

    edits, report = {}, {"variant": a.variant, "k": a.k, "seed": a.seed, "layers": {}}
    for L in d["layers"].tolist():
        v, cols = d[f"v{L}"].astype(np.float64), d[f"top{L}"]
        if a.variant == "randdir":
            r = rng.standard_normal(v.shape)
            r -= (r @ v) * v
            v = r / np.linalg.norm(r)
        if a.variant == "randcols":
            cols = rng.choice(d[f"score{L}"].shape[0], size=len(cols), replace=False)
        edits[f"layers.{L}.mlp.down_proj.weight"] = (v, np.sort(cols))
    for shard in sorted(set(index.values())):
        sd = load_file(os.path.join(a.base_dir, shard))
        for name in list(sd):
            key = next((k for k in edits if name.endswith(k)), None)
            if key is None:
                continue
            v, cols = edits[key]
            W = sd[name].double().numpy()
            sub = W[:, cols]
            W[:, cols] = sub - a.k * np.outer(v, v @ sub)
            new = torch.from_numpy(W).to(sd[name].dtype)
            report["layers"][name] = {"columns": cols.tolist(),
                                      "relative_change": float((new.float() - sd[name].float()).norm() / sd[name].float().norm())}
            sd[name] = new.contiguous()
        save_file(sd, os.path.join(a.out_dir, shard), metadata={"format": "pt"})
    assert len(report["layers"]) == len(edits), report
    json.dump(report, open(os.path.join(a.out_dir, "steer_edit.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}),
          {n: round(r["relative_change"], 4) for n, r in report["layers"].items()})


if __name__ == "__main__":
    main()
