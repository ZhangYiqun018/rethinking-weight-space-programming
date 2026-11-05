"""Build the edited Qwen3.5-0.8B from the public base weights and the authors' sparse edit (Exp8, run p50c).

Independent of the authors' apply_edit.py: reads edit.npz (per tensor: flat indices and bfloat16 bit patterns)
and manifest.json, checks the base and rebuilt tensors against the manifest's sha256, and writes a full HF model
directory with the edited tensors replaced.

    python apply_edit.py <base_model_dir> <edit_dir> <out_dir>
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def bf16_bits(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def sha256(bits):
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


def main(base_dir, edit_dir, out_dir):
    manifest = json.load(open(os.path.join(edit_dir, "manifest.json")))
    edit = np.load(os.path.join(edit_dir, "edit.npz"), allow_pickle=False)
    targets = {t["name"]: (i, t) for i, t in enumerate(manifest["tensors"])}
    assert all(t["form"] == "sparse" for _, t in targets.values()), "only sparse edits are expected for p50c"

    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"] if os.path.exists(index_path) else None
    shards = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]

    shutil.copytree(base_dir, out_dir, ignore=shutil.ignore_patterns("*.safetensors", ".cache"), dirs_exist_ok=True)
    report = {}
    for shard in shards:
        sd = load_file(os.path.join(base_dir, shard))
        for name in [n for n in sd if n in targets]:
            i, t = targets[name]
            bits = bf16_bits(sd[name])
            assert sd[name].dtype == torch.bfloat16 and list(bits.shape) == t["shape"], (name, sd[name].dtype)
            assert sha256(bits) == t["base_sha256"], f"{name}: base tensor differs from the one the edit was made on"
            flat = bits.reshape(-1).copy()
            flat[edit[f"t{i}_idx"]] = edit[f"t{i}_val"]
            new = flat.reshape(bits.shape)
            assert sha256(new) == t["edited_sha256"], f"{name}: rebuilt tensor does not match the evaluated checkpoint"
            changed = int((new != bits).sum())
            assert changed == t["changed_entries"], (name, changed, t["changed_entries"])
            sd[name] = torch.from_numpy(new.view(np.int16)).view(torch.bfloat16)
            report[name] = {"changed_entries": changed, "shard": shard}
            print(f"{name}: {changed:,} entries changed, sha256 OK")
        save_file(sd, os.path.join(out_dir, shard), metadata={"format": "pt"})
    missing = set(targets) - set(report)
    assert not missing, f"tensors not found in the base model: {missing}"
    json.dump(report, open(os.path.join(out_dir, "edit_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
