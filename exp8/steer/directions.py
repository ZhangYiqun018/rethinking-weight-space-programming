"""Scripted steering baseline, step 1 (REPRO_PLAN.md Phase 2b, H3): find the repetition direction without an LLM.

Takes the base model's own generations on the 'diag' practice split (config A) and labels token windows:
- repetitive: a window of answers cut at the token limit, kept only if most of its 8-grams already occurred earlier
  in the same answer;
- productive: the thinking of answers that closed </think> and were graded correct.
Then teacher-forces those texts and, at the MLP of each target layer, records the mean down_proj output (the
residual write) and the mean down_proj input (the 3,584 neuron activations) over each label. Per layer:
- v = normalised (mean write on repetitive - mean write on productive);
- neuron score s_j = (mean act_j repetitive - mean act_j productive) * (v . W[:, j]): how much neuron j's change
  pushes along v. The edit uses the top 64 neurons by s_j.

This is the textbook difference-of-means recipe. No step uses the authors' prompts, edits or notes.

    python directions.py <model_dir> <generations.jsonl> <out.npz> [--layers 15,19]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from configs import load_benchmark  # noqa: E402
from grade import answer_part, last_boxed, math_correct  # noqa: E402

WINDOW = 1024        # tokens per labelled window
REP_START = 8192     # repetitive windows start here (loops are well established by then)
MAX_TEXTS = 160      # per label


def repeated_ngram_share(ids, n=8):
    seen, rep = set(), 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(ids) - n + 1)


def label_windows(tok, gen_path):
    rows = {r["id"]: r for r in load_benchmark("practice_diag")}
    rep, prod = [], []
    for rec in map(json.loads, open(gen_path)):
        row = rows[rec["id"]]
        prompt = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        g_ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
        if rec["finish_reason"] == "length" and len(g_ids) >= REP_START + WINDOW:
            w = g_ids[REP_START:REP_START + WINDOW]
            if repeated_ngram_share(w) >= 0.5:
                rep.append((p_ids + g_ids[:REP_START + WINDOW], len(p_ids) + REP_START))
        elif "</think>" in rec["text"] and math_correct(last_boxed(answer_part(rec["text"])), row["answer"]):
            think_ids = tok(rec["text"].split("</think>")[0], add_special_tokens=False)["input_ids"]
            if len(think_ids) >= 256:
                start = max(64, len(think_ids) - WINDOW)
                prod.append((p_ids + think_ids, len(p_ids) + start))
    return rep[:MAX_TEXTS], prod[:MAX_TEXTS]


def find_down_proj(model, layer):
    hits = [m for n, m in model.named_modules() if n.endswith(f"layers.{layer}.mlp.down_proj")]
    assert len(hits) == 1, (layer, len(hits))
    return hits[0]


@torch.no_grad()
def mean_stats(model, texts, layers):
    sums = {L: {"act": 0.0, "write": 0.0} for L in layers}
    count, store = 0, {}
    handles = []
    for L in layers:
        mod = find_down_proj(model, L)
        handles.append(mod.register_forward_hook(
            lambda m, inp, out, L=L: store.__setitem__(L, (inp[0][0].float(), out[0].float()))))
    try:
        for ids, start in texts:
            store.clear()
            model(input_ids=torch.tensor([ids], device=model.device))
            for L in layers:
                act, write = store[L]
                sums[L]["act"] = sums[L]["act"] + act[start:].sum(0)
                sums[L]["write"] = sums[L]["write"] + write[start:].sum(0)
            count += len(ids) - start
    finally:
        for h in handles:
            h.remove()
    return {L: {k: (v / count).cpu().numpy() for k, v in d.items()} for L, d in sums.items()}, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("generations")
    ap.add_argument("out")
    ap.add_argument("--layers", default="15,19")
    ap.add_argument("--top", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-texts", type=int, default=MAX_TEXTS, help="per label; small values for a dry run")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    rep, prod = label_windows(tok, a.generations)
    rep, prod = rep[:a.max_texts], prod[:a.max_texts]
    print(f"labelled windows: {len(rep)} repetitive, {len(prod)} productive", flush=True)
    assert len(rep) >= 16 and len(prod) >= 8, "too few labelled windows to estimate a direction"
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16).to(a.device).eval()
    s_rep, n_rep = mean_stats(model, rep, layers)
    s_prod, n_prod = mean_stats(model, prod, layers)
    out = {"layers": np.array(layers), "n_rep_tokens": n_rep, "n_prod_tokens": n_prod,
           "n_rep_texts": len(rep), "n_prod_texts": len(prod)}
    for L in layers:
        W = find_down_proj(model, L).weight.detach().float().cpu().numpy()  # (hidden, intermediate)
        diff = s_rep[L]["write"] - s_prod[L]["write"]
        v = diff / np.linalg.norm(diff)
        score = (s_rep[L]["act"] - s_prod[L]["act"]) * (v @ W)
        top = np.argsort(-score)[:a.top]
        out[f"v{L}"], out[f"score{L}"], out[f"top{L}"] = v, score, top
        print(f"layer {L}: |diff|={np.linalg.norm(diff):.4f}, top-{a.top} share of positive push "
              f"{score[top].sum() / score[score > 0].sum():.3f}", flush=True)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
