"""Weight-only (no-LLM) baseline for Exp3. Usage: baseline.py predict | score | stats  (see PREREG.md).
`predict` never opens eval/; `score` is the only mode reading eval/. Run with python -I."""
import json, sys
import numpy as np

from pathlib import Path
OUT = str(Path(__file__).resolve().parent)
ROOT = str(Path(OUT).parent / "upstream/lm_understand-logs/exp3_language")
CORPORA = ["python", "romaji", "english"]
L, H, HD = 4, 6, 64
Z = 3 * 1.4826  # X = 3 robust SDs, fixed in PREREG.md


def load(c):
    z = np.load(f"{ROOT}/networks/{c}/blocks.npz", allow_pickle=False)
    return {k: z[k].astype(np.float64) for k in z.files}


def head_mats(W):
    out = {}
    for l in range(L):
        g = W[f"ln1_w.{l}"]
        for h in range(H):
            s = slice(h * HD, (h + 1) * HD)
            q = W[f"W_q.{l}"][s] * g
            k = W[f"W_k.{l}"][s] * g
            v = W[f"W_v.{l}"][s] * g
            out[(l, h)] = dict(qk=q.T @ k, ov=W[f"W_o.{l}"][:, s] @ v)
    return out


def fro(a):
    return np.linalg.norm(a)


def robust_thr(x):
    x = np.asarray(x)
    m = np.median(x)
    return m + Z * np.median(np.abs(x - m))


def predict(c):
    M = head_mats(load(c))
    ids = [(l, h) for l in range(L) for h in range(H)]
    qkn = {i: fro(M[i]["qk"]) for i in ids}
    ovn = {i: fro(M[i]["ov"]) for i in ids}
    kc = {}
    for b in ids:
        for a in ids:
            if a[0] < b[0]:
                kc[(a, b)] = fro(M[b]["qk"] @ M[a]["ov"]) / (qkn[b] * ovn[a])
    S = {b: max(kc[(a, b)] for a in ids if a[0] < b[0]) for b in ids if b[0] >= 1}
    arg = {b: max((a for a in ids if a[0] < b[0]), key=lambda a: kc[(a, b)]) for b in S}
    P = {}
    for i in ids:
        ev = np.linalg.eigvals(M[i]["ov"])
        P[i] = float(ev.real.sum() / np.abs(ev).sum())
    t1 = robust_thr(list(S.values()))
    t3 = robust_thr(list(P.values()))
    r1 = [b for b in S if S[b] > t1]
    r2 = [b for b in r1 if P[b] > 0]
    r3 = [i for i in ids if P[i] > t3]
    f = lambda hs: sorted(f"L{l}H{h}" for l, h in hs)
    return dict(R1=f(r1), R2=f(r2), R3=f(r3), thr_R1=t1, thr_R3=t3,
                S={f"L{l}H{h}": S[(l, h)] for l, h in S},
                S_argmax={f"L{b[0]}H{b[1]}": f"L{a[0]}H{a[1]}" for b, a in arg.items()},
                P={f"L{l}H{h}": P[(l, h)] for l, h in P})


def do_predict():
    res = {c: predict(c) for c in CORPORA}
    json.dump(res, open(f"{OUT}/predictions.json", "w"), indent=1)
    for c in CORPORA:
        r = res[c]
        print(c, "R1", r["R1"], "R2", r["R2"], "R3", r["R3"])


def do_stats():
    for c in CORPORA:
        W = load(c)
        row = []
        for l in range(L):
            att = np.mean([fro(W[f"W_{n}.{l}"]) for n in "qkvo"])
            mlp = np.mean([fro(W[f"W_fc.{l}"]), fro(W[f"W_proj.{l}"])])
            row.append(f"L{l}: attn {att:.2f} mlp {mlp:.2f} ln1 {W[f'ln1_w.{l}'].mean():.3f} ln2 {W[f'ln2_w.{l}'].mean():.3f}")
        print(c, " | ".join(row))


def prf(pred, truth):
    pred, truth = set(pred), set(truth)
    tp = len(pred & truth)
    p = tp / len(pred) if pred else float("nan")
    r = tp / len(truth) if truth else float("nan")
    f = 2 * p * r / (p + r) if tp else 0.0
    return tp, len(pred), len(truth), p, r, f


def do_score():
    pred = json.load(open(f"{OUT}/predictions.json"))
    heads = json.load(open(f"{ROOT}/eval/heads.json"))
    scores = json.load(open(f"{ROOT}/eval/scores.json"))
    astra = {}
    for r in scores:
        astra.setdefault(r["corpus"], []).append(sorted(r["copy_named"]))
    lines = []
    pooled = {(rule, c): [0, 0, 0] for rule in ("R1", "R2", "R3", "Astra") for c in (0.2, 0.3)}
    for c in CORPORA:
        hd = heads[[k for k in heads if k.startswith(c + "_")][0]]
        for cut in (0.2, 0.3):
            truth = sorted(h for h in hd if hd[h]["induct"] > cut)
            lines.append(f"{c} cutoff {cut}: truth {truth}")
            cands = {"R1": pred[c]["R1"], "R2": pred[c]["R2"], "R3": pred[c]["R3"]}
            for k, v in enumerate(astra[c]):
                cands[f"Astra_r{k}"] = v
            for rule, p in cands.items():
                tp, np_, nt, pr, rc, f = prf(p, truth)
                lines.append(f"  {rule:9s} pred {p} tp={tp}/{np_} pred, {tp}/{nt} true  P={pr:.2f} R={rc:.2f} F1={f:.2f}")
                key = ("Astra" if rule.startswith("Astra") else rule, cut)
                for a, v in zip(range(3), (tp, np_, nt)):
                    pooled[key][a] += v
    lines.append("POOLED (3 networks; Astra = both runs summed)")
    for (rule, cut), (tp, np_, nt) in pooled.items():
        p = tp / np_ if np_ else float("nan")
        r = tp / nt if nt else float("nan")
        lines.append(f"  {rule:6s} cutoff {cut}: tp={tp} npred={np_} ntrue={nt} P={p:.2f} R={r:.2f}")
    lines.append("POST-HOC (not pre-registered): rank of each truth head (cutoff 0.2) by S among the 18 candidates, 1=highest")
    for c in CORPORA:
        hd = heads[[k for k in heads if k.startswith(c + "_")][0]]
        S = pred[c]["S"]
        order = sorted(S, key=S.get, reverse=True)
        tr = [h for h in hd if hd[h]["induct"] > 0.2]
        lines.append(f"  {c}: " + ", ".join(f"{h}->{order.index(h)+1 if h in order else 'n/a(L0)'} (S={S.get(h, float('nan')):.3f})" for h in tr)
                     + f"; median S={np.median(list(S.values())):.3f} thr={pred[c]['thr_R1']:.3f}; top3 {order[:3]}")
    print("\n".join(lines))
    open(f"{OUT}/score_output.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    {"predict": do_predict, "stats": do_stats, "score": do_score}[sys.argv[1]]()
