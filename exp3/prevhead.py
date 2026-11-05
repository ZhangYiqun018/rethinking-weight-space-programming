"""Exp3 part 2 (see PREREG2.md). Usage: prevhead.py predict | score.  `predict` never opens eval/ or runs/.  Run with python -I."""
import json, sys
import numpy as np

from pathlib import Path
OUT = str(Path(__file__).resolve().parent)
ROOT = str(Path(OUT).parent / "upstream/lm_understand-logs/exp3_language")
CORPORA = ["python", "romaji", "english"]
L, H, HD = 4, 6, 64
Z = 3 * 1.4826


def load(c):
    z = np.load(f"{ROOT}/networks/{c}/blocks.npz", allow_pickle=False)
    return {k: z[k].astype(np.float64) for k in z.files}


def head_mats(W):
    out = {}
    for l in range(L):
        g = W[f"ln1_w.{l}"]
        for h in range(H):
            s = slice(h * HD, (h + 1) * HD)
            q, k, v = (W[f"W_{n}.{l}"][s] * g for n in "qkv")
            out[(l, h)] = dict(qk=q.T @ k, ov=W[f"W_o.{l}"][:, s] @ v)
    return out


def fro(a):
    return np.linalg.norm(a)


def thr(x):
    x = np.asarray(x)
    m = np.median(x)
    return m + Z * np.median(np.abs(x - m))


def name(i):
    return f"L{i[0]}H{i[1]}"


def predict(c):
    M = head_mats(load(c))
    l0 = [(0, h) for h in range(H)]
    later = [(l, h) for l in range(1, L) for h in range(H)]
    kc = {(a, b): fro(M[b]["qk"] @ M[a]["ov"]) / (fro(M[b]["qk"]) * fro(M[a]["ov"])) for a in l0 for b in later}
    D = {a: float(np.mean([kc[(a, b)] for b in later])) for a in l0}
    F = {}
    R = {}
    for a in l0:
        m = M[a]["qk"]
        F[a] = float(fro((m - m.T) / 2) / fro(m))
        R[a] = float(fro(m) ** 2 / np.linalg.svd(m, compute_uv=False)[0] ** 2)
    picks = {"P1": max(l0, key=lambda a: (D[a], -a[1])),
             "S1": max(l0, key=lambda a: (F[a], -a[1])),
             "S2": min(l0, key=lambda a: (R[a], a[1]))}
    res = dict(D={name(a): D[a] for a in l0}, F={name(a): F[a] for a in l0}, R={name(a): R[a] for a in l0}, picks={}, copy={}, ranking={})
    for rule, a in picks.items():
        S = {b: kc[(a, b)] for b in later}
        t = thr(list(S.values()))
        res["picks"][rule] = name(a)
        res["copy"][rule] = sorted(name(b) for b in later if S[b] > t)
        res["ranking"][rule] = [name(b) for b in sorted(later, key=S.get, reverse=True)]
        res.setdefault("thr", {})[rule] = t
    return res


def prf(pred, truth):
    pred, truth = set(pred), set(truth)
    tp = len(pred & truth)
    return tp, len(pred), len(truth)


def do_score():
    pred = json.load(open(f"{OUT}/predictions2.json"))
    heads = json.load(open(f"{ROOT}/eval/heads.json"))
    scores = json.load(open(f"{ROOT}/eval/scores.json"))
    astra = {}
    for r in scores:
        astra.setdefault(r["corpus"], []).append(sorted(r["copy_named"]))
    lines = []
    pooled = {}
    for c in CORPORA:
        hd = heads[[k for k in heads if k.startswith(c + "_")][0]]
        prev = {h: hd[h]["prev"] for h in hd if h.startswith("L0H")}
        order = sorted(prev, key=prev.get, reverse=True)
        lines.append(f"{c}: layer-0 'prev' scores {{{', '.join(f'{h}:{prev[h]:.3f}' for h in sorted(prev))}}} true prev head = {order[0]}")
        for rule in ("P1", "S1", "S2"):
            p = pred[c]["picks"][rule]
            lines.append(f"  {rule} pick {p}: {'RIGHT' if p == order[0] else 'WRONG'} (true rank {order.index(p)+1}/6)")
        for cut in (0.2, 0.3):
            truth = sorted(h for h in hd if hd[h]["induct"] > cut)
            lines.append(f"  cutoff {cut}: truth induct {truth}")
            cands = {f"C1|{r}": pred[c]["copy"][r] for r in ("P1", "S1", "S2")}
            for k, v in enumerate(astra[c]):
                cands[f"Astra_r{k}"] = v
            for rule, p in cands.items():
                tp, npd, nt = prf(p, truth)
                pr = tp / npd if npd else float("nan")
                lines.append(f"    {rule:9s} pred {p} tp={tp}/{npd} pred, {tp}/{nt} true  P={pr:.2f} R={tp/nt if nt else float('nan'):.2f}")
                key = ("Astra" if rule.startswith("Astra") else rule, cut)
                acc = pooled.setdefault(key, [0, 0, 0])
                for i, v in enumerate((tp, npd, nt)):
                    acc[i] += v
        lines.append("  descriptive: ranks of true copy heads (induct>0.2) by Kc from P1 pick: " + ", ".join(
            f"{h}->{pred[c]['ranking']['P1'].index(h)+1}/18" for h in sorted(hd) if hd[h]["induct"] > 0.2))
    lines.append("POOLED (Astra = both runs summed)")
    for (rule, cut), (tp, npd, nt) in pooled.items():
        lines.append(f"  {rule:7s} cutoff {cut}: tp={tp} npred={npd} ntrue={nt} P={tp/npd if npd else float('nan'):.2f} R={tp/nt:.2f}")
    print("\n".join(lines))
    open(f"{OUT}/score_output2.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        res = {c: predict(c) for c in CORPORA}
        json.dump(res, open(f"{OUT}/predictions2.json", "w"), indent=1)
        for c in CORPORA:
            print(c, res[c]["picks"], res[c]["copy"], "D", {k: round(v, 3) for k, v in res[c]["D"].items()})
    else:
        do_score()
