"""No-LLM baseline for exp2 (see PREREG.md). Blind stage: processes networks by id only; never reads eval/ or answers.
usage: python -I baseline.py <networks_dir> <out_dir>"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDS = ["246a5b58", "a6a9dfb7", "d88aba69"]
G = 12
BETA = np.linspace(1e-4, 0.02, 1000)
ABAR = np.cumprod(1 - BETA)


def load(d, i):
    z = np.load(f"{d}/{i}.npz", allow_pickle=False)
    return {k: z[k].astype(np.float64) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}


def tfeat(t):
    f = 1.0 / 10000 ** (np.arange(8) / 8)
    return np.concatenate([np.sin(t * f), np.cos(t * f)])


def net(p, x, t):  # x: (n,144)
    inp = np.concatenate([x, np.tile(tfeat(t), (len(x), 1))], 1)
    h1 = np.maximum(inp @ p["W1"] + p["b1"], 0)
    h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
    return h2 @ p["W3"] + p["b3"]


def sample(p, n, rng):
    x = rng.standard_normal((n, 144))
    for t in range(999, -1, -1):
        x0 = np.clip(net(p, x, t), -1, 1)
        if t == 0:
            x = x0
            break
        ab, abp = ABAR[t], ABAR[t - 1]
        a = 1 - BETA[t]
        mu = np.sqrt(abp) * BETA[t] / (1 - ab) * x0 + np.sqrt(a) * (1 - abp) / (1 - ab) * x
        var = BETA[t] * (1 - abp) / (1 - ab)
        x = mu + np.sqrt(var) * rng.standard_normal(x.shape)
    return np.clip(x, -1, 1)


def templates():
    yy, xx = np.mgrid[0:G, 0:G].astype(float)
    cy = cx = 5.5
    r = np.hypot(yy - cy, xx - cx)
    T = {}
    for s in (4, 6, 8, 10):
        h = s / 2
        T[("disc", s)] = (r <= h).astype(float)
        T[("ring", s)] = (np.abs(r - h) <= 0.75).astype(float)
        sq = (np.abs(yy - cy) <= h - 0.5) & (np.abs(xx - cx) <= h - 0.5)
        T[("square", s)] = sq.astype(float)
        inner = (np.abs(yy - cy) <= h - 2) & (np.abs(xx - cx) <= h - 2)
        T[("square_outline", s)] = (sq & ~inner).astype(float)
        top = cy - h + 0.5
        frac = np.clip((yy - top) / max(s - 1, 1), 0, 1)
        halfw = 0.5 + frac * (s - 1) / 2
        tri = (yy >= top - 0.01) & (yy <= top + s - 1 + 0.01) & (np.abs(xx - cx) <= halfw)
        T[("triangle", s)] = tri.astype(float)
        edge = tri & (~((yy < top + s - 1.5) & (np.abs(xx - cx) <= halfw - 1.5)))
        T[("triangle_outline", s)] = edge.astype(float)
        T[("cross", s)] = (((np.abs(yy - cy) <= 1) & (np.abs(xx - cx) <= h)) | ((np.abs(xx - cx) <= 1) & (np.abs(yy - cy) <= h))).astype(float)
        T[("hline", s)] = ((np.abs(yy - cy) <= 1) & (np.abs(xx - cx) <= h)).astype(float)
    return T


FAM = {"disc": "circle", "ring": "circle", "square": "square", "square_outline": "square",
       "triangle": "triangle", "triangle_outline": "triangle", "cross": "other", "hline": "other"}


def decide(img, T, sign=1.0):
    v = (sign * img).ravel()
    best = {}
    for (name, s), tm in T.items():
        m = tm.ravel()
        c = -1.0 if v.std() < 1e-12 else float(np.corrcoef(v, m)[0, 1])
        if name not in best or c > best[name][0]:
            best[name] = (c, s)
    top = max(best, key=lambda k: best[k][0])
    fam = FAM[top]
    return fam, (top, best[top][1]), {k: round(v_[0], 3) for k, v_ in best.items()}


def ascii_img(img, thr=None):
    thr = np.median(img) if thr is None else thr
    lo, hi = img.min(), img.max()
    mid = (lo + hi) / 2 if thr == "mid" else thr
    return "\n".join("".join("#" if v > mid else "." for v in row) for row in img)


def main(nd, od):
    T = templates()
    rng = np.random.default_rng(0)
    out, rep = {}, {}
    fig, ax = plt.subplots(len(IDS), 6, figsize=(15, 7.5))
    for r, i in enumerate(IDS):
        p = load(nd, i)
        ra = p["b3"].reshape(G, G)
        rb = p["W3"].mean(0).reshape(G, G)
        rbn = np.linalg.norm(p["W3"], axis=0).reshape(G, G)
        rc = net(p, np.zeros((1, 144)), 999)[0].reshape(G, G)
        rc2 = net(p, rng.standard_normal((256, 144)), 999).mean(0).reshape(G, G)
        S = sample(p, 64, np.random.default_rng(0))
        mean_s = S.mean(0).reshape(G, G)
        maps = {"R_a b3": ra, "R_b W3 col-mean": rb, "R_c zero-in t=999": rc, "R_c2 noise-mean t=999": rc2,
                "sampler mean (64)": mean_s}
        dec = {}
        for k, m in maps.items():
            fam, top, sc = decide(m, T)
            famf, topf, _ = decide(m, T, -1.0)
            dec[k] = {"family": fam, "template": f"{top[0]}", "size": top[1], "scores": sc, "flipped_sign_family": famf, "flipped_template": topf[0]}
        votes = [dec[k]["family"] for k in ["R_a b3", "R_b W3 col-mean", "R_c zero-in t=999", "R_c2 noise-mean t=999"]]
        maj = max(set(votes), key=votes.count) if votes.count(max(set(votes), key=votes.count)) >= 2 else "no-majority"
        out[i] = {"primary_R_c2": dec["R_c2 noise-mean t=999"]["family"], "majority": maj, "decisions": dec,
                  "col_norm_ascii": ascii_img(rbn)}
        rep[i] = {k: ascii_img(m, "mid") for k, m in maps.items()}
        for c, (k, m) in enumerate(list(maps.items()) + [("W3 col-norm", rbn)]):
            ax[r, c].imshow(m, cmap="gray", interpolation="nearest")
            ax[r, c].set_title(f"{i}\n{k}", fontsize=7)
            ax[r, c].axis("off")
        # sample grid
        f2, a2 = plt.subplots(4, 8, figsize=(10, 5))
        for q, a in enumerate(a2.ravel()):
            a.imshow(S[q].reshape(G, G), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            a.axis("off")
        f2.suptitle(f"{i}: first 32 of 64 samples")
        f2.savefig(f"{od}/figs/samples_{i}.png", dpi=90)
        plt.close(f2)
        np.save(f"{od}/figs/mean_sample_{i}.npy", mean_s)
    fig.tight_layout()
    fig.savefig(f"{od}/figs/readouts.png", dpi=110)
    json.dump({"decisions": out, "ascii": rep}, open(f"{od}/baseline_out.json", "w"), indent=1)
    for i in IDS:
        print(i, "primary:", out[i]["primary_R_c2"], "majority:", out[i]["majority"])
        for k, d in out[i]["decisions"].items():
            print("  ", k, "->", d["family"], d["template"], d["size"], "| flipped:", d["flipped_template"])
        for k, a in rep[i].items():
            print("  ##", k); print("\n".join("     " + l for l in a.split("\n")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
