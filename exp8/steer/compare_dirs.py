import sys, json, numpy as np, torch
from safetensors.numpy import load_file
from safetensors import safe_open
base_dir, edited_dir, dirs = sys.argv[1:4]
d = np.load(dirs, allow_pickle=False)
def load(dirn, name):
    import glob
    for f in glob.glob(dirn + "/*.safetensors"):
        with safe_open(f, "pt") as h:
            if name in h.keys(): return h.get_tensor(name).float().numpy()
for L in (15, 19):
    name = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    W0, W1 = load(base_dir, name), load(edited_dir, name)
    dW = W1 - W0
    J = np.where(np.abs(dW).sum(0) > 0)[0]
    U, S, _ = np.linalg.svd(dW[:, J], full_matrices=False)
    u = U[:, 0]
    v = d[f"v{L}"]; top = d[f"top{L}"]
    rank = np.argsort(np.argsort(-d[f"score{L}"]))  # 0 = best
    print(f"layer {L}: astra cols {len(J)}, sv ratio s2/s1 {S[1]/S[0]:.3f}, |cos(u_astra, v_ours)| = {abs(u @ v):.3f}, "
          f"overlap top64 = {len(set(J) & set(top.tolist()))}/64, median rank of astra cols in our score = {int(np.median(rank[J]))} / {len(rank)}")
    rnd = np.random.default_rng(0).standard_normal((1000, len(v))); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    print(f"   random-direction |cos| p99 = {np.quantile(abs(rnd @ v), .99):.3f}")
