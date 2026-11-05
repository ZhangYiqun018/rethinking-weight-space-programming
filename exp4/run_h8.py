"""H8 robustness: edited vs base vs hand-written open-loop controller."""
import time, sys
import numpy as np, jax, jax.numpy as jnp
from common import *

env, state, ep, st = make_env()
N = 256
def batch(s, n): return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s)
B, E = load_actor("base"), load_actor("edited")
REV, DRV = 2, 1                       # edited actor's own entry-0 sequence: value 2 for 16 steps, then value 1
N_REV = 16
rolls = {"base": make_rollout(env, ep, actor_policy(B)),
         "edited": make_rollout(env, ep, actor_policy(E)),
         "openloop": make_rollout(env, ep, openloop_policy(N_REV, REV, DRV))}
Z = jnp.zeros(679)
def run(name, s, n=N, aux=Z, seed=0):
    oc, ln, _, _ = rolls[name](jax.random.split(jax.random.PRNGKey(seed), n), batch(s, n), aux)
    return summarize(oc, ln)

# per-feature std over rollouts (live steps only), from the base actor and from the edited actor
def feat_std(name):
    oc, ln, _, ob = rolls[name](jax.random.split(jax.random.PRNGKey(123), 64), batch(state, 64), Z)
    ob, ln = np.asarray(ob), np.asarray(ln)
    rows = np.concatenate([ob[i, :ln[i]] for i in range(64)])
    return rows.std(0)
STD = {"base_rollout": feat_std("base"), "edited_rollout": feat_std("edited")}
print({k: (float((v > 0).sum()), float(np.median(v[v > 0]))) for k, v in STD.items()}, flush=True)

R = {"meta": {"episodes_per_setting": N, "openloop": {"reverse_value": REV, "drive_value": DRV, "n_reverse_steps": N_REV},
              "note": "all three policies are (nearly) deterministic given the start state; episodes within a setting differ only by action sampling / obs noise",
              "noise_std_features_nonzero": {k: int((v > 0).sum()) for k, v in STD.items()}}}
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

R["nominal"] = {k: run(k, state) for k in rolls}; log("nominal", {k: v["rate"] for k, v in R["nominal"].items()})

R["shift_x"] = {}
for d in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
    s = shift_x(state, d)
    R["shift_x"][str(d)] = {k: run(k, s) for k in rolls}; log("shift", d, {k: v["rate"] for k, v in R["shift_x"][str(d)].items()})

R["obs_noise"] = {}
for ref, sd in STD.items():
    for sig in [0.01, 0.05, 0.1]:
        aux = jnp.asarray(sig * sd, jnp.float32)
        R["obs_noise"][f"{ref}|{sig}"] = {k: run(k, state, aux=aux) for k in rolls}
        log("noise", ref, sig, {k: v["rate"] for k, v in R["obs_noise"][f"{ref}|{sig}"].items()})

R["wheel_friction_x"] = {}; R["car_mass_x"] = {}
for f in [0.8, 0.9, 1.1, 1.2]:
    R["wheel_friction_x"][str(f)] = {k: run(k, scale_wheel_friction(state, f)) for k in rolls}; log("fric", f, {k: v["rate"] for k, v in R["wheel_friction_x"][str(f)].items()})
    R["car_mass_x"][str(f)] = {k: run(k, scale_car_mass(state, f)) for k in rolls}; log("mass", f, {k: v["rate"] for k, v in R["car_mass_x"][str(f)].items()})

# finer sweeps (16 episodes per point): x shift, open-loop reverse length
R["shift_sweep"] = {}
for d in np.round(np.linspace(-0.6, 0.6, 25), 3):
    R["shift_sweep"][str(d)] = {k: run(k, shift_x(state, float(d)), n=16) for k in ["edited", "openloop"]}
log("sweep", {d: (v["edited"]["success"], v["openloop"]["success"]) for d, v in R["shift_sweep"].items()})
R["openloop_nrev_sweep_nominal"] = {}
for nr in [0, 4, 8, 12, 14, 16, 18, 20, 24, 32]:
    r = make_rollout(env, ep, openloop_policy(nr, REV, DRV))
    oc, ln, _, _ = r(jax.random.split(jax.random.PRNGKey(0), 1), batch(state, 1), Z)
    R["openloop_nrev_sweep_nominal"][str(nr)] = {"outcome": int(oc[0]), "len": int(ln[0])}
log("nrev sweep", R["openloop_nrev_sweep_nominal"])
save("h8_robustness.json", R)
