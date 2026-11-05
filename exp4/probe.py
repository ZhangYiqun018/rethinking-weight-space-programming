import numpy as np, jax, jax.numpy as jnp
from common import *
env, state, ep, st = make_env()
obs = env.get_obs(state)
print("obs size", obs.shape, "action space", env.action_space(ep), "static", st.num_polygons, st.num_circles, st.num_joints, st.frame_skip)
print("poly pos", np.asarray(state.polygon.position[:8]).round(3).tolist())
print("circ pos", np.asarray(state.circle.position[:4]).round(3).tolist(), "radius", np.asarray(state.circle.radius[:4]))
print("poly roles", np.asarray(state.polygon_shape_roles), "circ roles", np.asarray(state.circle_shape_roles))
print("fric", np.asarray(state.circle.friction[:2]), "invmass", np.asarray(state.polygon.inverse_mass[:8]), np.asarray(state.circle.inverse_mass[:4]))
B, E = load_actor("base"), load_actor("edited")
o = clean_obs(obs)
for n, P in [("base", B), ("edited", E)]:
    lg = np.asarray(logits_fn(P, o)); print(n, "finite", np.isfinite(lg).all(), lg.round(2).tolist())
# edited gap on trajectory obs: run edited rollout, 16 eps
roll = make_rollout(env, ep, actor_policy(E))
keys = jax.random.split(jax.random.PRNGKey(0), 16)
outc, ln, acts, obss = roll(keys, jax.tree.map(lambda x: jnp.broadcast_to(x, (16,)+x.shape), state))
print("edited outcome", np.asarray(outc), np.asarray(ln))
O = np.asarray(obss[0, :int(ln[0])]);
lg = np.asarray(jax.vmap(lambda x: logits_fn(E, x))(jnp.asarray(O)))
print("all finite", np.isfinite(lg).all())
gaps=[]
o_=0
for g in GROUPS:
    s=np.sort(lg[:,o_:o_+g],axis=1); gaps.append(s[:,-1]-s[:,-2]); o_+=g
gaps=np.stack(gaps,1); print("max-second gap per entry: min/median/max", gaps.min(0).round(1), np.median(gaps,0).round(1), gaps.max(0).round(1))
print("entry0 actions along traj", np.asarray(acts[0,:int(ln[0]),0]).tolist())
