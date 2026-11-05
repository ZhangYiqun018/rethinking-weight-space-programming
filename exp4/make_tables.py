import json
H=json.load(open('results/h1_determinism.json')); R=json.load(open('results/h8_robustness.json'))
def c(r): return f"{r['success']}/{r['n']} ({100*r['rate']:.1f}%) [{100*r['wilson95'][0]:.1f}, {100*r['wilson95'][1]:.1f}] len {r['mean_len']:.0f}"
def table(title, d, keyname):
    s=f"\n### {title}\n\n| {keyname} | base | edited | open-loop |\n|---|---|---|---|\n"
    for k,v in d.items(): s+=f"| {k} | {c(v['base'])} | {c(v['edited'])} | {c(v['openloop'])} |\n"
    return s
o=[]
o.append("# Exp4 results (Kinetix `l/h2_car_ramp`)\n\nCell format: successes/episodes (rate) [Wilson 95% CI] mean episode length. Raw numbers: `results/*.json`. Code: `common.py`, `run_h1.py`, `run_h8.py`, `make_tables.py`.\n")
o.append(f"""## Config used

- kinetix-env 3.0.2 (PyPI), jax 0.9.0 CPU; level file `l/h2_car_ramp.json` from the package (static params: size l = 12 polygons, 4 circles, 6 joints, 2 thrusters; frame_skip 2).
- `make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, auto_reset=False)`, `max_timesteps=256`, `dense_reward_scale=0.2` (level file has 0.1; 0.2 is the value in `prompt.txt`; the dense term does not enter success/red).
- Observation clipped to [-10, 10], non-finite -> 0 (as in `prompt.txt`). Actions: 6 entries sampled independently from softmax over groups [3,3,3,3,2,2] of the 16 logits.
- Success = `info["GoalR"]`; red = reward < -0.5; timeout = 256 steps.
- Evidence the setup matches: observation size 679 and 16 logits (checked in code); initial positions of all shapes equal the values in `prompt.txt` (body 2.8/0.725, green wheel 2.537/0.735, wheel 13 3.107/0.75, blue box 0.232/0.762, red 1.622/0.402); edited actor mean episode length 73.0 steps, equal to `oneshot_result.json` `search.length` = 73.0; base actor timeouts of 256 steps, equal to `base_result.json`.
- Not matched / not checked: the authors' PPO env wrapper (not in the package) -- clipping and nan handling follow the prompt text; the authors' random streams.

## H1 and determinism

| | episodes | success | red | timeout | mean len | argmax episode |
|---|---|---|---|---|---|---|
| base | 256 | {H['base']['success']} [{100*H['base']['wilson95'][0]:.1f}, {100*H['base']['wilson95'][1]:.1f}]% | {H['base']['red']} | {H['base']['timeout_or_other']} | {H['base']['mean_len']:.0f} | {H['base_argmax_outcome']} (outcome, len) |
| edited | 1024 | {H['edited']['success']} [{100*H['edited']['wilson95'][0]:.1f}, {100*H['edited']['wilson95'][1]:.1f}]% | {H['edited']['red']} | {H['edited']['timeout_or_other']} | {H['edited']['mean_len']:.0f} | {H['edited_argmax_outcome']} |

(argmax outcome: 1 success, 0 timeout, -1 red.)

Determinism (edited, 1024 episodes):

| quantity | count |
|---|---|
| reset state identical for different PRNG keys | {H['reset_identical_for_different_keys']} |
| initial observation identical across episodes | {H['edited']['initial_obs_identical_across_episodes']} |
| distinct entry-0 (motor) action sequences | {H['edited']['distinct_action_sequences_entry0']} |
| distinct all-6-entry action sequences (entries 1-5 have no effect on the scene) | {H['edited']['distinct_action_sequences_all6']} |
| distinct observation-trajectory hashes | {H['edited']['distinct_state_trajectories_obs_hash']} (1020 + 4 episodes; max abs difference 1.0e-4 between the two, first at step 2, same motor actions) |
| episode lengths | {H['edited']['episode_lengths_distinct']} |

Base actor (256 episodes): distinct entry-0 sequences {H['base']['distinct_action_sequences_entry0']}, all end by timeout.
Edited actor logit margin on one trajectory (73 steps): entry-0 max-minus-second logit min 10.5, median 583, max 622 (all logits finite). Entries 1-5 margins: median 0.8-1.4 (unaffected by the edit's controller).

Entry-0 action sequence of the edited actor (all 1024 episodes): value 2 for steps 0-15, then value 1 for steps 16-72 (episode ends at step 73).
""")
m=R['meta']
o.append(f"## H8 robustness\n\n256 episodes per setting, same start state per setting (except noise/sampling). Open-loop controller: entry 0 = {m['openloop']['reverse_value']} for {m['openloop']['n_reverse_steps']} steps, then {m['openloop']['drive_value']} (copied from the edited actor's own trajectory); other entries 0. The open-loop controller ignores the observation and is deterministic, so its 256 episodes within a setting are identical.\n")
o.append(table("Nominal", {"nominal":R['nominal']},"setting"))
o.append(table("(a) car start x shifted by d (body and both wheels; scene units; car body is 0.66 wide, starts at x=2.8)", R['shift_x'], "d"))
o.append(table("(b) Gaussian observation noise, sigma x per-feature std; std taken over 64 rollouts of the named actor (live steps only); noise redrawn each step", R['obs_noise'], "std source\\|sigma"))
o.append(table("(c) wheel friction multiplier (both wheels)", R['wheel_friction_x'], "x"))
o.append(table("(c) car mass multiplier (body and both wheels, inverse mass and inverse inertia scaled)", R['car_mass_x'], "x"))
s="\n### Fine x-shift sweep (16 episodes per point; successes out of 16)\n\n| d | edited | open-loop |\n|---|---|---|\n"
for k,v in R['shift_sweep'].items(): s+=f"| {k} | {v['edited']['success']} | {v['openloop']['success']} |\n"
o.append(s)
s="\n### Open-loop: reverse steps N at nominal start (outcome 1 success, -1 red, 0 timeout)\n\n| N | outcome | length |\n|---|---|---|\n"
for k,v in R['openloop_nrev_sweep_nominal'].items(): s+=f"| {k} | {v['outcome']} | {v['len']} |\n"
o.append(s)
open('RESULTS.md','w').write("\n".join(o))

B=json.load(open('results/h8b_supplement.json')); C=json.load(open('results/h8c_friction_sweep.json'))
x=["\n## Supplement\n",
"Uniform scaling of body+wheel mass (table (c) above) leaves the trajectory bit-identical to nominal (checked: body x at step 40 equal to 4 decimals; wheel friction x1.2 changes it). The only dynamic bodies that move are the car body and two wheels, and the engine's motor/joint solver is impulse-based with motor power not binding, so a uniform mass scale cancels. Body-only mass is tested below (wheel mass unchanged).\n"]
x.append(table("Body-only mass multiplier (256 episodes)", B['body_mass_x'], "x"))
x.append(table("Wheel friction, wider range (256 episodes)", B['wheel_friction_wide_x'], "x"))
x.append(table("Uniform car mass, wider range (256 episodes)", B['car_mass_uniform_wide_x'], "x"))
s="\n### Fine wheel-friction sweep (16 episodes per point; successes out of 16)\n\n| friction x | base | edited | open-loop |\n|---|---|---|---|\n"
for k,v in C.items(): s+=f"| {k} | {v['base']['success']} | {v['edited']['success']} | {v['openloop']['success']} |\n"
x.append(s)
open('RESULTS.md','a').write("\n".join(x))
