# Exp4 results (Kinetix `l/h2_car_ramp`)

Cell format: successes/episodes (rate) [Wilson 95% CI] mean episode length. Raw numbers: `results/*.json`. Code: `common.py`, `run_h1.py`, `run_h8.py`, `make_tables.py`.

## Config used

- kinetix-env 3.0.2 (PyPI), jax 0.9.0 CPU; level file `l/h2_car_ramp.json` from the package (static params: size l = 12 polygons, 4 circles, 6 joints, 2 thrusters; frame_skip 2).
- `make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, auto_reset=False)`, `max_timesteps=256`, `dense_reward_scale=0.2` (level file has 0.1; 0.2 is the value in `prompt.txt`; the dense term does not enter success/red).
- Observation clipped to [-10, 10], non-finite -> 0 (as in `prompt.txt`). Actions: 6 entries sampled independently from softmax over groups [3,3,3,3,2,2] of the 16 logits.
- Success = `info["GoalR"]`; red = reward < -0.5; timeout = 256 steps.
- Evidence the setup matches: observation size 679 and 16 logits (checked in code); initial positions of all shapes equal the values in `prompt.txt` (body 2.8/0.725, green wheel 2.537/0.735, wheel 13 3.107/0.75, blue box 0.232/0.762, red 1.622/0.402); edited actor mean episode length 73.0 steps, equal to `oneshot_result.json` `search.length` = 73.0; base actor timeouts of 256 steps, equal to `base_result.json`.
- Not matched / not checked: the authors' PPO env wrapper (not in the package) -- clipping and nan handling follow the prompt text; the authors' random streams.

## H1 and determinism

| | episodes | success | red | timeout | mean len | argmax episode |
|---|---|---|---|---|---|---|
| base | 256 | 0 [0.0, 1.5]% | 0 | 256 | 256 | [0, 256] (outcome, len) |
| edited | 1024 | 1024 [99.6, 100.0]% | 0 | 0 | 73 | [1, 73] |

(argmax outcome: 1 success, 0 timeout, -1 red.)

Determinism (edited, 1024 episodes):

| quantity | count |
|---|---|
| reset state identical for different PRNG keys | True |
| initial observation identical across episodes | True |
| distinct entry-0 (motor) action sequences | 1 |
| distinct all-6-entry action sequences (entries 1-5 have no effect on the scene) | 1024 |
| distinct observation-trajectory hashes | 2 (1020 + 4 episodes; max abs difference 1.0e-4 between the two, first at step 2, same motor actions) |
| episode lengths | [73] |

Base actor (256 episodes): distinct entry-0 sequences 256, all end by timeout.
Edited actor logit margin on one trajectory (73 steps): entry-0 max-minus-second logit min 10.5, median 583, max 622 (all logits finite). Entries 1-5 margins: median 0.8-1.4 (unaffected by the edit's controller).

Entry-0 action sequence of the edited actor (all 1024 episodes): value 2 for steps 0-15, then value 1 for steps 16-72 (episode ends at step 73).

## H8 robustness

256 episodes per setting, same start state per setting (except noise/sampling). Open-loop controller: entry 0 = 2 for 16 steps, then 1 (copied from the edited actor's own trajectory); other entries 0. The open-loop controller ignores the observation and is deterministic, so its 256 episodes within a setting are identical.


### Nominal

| setting | base | edited | open-loop |
|---|---|---|---|
| nominal | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |


### (a) car start x shifted by d (body and both wheels; scene units; car body is 0.66 wide, starts at x=2.8)

| d | base | edited | open-loop |
|---|---|---|---|
| -0.5 | 0/256 (0.0%) [0.0, 1.5] len 197 | 256/256 (100.0%) [98.5, 100.0] len 80 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| -0.25 | 0/256 (0.0%) [0.0, 1.5] len 223 | 256/256 (100.0%) [98.5, 100.0] len 74 | 256/256 (100.0%) [98.5, 100.0] len 76 |
| -0.1 | 0/256 (0.0%) [0.0, 1.5] len 251 | 256/256 (100.0%) [98.5, 100.0] len 71 | 256/256 (100.0%) [98.5, 100.0] len 71 |
| 0.1 | 0/256 (0.0%) [0.0, 1.5] len 206 | 256/256 (100.0%) [98.5, 100.0] len 77 | 256/256 (100.0%) [98.5, 100.0] len 78 |
| 0.25 | 0/256 (0.0%) [0.0, 1.5] len 60 | 256/256 (100.0%) [98.5, 100.0] len 76 | 256/256 (100.0%) [98.5, 100.0] len 79 |
| 0.5 | 0/256 (0.0%) [0.0, 1.5] len 159 | 256/256 (100.0%) [98.5, 100.0] len 78 | 256/256 (100.0%) [98.5, 100.0] len 76 |


### (b) Gaussian observation noise, sigma x per-feature std; std taken over 64 rollouts of the named actor (live steps only); noise redrawn each step

| std source\|sigma | base | edited | open-loop |
|---|---|---|---|
| base_rollout|0.01 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| base_rollout|0.05 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| base_rollout|0.1 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| edited_rollout|0.01 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| edited_rollout|0.05 | 0/256 (0.0%) [0.0, 1.5] len 256 | 253/256 (98.8%) [96.6, 99.6] len 72 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| edited_rollout|0.1 | 0/256 (0.0%) [0.0, 1.5] len 250 | 252/256 (98.4%) [96.1, 99.4] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |


### (c) wheel friction multiplier (both wheels)

| x | base | edited | open-loop |
|---|---|---|---|
| 0.8 | 0/256 (0.0%) [0.0, 1.5] len 233 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 0.9 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 1.1 | 0/256 (0.0%) [0.0, 1.5] len 42 | 256/256 (100.0%) [98.5, 100.0] len 71 | 256/256 (100.0%) [98.5, 100.0] len 71 |
| 1.2 | 0/256 (0.0%) [0.0, 1.5] len 30 | 256/256 (100.0%) [98.5, 100.0] len 72 | 256/256 (100.0%) [98.5, 100.0] len 72 |


### (c) car mass multiplier (body and both wheels, inverse mass and inverse inertia scaled)

| x | base | edited | open-loop |
|---|---|---|---|
| 0.8 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 0.9 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 1.1 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 1.2 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |


### Fine x-shift sweep (16 episodes per point; successes out of 16)

| d | edited | open-loop |
|---|---|---|
| -0.6 | 16 | 0 |
| -0.55 | 0 | 0 |
| -0.5 | 16 | 16 |
| -0.45 | 16 | 16 |
| -0.4 | 16 | 16 |
| -0.35 | 16 | 16 |
| -0.3 | 16 | 16 |
| -0.25 | 16 | 16 |
| -0.2 | 16 | 16 |
| -0.15 | 16 | 16 |
| -0.1 | 16 | 16 |
| -0.05 | 16 | 16 |
| 0.0 | 16 | 16 |
| 0.05 | 16 | 16 |
| 0.1 | 16 | 16 |
| 0.15 | 16 | 16 |
| 0.2 | 16 | 16 |
| 0.25 | 16 | 16 |
| 0.3 | 16 | 16 |
| 0.35 | 16 | 16 |
| 0.4 | 16 | 16 |
| 0.45 | 16 | 16 |
| 0.5 | 16 | 16 |
| 0.55 | 16 | 16 |
| 0.6 | 16 | 16 |


### Open-loop: reverse steps N at nominal start (outcome 1 success, -1 red, 0 timeout)

| N | outcome | length |
|---|---|---|
| 0 | -1 | 32 |
| 4 | -1 | 43 |
| 8 | 1 | 61 |
| 12 | 1 | 74 |
| 14 | 1 | 79 |
| 16 | 1 | 73 |
| 18 | 1 | 72 |
| 20 | 1 | 71 |
| 24 | 1 | 77 |
| 32 | 1 | 86 |

## Supplement

Uniform scaling of body+wheel mass (table (c) above) leaves the trajectory bit-identical to nominal (checked: body x at step 40 equal to 4 decimals; wheel friction x1.2 changes it). The only dynamic bodies that move are the car body and two wheels, and the engine's motor/joint solver is impulse-based with motor power not binding, so a uniform mass scale cancels. Body-only mass is tested below (wheel mass unchanged).


### Body-only mass multiplier (256 episodes)

| x | base | edited | open-loop |
|---|---|---|---|
| 0.5 | 0/256 (0.0%) [0.0, 1.5] len 165 | 256/256 (100.0%) [98.5, 100.0] len 74 | 256/256 (100.0%) [98.5, 100.0] len 74 |
| 0.8 | 0/256 (0.0%) [0.0, 1.5] len 68 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 0.9 | 0/256 (0.0%) [0.0, 1.5] len 65 | 256/256 (100.0%) [98.5, 100.0] len 72 | 256/256 (100.0%) [98.5, 100.0] len 72 |
| 1.1 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 1.2 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 71 | 256/256 (100.0%) [98.5, 100.0] len 71 |
| 2 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 4 | 0/256 (0.0%) [0.0, 1.5] len 218 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 74 |


### Wheel friction, wider range (256 episodes)

| x | base | edited | open-loop |
|---|---|---|---|
| 0.25 | 0/256 (0.0%) [0.0, 1.5] len 256 | 0/256 (0.0%) [0.0, 1.5] len 73 | 0/256 (0.0%) [0.0, 1.5] len 76 |
| 0.5 | 0/256 (0.0%) [0.0, 1.5] len 256 | 0/256 (0.0%) [0.0, 1.5] len 71 | 0/256 (0.0%) [0.0, 1.5] len 74 |
| 2 | 250/256 (97.7%) [95.0, 98.9] len 35 | 256/256 (100.0%) [98.5, 100.0] len 135 | 0/256 (0.0%) [0.0, 1.5] len 71 |
| 4 | 256/256 (100.0%) [98.5, 100.0] len 33 | 0/256 (0.0%) [0.0, 1.5] len 77 | 0/256 (0.0%) [0.0, 1.5] len 76 |


### Uniform car mass, wider range (256 episodes)

| x | base | edited | open-loop |
|---|---|---|---|
| 0.5 | 0/256 (0.0%) [0.0, 1.5] len 228 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 2 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |
| 4 | 0/256 (0.0%) [0.0, 1.5] len 256 | 256/256 (100.0%) [98.5, 100.0] len 73 | 256/256 (100.0%) [98.5, 100.0] len 73 |


### Fine wheel-friction sweep (16 episodes per point; successes out of 16)

| friction x | base | edited | open-loop |
|---|---|---|---|
| 0.5 | 0 | 0 | 0 |
| 0.6 | 0 | 0 | 0 |
| 0.7 | 0 | 16 | 16 |
| 0.75 | 0 | 16 | 16 |
| 0.8 | 0 | 16 | 16 |
| 1.0 | 0 | 16 | 16 |
| 1.2 | 0 | 16 | 16 |
| 1.3 | 0 | 16 | 16 |
| 1.4 | 0 | 16 | 16 |
| 1.5 | 0 | 16 | 16 |
| 1.6 | 0 | 16 | 16 |
| 1.8 | 0 | 16 | 16 |
| 2.0 | 16 | 16 | 0 |
| 2.5 | 16 | 0 | 0 |
| 3 | 16 | 0 | 0 |
| 3.5 | 16 | 0 | 0 |
| 4 | 16 | 0 | 0 |
