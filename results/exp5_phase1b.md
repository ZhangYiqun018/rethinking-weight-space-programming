# Exp5 (MinAtar) — phase 1b results tables

## Table A — Astra network: authors' eval vs our re-eval (1,000 episodes each)

Rows are the Astra-written network (`hand.npz`) per game; diff = ours − authors; diff/SE = diff in units of the combined SE sqrt(se1² + se2²).

| game | authors mean ± se | ours mean ± se | diff | diff (combined SE) |
|---|---|---|---|---|
| breakout | 517.4 ± 7.4 | 500.1 ± 7.8 | -17.3 | -1.6 |
| seaquest | 53.1 ± 1.4 | 53.7 ± 1.4 | 0.6 | 0.3 |
| asterix | 53.2 ± 1.0 | 54.3 ± 1.0 | 1.1 | 0.8 |

## Table B — authors' PPO seeds, per game and budget

Rows are the authors' PPO evaluations per game and training budget; best/median/worst are over per-seed mean returns; blog seed = the seed the blog showed (the best one).

| game | budget | seeds | best | median | worst | blog seed |
|---|---|---|---|---|---|---|
| breakout | 100M | 10 | 15.8 | 6.6 | 5.6 | ppo100_s2 |
| breakout | 500M | 3 | 25.3 | 19.8 | 12.2 | ppo500_s0 |
| seaquest | 100M | 3 | 5.6 | 4.4 | 4.4 | ppo100_s2 |
| seaquest | 500M | 3 | 88.9 | 7.5 | 4.4 | ppo500_s2 |
| asterix | 100M | 10 | 25.8 | 17.1 | 13.2 | ppo100_s6 |
| asterix | 500M | 3 | 55.2 | 46.9 | 44.7 | ppo500_s1 |

## Table C — our reference policies (1,000 episodes each)

Rows are our reference-policy evaluations on official MinAtar 1.0.15; cut = share of episodes truncated at 10,000 steps.

| game | policy | mean ± se | mean episode length | cut at 10,000 steps |
|---|---|---|---|---|
| breakout | random | 0.5 ± 0.0 | 11.2 | 0.0% |
| breakout | heuristic | 13.2 ± 0.3 | 1656.7 | 8.6% |
| breakout | heuristic_keepdir | 16.8 ± 0.6 | 206.6 | 0.0% |
| breakout | heuristic_sim | 546.4 ± 9.5 | 6627.9 | 42.6% |
| seaquest | random | 0.1 ± 0.0 | 31.5 | 0.0% |
| asterix | random | 0.5 ± 0.0 | 72.1 | 0.0% |
