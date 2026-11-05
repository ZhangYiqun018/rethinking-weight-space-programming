# Results

Win rate of the row player; brackets are 95% Wilson intervals. Every measured cell uses 2,000 games, 1,000 per color. Unmeasured cells are marked --. Astra/PPO cells reuse the validation matches.

| Player | Random | PPO s0 | PPO s1 | PPO s2 | Astra b0 | Astra b50 |
|---|---|---|---|---|---|---|
| R0 | 85.75% [84.15%, 87.21%] | 14.75% [13.26%, 16.37%] | 10.85% [9.56%, 12.29%] | 13.25% [11.83%, 14.81%] | 0.75% [0.46%, 1.23%] | 0.15% [0.05%, 0.44%] |
| R1 | 99.70% [99.35%, 99.86%] | 74.10% [72.14%, 75.97%] | 61.10% [58.94%, 63.21%] | 69.25% [67.19%, 71.23%] | 9.85% [8.62%, 11.23%] | 4.90% [4.04%, 5.94%] |
| b0 | 99.65% [99.28%, 99.83%] | 42.10% [39.95%, 44.28%] | 27.60% [25.69%, 29.60%] | 34.40% [32.35%, 36.51%] | -- | -- |
| b50 | 99.90% [99.64%, 99.97%] | 75.50% [73.57%, 77.33%] | 69.45% [67.40%, 71.43%] | 80.95% [79.17%, 82.61%] | -- | -- |

Additional preregistered matchups (also in raw JSON):

- R0 vs Astra b10: 1.80% [1.30%, 2.48%].
- R0 vs Astra b25: 2.15% [1.60%, 2.88%].
- R1 vs Astra b10: 12.15% [10.79%, 13.65%].
- R1 vs Astra b25: 13.10% [11.69%, 14.65%].

## Preregistered decision

- R0: weakened criterion not met; strongly weakened criterion not met.
- R1: weakened criterion met; strongly weakened criterion not met.

Decision: **weakened**.

R1 wins only 9.85% directly against Astra b0 and 4.90% against Astra b50. The PPO comparison therefore does not establish general playing-strength equivalence: the opponent-specific results are nontransitive. Beating these released PPO policies is achievable with a fixed tactical script, while Astra still decisively beats that script.

This decision concerns the specified fixed-rule baseline comparison. Failure of these two heuristics would not establish that PPO is strong, nor prove that other fixed scripts cannot beat it.

## Caveats

- Hidden small PPO and practice weak/strong opponents were not released and cannot be tested.
- Only two fixed heuristics were tested; no result-driven tuning, training, or sequence search.
- Point-estimate thresholds use the published numbers, as preregistered. Intervals are per-match, not simultaneous; the published estimates also have sampling uncertainty.
- The original shared evaluation helper and environment lockfile were absent. This independently implemented engine passed all six prescribed tolerance checks, but does not claim bitwise reproduction of the original PRNG stream.
- R1 uses exact one-move group/liberty calculations and softmax(10 * score). Its raw-input access is comparable, but its computation is not constrained to the network architecture.
- The 162-move cap, komi 7.5, and repetition-loss rule are pgx's task-specific Go setting.
