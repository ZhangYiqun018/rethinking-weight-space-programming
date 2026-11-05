# Validation gate

2,000 games per pair; seed 31337; exactly 1,000 games per color.
The unreleased ppo_ref helper prevents bitwise PRNG replication; the prescribed tolerance is used.

| Astra | PPO | Published | Measured | 95% Wilson | Difference (pp) | Pass |
|---|---|---:|---:|---|---:|---|
| b0 | s0 | 42.6% | 42.10% | 39.95%-44.28% | -0.50 | True |
| b0 | s1 | 26.3% | 27.60% | 25.69%-29.60% | +1.30 | True |
| b0 | s2 | 36.3% | 34.40% | 32.35%-36.51% | -1.90 | True |
| b50 | s0 | 74.2% | 75.50% | 73.57%-77.33% | +1.30 | True |
| b50 | s1 | 70.0% | 69.45% | 67.40%-71.43% | -0.55 | True |
| b50 | s2 | 78.5% | 80.95% | 79.17%-82.61% | +2.45 | True |

Gate: PASS.

Engine: pgx go v1, 9x9, komi 7.5, two-pass/repetition/162-move termination. Input is interleaved own/opponent/White-to-move; row-vector affine layers, six ReLUs; illegal actions have negative-infinite logits; categorical sampling at temperature one. Finished states are frozen and rewards are indexed by color, not randomized pgx player ID.
