# Exp8: edited model under the blog's protocol (A) vs every configuration

Accuracy % (mean over samples, then problems). Δ = edited-A minus the row, paired over problems, 95% CI.

## math500

| model | config | acc % | Δ (edited-A − row) | 95% CI |
|---|---|---|---|---|
| base | A | 8.3 | +42.4 | [+39.0, +45.8] |
| base | B | 44.2 | +6.4 | [+4.0, +8.8] |
| base | C | 3.0 | +47.6 | [+44.0, +51.2] |
| base | D | 23.1 | +27.6 | [+24.8, +30.4] |
| base | E | 35.4 | +15.3 | [+13.0, +17.5] |
| edited | A | 50.6 | (reference) | |
| edited | B | 44.6 | +6.0 | [+3.5, +8.4] |
| edited | C | 44.4 | +6.3 | [+4.3, +8.2] |
| edited | D | 40.6 | +10.1 | [+7.9, +12.2] |
| edited | E | 42.6 | +8.1 | [+6.0, +10.1] |

## gsm8k

| model | config | acc % | Δ (edited-A − row) | 95% CI |
|---|---|---|---|---|
| base | A | 12.4 | +55.6 | [+53.3, +57.8] |
| base | B | 56.0 | +11.9 | [+9.9, +14.0] |
| base | C | 5.3 | +62.6 | [+60.3, +64.7] |
| base | D | 42.8 | +25.2 | [+23.2, +27.3] |
| base | E | 56.4 | +11.5 | [+9.7, +13.4] |
| edited | A | 67.9 | (reference) | |
| edited | B | 54.4 | +13.6 | [+11.5, +15.7] |
| edited | C | 61.6 | +6.3 | [+4.6, +8.2] |
| edited | D | 56.1 | +11.8 | [+9.8, +13.7] |
| edited | E | 64.8 | +3.1 | [+1.6, +4.7] |

## ifeval

| model | config | acc % | Δ (edited-A − row) | 95% CI |
|---|---|---|---|---|
| base | A | 20.5 | +42.3 | [+38.5, +46.3] |
| base | B | 54.7 | +8.1 | [+4.3, +12.0] |
| base | C | 1.9 | +60.9 | [+57.4, +64.5] |
| base | D | 46.8 | +16.1 | [+12.5, +19.6] |
| base | E | 65.1 | -2.2 | [-5.3, +0.9] |
| edited | A | 62.8 | (reference) | |
| edited | B | 49.8 | +13.0 | [+9.1, +16.6] |
| edited | C | 61.6 | +1.3 | [-1.8, +4.3] |
| edited | D | 38.5 | +24.3 | [+20.6, +27.9] |
| edited | E | 63.3 | -0.5 | [-3.2, +2.6] |

