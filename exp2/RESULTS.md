# Exp 2 no-LLM baseline: results

Order followed: (1) PREREG.md written from README/description/prompt only; (2) baseline.py run blind (networks by id only);
(3) only then answer_key.json, runs/*/answer.md and the blog text were opened for scoring.
Environment: /opt/homebrew/bin/python3 -I (numpy, matplotlib); whole run ~1.4 s CPU, including the 64-sample sampler per net.

## Table (family decision per read-out; truth from answer_key.json)
| Net id | Truth | R_a (b3) | R_b (W3 col-mean) | R_c (zero input, t=999) | R_c2 (noise-mean, t=999) = primary | Majority | Astra |
|---|---|---|---|---|---|---|---|
| 246a5b58 | triangle | triangle | triangle | triangle | **triangle** | triangle OK | triangle (hollow) OK |
| a6a9dfb7 | square | square | circle | circle | **circle** WRONG | circle WRONG | square (outline) OK |
| d88aba69 | circle | circle | circle | circle | **circle** | circle OK | circle (outline) OK |

Script primary: 2/3. Astra: 3/3 on shape name.
Margins for the square net on R_c2: disc 0.879, square 0.856, triangle 0.740 (centred templates, so a ~0.02 gap; weak).
Triangle R_c2: triangle 0.884 vs disc 0.779 / square 0.776. Circle R_c2: disc 0.904 vs square 0.865.
Full score dicts: baseline_out.json. ASCII of every read-out: baseline_out.json ("ascii"), or rerun baseline.py.

## Sampler ground truth (64 samples/net, seed 0)
Mean sample per net is a filled, centred blob (figs/mean_sample_*.npy; figs/samples_<id>.png shows 32 individual samples; figs/readouts.png shows
all read-outs + sampler mean). Triangle mean: wide at bottom, narrow top; square mean: boxy ~6x6; circle mean: round. Sampler mean classified by the same
template rule gives triangle / circle (square net!) / circle: the mean image of a square at random position is itself blob-like and the centred-template
rule confuses it with a disc. So the one error is a limit of the decision rule on position-averaged images, not of the read-out
(R_c2 is nearly identical to the sampler mean in all three nets: one-step denoising of pure noise already shows the shape).
The individual samples in eval/samples/ (opened only in step 3) are filled shapes at random positions, e.g. square sample has a flat-topped box.

## Hollow outline detail
Not visible in any read-out. R_a, R_c, R_c2 and the sampler mean are all filled blobs; ring / outline templates scored lower than
filled ones in every case (e.g. circle net R_c2: disc 0.904 vs ring 0.672; triangle: 0.884 vs 0.699; square: 0.856 vs 0.741). Fill-vs-outline
choice by script = filled for all three, which matches the actual (filled) models and contradicts Astra's "outline" drawings. R_b (W3 col-mean) is noisy
but also blob-like. The blog's claim that outlines are read from "pieces of the shape's edge" in the output layer was not testable here (no per-unit analysis done).

## Caveats
- Knowledge contamination: README (read in step 1 because it is the task spec) names the three shapes and says shapes are filled, so
  the candidate set {circle, square, triangle} was not blind for me. The task asked for README use; Astra got an open question without that list.
- The sampler mean was printed in the same run as the blind read-outs, before the answer key; the decision rule had already been fixed in PREREG.
- Ids -> shapes mapping was not known during step 2; read-outs R_c/R_c2 are a forward pass, which Astra was forbidden to do. So this is not
  a like-for-like comparison with Astra; it shows the forward pass at one fixed noisy input (cheap) suffices to see the shape; b3 alone (no forward pass) got the family right 3/3, by narrow margins, e.g. square net: square 0.826 vs disc 0.787).
- Templates are centred only; shapes have random position/size. A 3-way choice has a 33% chance level; n=3 networks means no statistical claim.
- Cosmetic bug fixed after first run (printed template name was truncated to its first letter); decisions and scores were unchanged by the fix.
- The "majority" column ignores R_a's ties; R_a alone is the best of the cheap, forward-pass-free read-outs (3/3), R_b (W3 col-mean) 2/3.

## Files
baseline.py, PREREG.md, baseline_out.json, figs/readouts.png, figs/samples_<id>.png, figs/mean_sample_<id>.npy
