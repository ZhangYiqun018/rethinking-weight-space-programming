# PREREG (written BEFORE any result, from README.md + networks/description.txt + one prompt.txt only)

Not opened before scoring: eval/*, runs/*/answer.md, events.jsonl, run.json, training/*, notes/blog_src. (Only filenames were
listed; README.md does contain Astra's one-line answers and says the models make filled shapes of circle/square/triangle
type -- that is known to me, so the *candidate set* {circle, square, triangle} is not a blind choice. Flagged as caveat.)

## Format
npz keys W1(160x64), b1, W2(64x64), b2, W3(64x144), b3(144). Input = 144 pixels x_t (row-major 12x12, -1 black/+1 white)
+ 16 step features: sin(t f_k) (k=0..7), cos(t f_k), f_k = 1/10000^(k/8). h1=relu(xW1+b1), h2=relu(h1W2+b2), x0=h2W3+b3.
npz loaded with allow_pickle=False, python -I.

## Read-outs (fixed, no tuning)
- R_a: b3 reshaped to 12x12.
- R_b: column-mean of W3 (mean over 64 hidden units, per pixel), reshaped 12x12. (Also saved: column L2 norm, not used for decision.)
- R_c: x0_hat at t=999 for the all-zero pixel input (step features included), reshaped, no clipping.
- R_c2: mean of x0_hat at t=999 over 256 pure-noise inputs x_999 ~ N(0, I) (seed 0).
- Ground truth (not a read-out): run DDPM sampler, 64 samples/network, seed 0, mean sample.
  Sampler: abar_t=cumprod(1-beta_t), beta linspace(1e-4,0.02,1000); x_999~N(0,I); each step x0=clip(net(x_t,t),-1,1);
  posterior mean mu = [sqrt(abar_{t-1}) beta_t/(1-abar_t)] x0 + [sqrt(alpha_t)(1-abar_{t-1})/(1-abar_t)] x_t;
  variance beta_t(1-abar_{t-1})/(1-abar_t) for t>0; at t=0 output = x0 (abar_{-1}=1). Final output clipped.

## Decision rule (same for every read-out and network)
Templates on the 12x12 grid, centred at (5.5,5.5), values 0/1, sizes (diameter/side/height) s in {4,6,8,10}:
disc (filled), ring (|r-s/2|<=0.75), square (filled), square_outline (1-px-ish border, width 1.5), triangle (filled, apex up,
row-wise width growing linearly from 1 at top to s at bottom), triangle_outline (edges only), cross (plus, arm thickness 2), hline (2 thick).
Score = Pearson correlation between the read-out and the template (max over s). Per read-out, answer = family with best score, where
family(disc,ring)=circle, family(square,square_outline)=square, family(triangle,triangle_outline)=triangle; cross/hline are
distractors (if they win, answer = "other"). Fill vs outline = which variant of the winning family scores higher.
Sign: read-out used as is (positive correlation only); R_a/R_b also reported with sign flipped, flagged but the primary
decision uses the as-is sign. Final script answer per network: the primary read-out is R_c2 if it exists, with R_a, R_b, R_c
reported separately; a majority across the four is also reported. (Chosen now, not after looking.)
Note: shapes have random position, so mean-type read-outs are blurry centred blobs; templates are centred only. That
limits discrimination by design; accepted.
