# Measured experiment results

Generated from saved artifacts. Missing measurements remain pending.

## Training comparisons

| Experiment | Best validation pet IoU | Epochs | Training steps |
|---|---:|---:|---:|
| A | 0.8028 | 12 | 6000 |
| B | 0.8877 | 12 | 6000 |
| C | 0.8884 | 12 | 6000 |

Selected experiment: **C**. Single seed; no statistical significance claim.

## CPU resolution comparison

| Resolution | Validation pet IoU | Forward median / p95 ms | Pipeline median / p95 ms |
|---|---:|---:|---:|
| 256 | 0.8884 | 42.92 / 52.11 | 105.42 / 135.90 |
| 192 | 0.8645 | 28.14 / 34.34 | 92.84 / 119.56 |
| 128 | 0.4619 | 18.95 / 23.22 | 83.45 / 111.72 |

Selected deployment resolution: **256**.

Model parameters: 3218308; checkpoint bytes: 38986449 (same weights at every resolution).

Protocol: FP32 CPU, batch 1; model-only and in-memory preprocessing+forward+resize+argmax; no loading, rendering, or PNG encoding.

Hardware: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD; Windows-11-10.0.26200-SP0. Torch 2.9.1+cpu; 1 CPU thread(s). Timing uses 100 images, 10 warm-up calls per candidate/path, and 100 timed calls per candidate/path.

## Held-out test results

| Resolution | Pet IoU | Background IoU | Mean IoU | Pet Dice | Pixel accuracy |
|---|---:|---:|---:|---:|---:|
| 256 | 0.9397 | 0.9683 | 0.9540 | 0.9689 | 0.9788 |

All reported metrics exclude ambiguous trimap boundary pixels. Inspect ranked_examples.png and per_image.csv for actual errors.

## Interpretation still requiring human review

Inspect the saved overlays and describe observed failures (clutter, occlusion, similar colors, small pets). Do not describe a failure as observed until inspecting the corresponding output.

No automatic accuracy acceptance threshold is imposed. A generated table does not replace an engineering discussion or candidate code review.
