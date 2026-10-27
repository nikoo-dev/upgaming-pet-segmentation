# Model card

## Model and artifact

**Model:** Torchvision LR-ASPP with MobileNetV3-Large, binary pet/background semantic segmentation, **3,218,308 parameters**. The backbone started from ImageNet1K V1 weights; the two-class segmentation head was newly initialized.

**Selected run:** experiment **C**, full fine-tuning with paired horizontal flips and image-only color jitter. Best checkpoint: **epoch 12** (1-based), from a 12-epoch / 6,000-step run. Validation foreground IoU: **0.888393** on all 736 validation images. Selection used validation only; C exceeded B by just 0.000688, without a statistical significance claim.

**Artifact:** [download selected.pt](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/download/v1.0.0/selected.pt), **38,986,449 bytes**; save to `artifacts/final/selected.pt`. It includes optimizer and RNG state, so this is not a model-tensors-only file size. Its SHA-256 is:

```text
9a4a8d23e09c29479a6969a9f4fba4533c7998da71b35b82576a07e86ed4ce16
```

The sidecar, [selection record](../artifacts/final/selection.json), [deployment decision](../artifacts/final/decision.json) and [freeze record](../artifacts/final/freeze.json) accompany the model. The decision SHA-256 is `db611c75e96f5dd20195aba19663e8345edfde7375e235839ae1dc4db8b7b07c`. Load using the shared project inference helper, which validates checkpoint metadata and disables automatic weight downloads. Only load trusted project artifacts; loading uses `weights_only=True`.

## Intended use and interface

This is a local educational demonstration for pet photographs: visualizing foreground, inspecting an overlay and replacing the background with a solid color. It does not separate individual pets, classify breeds or estimate alpha transparency.

Input is EXIF-oriented RGB, resized to **256 × 256** for training and deployment and normalized with ImageNet mean/std. Grayscale/RGBA inputs convert to RGB; alpha is discarded. Two-channel logits are resized to the original oriented image dimensions before argmax. Internal predictions are 0 background / 1 pet; downloaded mask PNGs are 0 background / 255 pet. Uploads support JPEG/PNG up to 10 MiB and 20 megapixels. Output dimensions match the oriented input.

The app and default CLI share the same frozen deployment decision. An explicit CLI resolution override is labeled manual exploration and does not represent the frozen deployment result.

## Training data and compute

Oxford-IIIT Pet was split using seed 42, breed stratification and decoded-content grouping into a 2,944-image training pool and 736 validation images within official trainval; all 3,669 official test images were held out. The full structural audit found no exclusions or exact decoded cross-partition duplicates. It does not prove near-duplicate or pretraining independence, and final visual analysis found semantically inconsistent official annotations.

The final CPU experiments used the **same 1,000 breed-stratified training IDs** for A/B/C. The other 1,944 training-pool images were unused. Validation and test were full. Training used batch 2, seed 42, AdamW, head LR 0.001, backbone LR 0.0001 and weight decay 0.0001. The final model ran locally on an AMD Ryzen 7 5800H; no CUDA, Colab or paid compute was used. The notebook is an optional reproduction route checked offline, not an executed GPU experiment.

Manifest SHA-256: `cfd32d2dd01ea46fdc53b1aeeb1851e204bc78c3ae3f9707aa82411c4f9b74f1`. Actual IDs, configs, environments, histories and complete final A/B/C best/latest checkpoints are retained. Dataset source and licensing details are in [DATASET.md](DATASET.md).

## Evaluation and ignored pixels

The model and resolution were frozen on **October 24, 2026 at 12:45:14 UTC / 16:45:14 Asia/Tbilisi**, before official test evaluation. Baseline and deployment are both 256, so the full official test set was evaluated once.

| Official test, 3,669 images | Value |
|---|---:|
| Foreground IoU | 0.939749 |
| Background IoU | 0.968331 |
| Mean IoU | 0.954040 |
| Foreground Dice | 0.968939 |
| Pixel accuracy | 0.978802 |
| Valid scored pixels | 575,035,253 |
| Ignored outline pixels | 79,778,469 |

Scores use one confusion matrix accumulated at original annotation resolution. Raw trimap label 3 maps to **training target 255 = ignore** and is excluded from both loss and every reported metric. This is distinct from **exported PNG 255 = pet**. Ignored pixels are 12.18% of test pixels, so scores do not establish fine-boundary or fur quality. The greater test than validation IoU has no established causal explanation in this experiment.

Official test annotations were not corrected or excluded after inspection, and no cleaned score is reported. [Error analysis](../artifacts/reports/error_analysis.md) documents both annotation contradictions and real model failures, including nearly missing `British_Shorthair_8`, cushion leakage in `Bengal_24`, fragmented small-dog prediction in `miniature_pinscher_5`, and carrier leakage in `Bombay_36`.

## CPU performance and resolution choice

Measurements used Windows, Python 3.12.14, torch 2.9.1+cpu / torchvision 0.24.1+cpu, AMD Ryzen 7 5800H, FP32, batch 1 and one torch thread. The same 100 validation images were preloaded, with 10 warm-ups and 100 timed calls per resolution/path. All six path/resolution combinations were interleaved in seeded random order.

At 256, model-forward latency was **42.92 ms median / 52.11 ms p95**. The in-memory preprocessing + forward + original-size resize + argmax pipeline was **105.42 ms median / 135.90 ms p95**. These exclude image reading/decoding, PNG encoding, browser rendering and UI overhead.

The 192 pipeline median was 92.84 ms, but validation IoU fell to 0.864532, a **0.023861** loss from 256. The 128 IoU was 0.461857. Both failed the predeclared maximum **0.02 absolute IoU loss**, so **256 was retained**. No deployment speedup or checkpoint-size reduction is claimed.

## Limitations and verification scope

The model has single-seed, reduced-data evidence, square-resize distortion and limited domain coverage. It can miss most of a visible pet, merge background objects with foreground and produce disconnected fragments. It has no reliable no-pet rejection: an external astronaut photograph produced small false-foreground patches (0.401% mask occupancy, not an accuracy score). An external cat photograph retained the face but lost thin whiskers. [Qualitative sources and observations](QUALITATIVE_SOURCES.md) are recorded without benchmark claims.

The final code suite passed 48 tests. Real trained-app Edge QA passed nine scenarios with actual upload, color change, downloads and error handling; nine screenshots were visually inspected. A second existing CPU environment reproduced the frozen external prediction exactly. These checks establish exercised behavior and do not guarantee perfect software or segmentation. See [VERIFICATION.md](../VERIFICATION.md) for executed reproduction and packaging checks.

The [README](../README.md) provides setup, demo and reproduction commands. Inference/report review needs the supplied source and artifacts; retraining requires downloading the dataset and backbone weights. Bitwise equality across devices or library versions is not guaranteed. The public repository and release were requested by the candidate. AI assistance is disclosed in [AI_USAGE.md](../AI_USAGE.md); candidate understanding and independent modifications remain unverified.
