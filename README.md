# Pet foreground segmentation

An Upgaming internship project: train a lightweight binary segmentation model, compare controlled experiments, measure a CPU resolution tradeoff, and apply it to new pet photographs in a local Streamlit app.

**Measured submission:** three comparable CPU experiments are complete. Experiment C was selected using validation only and is deployed at 256 × 256. Official test foreground IoU is **0.939749** and Dice is **0.968939** on all **3,669 test images**, excluding ambiguous boundary pixels. Training used the same 1,000-image subset for A/B/C; all 736 validation images were retained. [Technical report](docs/TECHNICAL_REPORT.md), [measured results](artifacts/reports/results.md), and [verification record](VERIFICATION.md) contain the evidence and limitations.

The 192/128 resolution optimization reduced latency but failed the predeclared quality constraint, so deployment retains 256. The CPU in-memory pipeline median is **105.42 ms** (p95 **135.90 ms**), excluding disk/decoding, PNG encoding and UI time. Actual browser upload/download QA passed nine scenarios. Candidate understanding and independent modifications remain unverified; see [AI disclosure](AI_USAGE.md) and the [requirement audit](docs/REVIEW_AUDIT.md).

## Run the supplied model

Download [the complete submission ZIP](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/download/v1.0.0/upgaming-petseg-submission.zip) and its [SHA-256 checksum](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/download/v1.0.0/upgaming-petseg-submission.zip.sha256) from the [v1.0.0 release](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/tag/v1.0.0). Extract it into a new folder. It contains source files directly at the root, the trained model, full A/B/C evidence and reports. Complete the Windows CPU setup below from that folder, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit and upload a JPEG/PNG. The included example is `artifacts/qualitative/inputs/chelsea.png`. For a command-line check:

```powershell
.\.venv\Scripts\python.exe -m petseg predict artifacts/qualitative/inputs/chelsea.png --output artifacts/my_prediction
```

The supplied model runs without downloading the training dataset or pretrained weights. The source-only ZIP contains no trained model; use the complete submission ZIP for the ready demo. The `.zip.sha256` file records the archive checksum; `SUBMISSION_INVENTORY.json` inside the complete ZIP records each included file's size and SHA-256. In PowerShell, compare `(Get-FileHash .\upgaming-petseg-submission.zip -Algorithm SHA256).Hash` with `Get-Content .\upgaming-petseg-submission.zip.sha256`.

If you cloned this repository instead, download [selected.pt](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/download/v1.0.0/selected.pt) into `artifacts/final/selected.pt` and verify it against the tracked `selected.pt.sha256`. The clone already includes its matching decision and the small measured artifacts. That is enough for the demo; use the full ZIP to reproduce provenance checks and report generation, which also validate the complete A/B/C checkpoints. Git preserves exact bytes through `.gitattributes`, including the hashed JSON manifests.

The workflow in sections 2–5 is for reproducing experiments in a **separate working copy**. Preserve the supplied frozen results. The final experiments ran locally on CPU; the [Colab notebook instructions](docs/COLAB_HANDOFF.md) describe an optional workflow that was checked offline but never executed on Colab.

## Quick orientation

- `petseg/`: data audit, paired transforms, LR-ASPP model, resumable training, original-resolution evaluation, CPU benchmark, and CLI.
- `configs/`: A/B/C experiments, consistent CPU alternatives, and clearly labeled diagnostics.
- `notebooks/train_colab.ipynb`: notebook calling the same package code.
- `app.py`: thin local Streamlit application.
- `tests/`: offline correctness tests, including deterministic interrupted/resumed training.
- `docs/`: technical report, model card, Georgian interview guide, attribution, and submission checklist.
- `artifacts/`: curated small results, figures and provenance are tracked; final weights are release assets. Other generated content stays ignored.
- `data/`: the downloaded raw dataset, excluded from Git.

## 1. Windows CPU setup

Install Python 3.12 from the official Python distribution if `py` does not work. In PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m petseg env
.\.venv\Scripts\python.exe -m pytest -q
```

Activation is optional; the explicit interpreter avoids PowerShell activation-policy issues. If using the Codex bundled runtime, create the venv with its absolute `python.exe` path instead of `py -3.12`.

The locally verified pair is torch 2.9.1 CPU / torchvision 0.24.1 CPU. These are an official matched pair, deliberately not arbitrary latest versions. See [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/). The notebook preserves a working Colab CUDA installation and records its actual versions. `requirements-local-lock.txt` records this workstation's verified environment; use the CPU requirements for a fresh installation, not a Windows lock file in Colab.

Linux CPU equivalent: use `python3.12 -m venv .venv`, then `.venv/bin/python` in place of the Windows interpreter below.

## 2. Prepare and inspect data

```powershell
.\.venv\Scripts\python.exe -m petseg prepare --download
```

This downloads about 800 MB from the official dataset source, checks Torchvision's archive checksums, reads the actual annotation README, validates masks, audits dimensions and distributions, and writes:

- `artifacts/data/splits.json`: deterministic seed-42, approximately 80/20 breed-stratified train/validation content groups within the official trainval partition; official test retained.
- `audit.json`, `exclusions.json`, `ANNOTATION_README.txt`, and `audit_examples.png`.

Inspect the annotation README and overlays before training. Raw trimap **1 = pet**, **2 = background**, **3 = ambiguous outline**. Map to **1, 0, 255** respectively. Outline pixels are ignored in loss and metrics. Yellow in annotation audit images marks ignored pixels.

Decoded RGB content hashes detect exact duplicates, including equivalent pixels encoded differently. Cross-test matches are removed from trainval; conflicting duplicate breed labels are excluded; duplicate groups stay in the same train/validation split. This does not detect all near-duplicates. A corrupt official test sample aborts preparation rather than silently reducing the test set. All exclusions are recorded. Pixel proportions in the audit explicitly refer to structurally valid pairs before duplicate exclusions. The actual audit found no exclusions, but later visual error analysis found semantically inconsistent official masks. Those test labels and scores were preserved; [error analysis](artifacts/reports/error_analysis.md) identifies the cases.

## 3. Diagnose, then train

Train on a free Colab GPU using the notebook, or use the following commands. Create its source ZIP with `.\.venv\Scripts\python.exe scripts/package_source.py`, then upload `artifacts/upgaming-petseg-source.zip` to Colab. Replace `cpu` with `cuda` only on a runtime with an available CUDA GPU.

```powershell
.\.venv\Scripts\python.exe -m petseg train --config configs/overfit.json --output artifacts/diagnostics/overfit --device cpu
.\.venv\Scripts\python.exe -m petseg train --config configs/smoke.json --output artifacts/diagnostics/smoke --device cpu
```

The tiny overfit diagnostic uses two real images, a frozen pretrained backbone and a higher head learning rate. Inspect the loss history for a substantial decrease and inspect predictions on those same images. These runs diagnose the pipeline and are forbidden from final model selection.

For full experiments with GPU:

```powershell
.\.venv\Scripts\python.exe -m petseg train --config configs/A.json --output artifacts/runs/A --device cuda
.\.venv\Scripts\python.exe -m petseg train --config configs/B.json --output artifacts/runs/B --device cuda
.\.venv\Scripts\python.exe -m petseg train --config configs/C.json --output artifacts/runs/C --device cuda
```

The actual final CPU runs used these commands, each with the default one torch CPU thread. They ran in separate concurrent processes; their elapsed times are not a controlled speed comparison:

```powershell
.\.venv\Scripts\python.exe -m petseg train --config configs/cpu_A.json --output artifacts/runs/A --device cpu
.\.venv\Scripts\python.exe -m petseg train --config configs/cpu_B.json --output artifacts/runs/B --device cpu
.\.venv\Scripts\python.exe -m petseg train --config configs/cpu_C.json --output artifacts/runs/C --device cpu
```

These cap only training to the same 1,000 breed-stratified images and use batch size 2. Validation/test remain full. Do not mix full and reduced runs in one comparison. All final runs completed 12 epochs / 6,000 optimizer steps. Best validation pet IoUs were A **0.802834**, B **0.887705**, C **0.888393**. The C–B difference is too small to establish a repeatable augmentation benefit from one seed.

| Run | Backbone | Augmentation |
|---|---|---|
| A | Frozen, including BatchNorm statistics | None |
| B | Fully trainable | None |
| C | Fully trainable | Paired horizontal flip; image-only color jitter |

Every run starts independently from the same seed and ImageNet-backbone/new-head initialization, uses 256-square RGB inputs, cross-entropy ignoring 255, AdamW, head LR 0.001, backbone LR 0.0001, weight decay 0.0001, and a maximum of 12 epochs. Stop after three non-improving epochs once at least three epochs have completed. Best checkpoint uses validation foreground IoU. Actual run lengths and steps are recorded. A single seed is not a significance study.

If GPU memory is insufficient, create copies of all three configs with a common batch size 4, then 2, and restart comparisons in new directories. Do not change batch size while resuming a run. This explicit restart avoids silently mixing incomparable experiments.

### Persistence and resuming

```powershell
.\.venv\Scripts\python.exe -m petseg train --config configs/B.json --output artifacts/runs/B --device cuda --resume artifacts/runs/B/latest.pt
```

Each epoch saves atomic `best.pt` and `latest.pt` plus SHA-256 sidecars, config, actual training IDs, history CSV/JSON, and environment metadata. Copy the **complete run directory** when moving between machines. Use a persistent Drive run directory in Colab as shown in the notebook; otherwise session termination can lose runtime files. Resumption restores model, optimizer, stopping state, and Python/NumPy/torch/loader RNGs at epoch boundaries. Work since the last completed epoch is repeated. Exact equality across GPU hardware, library versions, and operating systems is not guaranteed. Use the same environment when continuing a run.

## 4. Select a model and measure CPU deployment

For optional Colab reproduction, the notebook exports a ZIP containing `artifacts/`; transfer the full run directories to the separate reproduction copy. Local CPU training already writes these files directly. The dataset is needed for benchmarking and final evaluation, but not for running the supplied demo.

```powershell
.\.venv\Scripts\python.exe -m petseg select
.\.venv\Scripts\python.exe -m petseg benchmark
```

Selection checks all three runs are completed, non-diagnostic and comparable. Highest validation pet IoU wins; exact ties use A, B, C order. The selected checkpoint is copied to `artifacts/final/selected.pt` with its hash and a selection record.

Benchmark runs on local CPU, fixed one torch thread, batch size one, FP32, ten warm-up calls per resolution per measured path and 100 timed calls per candidate, interleaved in seeded random order. It uses up to 100 breed-stratified validation images (seed 42, then sorted by ID), preloaded in memory. It measures model-forward and full in-memory preprocessing/forward/logit-resizing/argmax paths separately; excludes loading, visualization and encoding. Full validation IoU is measured at original annotation resolution.

Choose the smallest of 128/192/256 whose validation pet IoU is no more than 0.02 below 256 and whose median pipeline latency improves over 256. Otherwise retain 256. Same weights and checkpoint size at every resolution. Results and frozen deployment choice are in `artifacts/final/decision.json`. CPU platform information, parameter count, checkpoint size, and p95 latency are included. Do not substitute GPU timing for this local CPU comparison.

## 5. Final test evaluation and measured report

After model and deployment choices are frozen, evaluate the baseline and chosen deployment resolution. The CLI rejects mismatched model/split/decision artifacts.

```powershell
.\.venv\Scripts\python.exe -m petseg evaluate --split test --size 256
$deploymentSize = (.\.venv\Scripts\python.exe -c "import json; print(json.load(open('artifacts/final/decision.json'))['deployment_resolution'])").Trim()
if ($deploymentSize -ne '256') {
    .\.venv\Scripts\python.exe -m petseg evaluate --split test --size $deploymentSize
}
.\.venv\Scripts\python.exe -m petseg report
```

If deployment is 256, evaluate once. Never tune the project in response to final test scores. Validation-only exploration is available through `evaluate --split val --checkpoint <path> --size <size>`.

Metrics accumulate a dataset-level confusion matrix over valid original-resolution pixels. Foreground IoU is primary; also report background IoU, mean IoU, foreground Dice, and secondary pixel accuracy. Undefined denominator -> JSON null, excluded from a mean, never perfect by default. Per-image undefined IoUs are omitted from ranking. Ranked examples are worst, upper median, and best in ascending `(IoU, image ID)` order. These are descriptive examples, not subgroup performance claims.

The supplied evaluation was run once at 256 after the recorded freeze. The report contains actual tables and learning curves. [Observed errors](artifacts/reports/error_analysis.md) distinguish model failures from inconsistent official annotations and explain the deterministic example-selection rules. The test IoU exceeds validation IoU; this experiment does not establish the cause of that gap.

## 6. Run the application or CLI prediction

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m petseg predict "C:\path\to\new-pet.jpg"
```

Open the local URL printed by Streamlit. The app requires the trained selected checkpoint and its matching measured decision file. It caches the model and shares the CLI prediction path. Optional environment overrides: `PETSEG_CHECKPOINT` and `PETSEG_DECISION`. Neither path may point to arbitrary incompatible weights.

Uploads: JPEG/PNG, <=10 MiB and <=20 megapixels. EXIF orientation is respected, grayscale and RGBA convert to RGB (alpha is discarded). Output dimensions match the oriented input. Downloaded PNG mask is 0 background / 255 pet; internal mask is 0/1. No generic confidence score is displayed. `predict --size 192` is an explicit manual-resolution override for exploration; the app always uses the frozen deployment resolution.

## Limitations and responsible interpretation

This is pet foreground semantic segmentation, not instance segmentation or alpha matting. Square resizing distorts aspect ratio, and ignored outlines mean headline IoU does not measure fine boundary quality. Actual failures include near-total pet misses, background leakage into a cushion/carrier, and fragmented prediction on a small dog. Sourced external checks show lost whiskers and small false foreground islands on a no-pet photograph. [Attribution and qualitative outputs](docs/QUALITATIVE_SOURCES.md) accompany the two examples; they are not a new benchmark. Single-seed reduced-data results do not establish general-domain performance.

## Submission

The complete transfer file is `artifacts/upgaming-petseg-submission.zip`; `artifacts/upgaming-petseg-source.zip` is the smaller source-only version. The complete ZIP contains seven final checkpoint payloads (selected plus A/B/C best/latest), configs, histories, measured evaluation/optimization, figures, browser QA evidence, attribution and documentation. Retaining the complete runs lets the report and provenance checks work after extraction. Raw training data, caches, virtual environments and diagnostic weight payloads are excluded.

Rebuild the transfer files after any documentation change with `.\.venv\Scripts\python.exe scripts/package_submission.py`. This checks measured identities and the required evidence, rebuilds the source ZIP and verifies every bundled file against an embedded inventory.

The submission repository is [nikoo-dev/upgaming-pet-segmentation](https://github.com/nikoo-dev/upgaming-pet-segmentation); the [v1.0.0 release](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/tag/v1.0.0) supplies trained weights and the complete verified bundle. Publication was explicitly requested by the candidate. Source and curated small evidence are tracked; raw data, caches, environments and weights remain outside ordinary Git history.

Follow `docs/SUBMISSION_CHECKLIST.md`, read `AI_USAGE.md`, and study `docs/INTERVIEW_KA.md`. Personal review, independent interview preparation and sending these links through the required submission channel remain candidate actions.

Dataset citation and licensing: `docs/DATASET.md`. Project decisions and current measured-status caveats: `docs/TECHNICAL_REPORT.md` and `docs/MODEL_CARD.md`.
