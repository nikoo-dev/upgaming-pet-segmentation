# Execution and verification record

Final assembly: October 26, 2026 (Asia/Tbilisi). Real training, benchmark, test and browser checks were executed on October 24. This record supersedes earlier pending-status documents; the earlier execution record is retained in the working directory at `artifacts/review/VERIFICATION-before-final.md`.

## Final measured result

| Evidence | Executed result |
|---|---|
| Dataset preparation | 2,944 training-pool / 736 validation / 3,669 official test images; no structural exclusions or exact decoded cross-partition duplicates |
| Comparable training | A/B/C used identical 1,000 training IDs, full validation, seed 42, batch 2, input 256, 12 epochs / 6,000 optimizer steps each |
| Validation selection | A 0.802834416 (best epoch 10), B 0.887705251 (epoch 12), C 0.888393296 (epoch 12); C selected |
| Local CPU comparison | 256/192/128 measured with fixed selected weights; neither smaller input met the 0.02 validation IoU loss limit |
| Deployment | C at 256; forward median/p95 42.92/52.11 ms; in-memory pipeline 105.42/135.90 ms |
| Official test | All 3,669 images evaluated once at 256 after the freeze; pet IoU 0.939748917, background IoU 0.968330888, mean IoU 0.954039902, Dice 0.968938720, pixel accuracy 0.978801757 |
| Error analysis | Six lowest saved per-image IoUs, upper median and best inspected; then fixed ranking positions 7–20 inspected after annotation contradictions were found |
| External photographs | Frozen-model CLI outputs for sourced Chelsea/astronaut images; dimensions, encoding, model/decision hashes and visual outputs checked |
| Trained browser application | Nine real Edge scenarios passed; nine PNG downloads and nine screenshots verified; zero JavaScript page errors |

The CPU is an AMD Ryzen 7 5800H with Radeon Graphics (16 logical CPUs). Python 3.12.14, torch 2.9.1+cpu, torchvision 0.24.1+cpu and NumPy 2.5.2 were recorded. No CUDA GPU or connected Colab runtime was available. No paid compute was used. A/B/C ran concurrently in separate one-thread processes, so their elapsed training times are not a controlled speed comparison.

Timing used FP32, batch 1, one torch CPU thread, 100 shared stratified validation images, ten warm-ups per resolution/path and 100 timed calls per resolution/path. All six combinations were shuffled per image/call. Pipeline timing excludes disk reading/decoding, PNG encoding and UI rendering. The full 736-image validation set supplied resolution quality. Benchmarking completed before other project inference resumed.

The test confusion matrix includes **575,035,253 valid pixels** and excludes **79,778,469 ambiguous boundary pixels** (12.18% of all test pixels). A high score does not measure fine fur-boundary quality. The semantic inconsistencies found in some official annotations were documented without removing or relabeling test images or reporting a cleaned score. Test IoU exceeds validation IoU; no cause is established by this run.

## Artifact identity and holdout policy

The freeze was recorded at **2026-10-24 12:45:14 UTC / 16:45:14 Asia/Tbilisi**, before official test inference. Baseline and deployment both use 256, so there was one final test evaluation. No model, configuration or deployment tuning occurred after test inspection.

| Artifact | SHA-256 |
|---|---|
| Selected C checkpoint | `9a4a8d23e09c29479a6969a9f4fba4533c7998da71b35b82576a07e86ed4ce16` |
| Canonical split manifest | `cfd32d2dd01ea46fdc53b1aeeb1851e204bc78c3ae3f9707aa82411c4f9b74f1` |
| Deployment decision | `db611c75e96f5dd20195aba19663e8345edfde7375e235839ae1dc4db8b7b07c` |
| Selection record | `396d74263d1387470450de1c40740715626b970b6f39c98d012c116b943e8519` |
| Execution protocol | `9e362c8cb90aba1b0c30cc73154ee9dcfc40d7b35f37669687032ee5f398db9f` |

The October 26 independent read-only audit rechecked all five hashes, completed run states, common initialization, selection policy, test-score arithmetic, all 3,669 unique official CSV IDs and browser file hashes. It passed. See [final_identity_audit.json](artifacts/reports/final_identity_audit.json).

## Code and reproduction checks

- Final primary suite: **48 tests passed in 32.75 seconds**. Dependency consistency and compilation passed. Exact commands/output are in [final_code_verification.json](artifacts/reports/final_code_verification.json).
- The final wheel built offline and was installed in the existing second verification environment. `uv pip check` found 55 compatible packages. Python isolated mode imported from site-packages and a random-weight forward returned `[1, 2, 64, 64]`.
- That installed wheel also predicted the frozen Chelsea example. Mask, overlay, composite and metadata matched the primary environment byte for byte. The wheel package-source bytes match the current source. Exact commands, output and hashes are in [second_environment_verification.json](artifacts/reports/second_environment_verification.json). This reused an existing dependency environment; it was not a fresh internet installation.
- All seven notebook code cells compile. Selected exact fragments were checked offline for extraction/reuse, traversal/changed-source rejection, persistent-path read/write probes, loaded-dependency restart handling and export inventory/checksum behavior. **Colab, Google Drive mounting and CUDA training were not executed.**
- Real trained-app browser QA covered pet/no-pet uploads, three PNG outputs per upload, an actual background-color change to `#112233`, invalid image bytes, >10 MiB and >20 megapixel input rejection, and separate missing/corrupt/mismatched-artifact servers. Screenshots were visually inspected and owned server processes were stopped. [qa.json](artifacts/app_qa/qa.json) records actual scenarios and files. Its optional browser harness requires local Node/Playwright/Edge setup; ordinary demo use requires only the Python setup in README.
- The complete bundle was extracted into a new directory and imported from that extracted source using the second environment. **All 48 tests passed in 18.77 seconds**, and all seven notebook code cells compiled. The relocated report reproduced byte for byte with all five evidence flags true. A default frozen-model Chelsea prediction reproduced mask, overlay, composite and metadata byte for byte. All 182 inventory files and exactly seven permitted final weight payloads passed size/SHA-256 checks. The five freeze identities remained intact. [submission_reproduction.json](artifacts/reports/submission_reproduction.json) records the commands and tested executable-source hashes.
- After final documentation updates, the delivery archive is checked against those tested executable-source hashes, its own checksum and every inventory entry, then extracted separately for another report/prediction parity check. The external delivery record is `artifacts/submission_delivery_verification.json`; it is kept beside the archive to avoid a self-referential archive checksum. Raw data, caches, virtual environments, diagnostic weight payloads and common credential signatures were absent from the checked package.

## Diagnostics and observed fixes

Fresh pre-experiment diagnostics used the final canonical manifest. The two-image, 30-step overfit check reduced loss from **0.8003216 to 0.0155660** and reached **0.9922849 IoU on those same training images**. The four-step full-model smoke check had finite loss. They diagnose the pipeline and were never eligible for final model selection. Their summaries accompany the submission; diagnostic weights are omitted.

Concrete fixes covered interrupted terminal completion and supported latest/best/log/status recovery windows, exact run/model/data provenance, original-pixel metric weighting, ignored-pixel counts, EXIF geometry, frozen BatchNorm, balanced timing order, stale report rejection and deterministic UTF-8/LF JSON. Checkpoint and checksum replacements remain separate filesystem operations: interruption between them fails closed. Cross-device or cross-version bitwise training equality is not promised.

A full re-audit reproduced the original Windows CRLF manifest, then canonical LF preparation reproduced the same parsed split and counts. Final runs consistently use the canonical manifest above. A live Linux/Colab regeneration was not performed. Structural audits validate labels/dimensions/decoding, not semantic correctness; observed test-label contradictions are in [error_analysis.md](artifacts/reports/error_analysis.md).

## Publication verification, October 27, 2026

The publication package adds explicit release links and `.gitattributes` to preserve byte-identical hashed artifacts in Git clones. The packaging checks passed. A fresh extraction of this version passed all 48 tests in the existing second environment, reproduced the report and frozen external prediction byte for byte, and retained all five frozen identities. See [publication_reproduction.json](artifacts/reports/publication_reproduction.json). An initial attempt exhausted free space in the system temporary directory on C:; its file-write failures were resolved by running the same tests with temporary files on D:. No model, training, evaluation or deployment setting was changed.

## Remaining personal actions

The technical artifacts and machine checks do not establish candidate understanding or personal review. The candidate still needs to read the code/report, run the demo, explain decisions and complete independent practice tasks. [AI_USAGE.md](AI_USAGE.md) states the assistance honestly; [INTERVIEW_KA.md](docs/INTERVIEW_KA.md) supports that preparation.

On October 27 the candidate explicitly authorized public publication to [nikoo-dev/upgaming-pet-segmentation](https://github.com/nikoo-dev/upgaming-pet-segmentation). The [v1.0.0 release](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/tag/v1.0.0) provides the complete bundle and selected model with checksums. Git tracks curated small evidence and source, preserves exact bytes through `.gitattributes`, and excludes raw data, caches, environments and checkpoint payloads. Sending the repository/release links through the required submission channel remains a candidate action. Optional future experiments require validation-based decisions and a genuinely new holdout for new generalization claims.
