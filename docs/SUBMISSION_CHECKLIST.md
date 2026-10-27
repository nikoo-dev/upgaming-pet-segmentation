# Submission checklist

Deadline: October 29, 2026, 18:00 Asia/Tbilisi. Aim to package by 15:00.

Final documentation review: October 26, 2026. Actual A/B/C training, validation selection, the CPU optimization attempt, frozen official test evaluation, image-specific error analysis, attributed new-image prediction and trained-app browser QA are complete. All experiments consistently used 1,000 training images with full validation/test. Official test results cover all 3,669 images: pet IoU 0.939749 and pet Dice 0.968939, excluding ambiguous boundary pixels. Evidence and limitations: `docs/REVIEW_AUDIT.md` and `VERIFICATION.md`.

| Assignment requirement | Evidence to inspect | Current status |
|---|---|---|
| Data processing | `artifacts/data/audit.json`, `splits.json`, annotation README, overlay | **Complete:** structural audit and portable split retained; semantic annotation defects documented separately |
| Training/fine-tuning | `artifacts/runs/A`, `B`, `C`; `artifacts/execution_protocol.json` | **Complete:** real 12-epoch/6,000-step runs; diagnostics excluded |
| Comparative experiments | `artifacts/final/selection.json`, histories, tables/curves | **Complete:** validation selected C; common settings/initialization |
| Suitable evaluation | `artifacts/final/freeze.json`; `artifacts/evaluations/test_256/` | **Complete:** all 3,669 test images, original-resolution metrics and matching frozen identities |
| Error analysis | `artifacts/reports/error_analysis.md`, `error_examples.*`, `error_followup*`; `artifacts/qualitative/` | **Complete:** inspected model errors and annotation defects by ID, median/best and external examples; official scores preserved |
| Optimization attempt | `artifacts/final/decision.json` | **Complete:** measured CPU speed/quality; retain 256 because smaller inputs failed quality limit |
| New-image prediction | Attributed inputs/outputs/metadata in `artifacts/qualitative/` | **Complete:** real pet/no-pet checks; no new benchmark claim |
| Working demo | `artifacts/app_qa/qa.json`, screenshots, actual downloads | **Complete:** nine real Edge scenarios and agent visual inspection; candidate review unverified |
| Source/reproduction | Package, tests, configs, notebook, dependencies, README; verification JSON records | **Verified:** 48 tests, build, second-environment installed-wheel prediction parity; optional Colab route unexecuted |
| Trained model | `artifacts/final/selected.pt`, `.sha256`, selection/decision/freeze | **Complete local artifact:** include actual weights/hash in transfer bundle |
| Technical report/model card | `docs/TECHNICAL_REPORT.md`, `docs/MODEL_CARD.md`, generated results | **Complete:** measured training/CPU/test tables, curves, specific errors and limitations; all five generated evidence flags true |
| AI disclosure | `AI_USAGE.md` | Assistance disclosed; candidate adds only actions personally performed |
| Interview ownership | `docs/INTERVIEW_KA.md`, candidate explanations/modifications | **Unverified:** candidate must demonstrate independent understanding |
| Final package | `artifacts/upgaming-petseg-submission.zip`, `.sha256`; embedded inventory; `artifacts/reports/submission_reproduction.json` | **Complete:** extracted source passed 48 tests; report and frozen prediction reproduced exactly; final executable-source hashes verified |

Before delivery, personally inspect actual results/predictions, run the trained app, explain the data/model/loss/metrics path, and complete independent exercises. Update AI disclosure only with actions actually performed. Agent checks establish software execution, not candidate ownership.

The final bundle includes source/configs, dependencies/reproduction instructions, selected weights/hash, audit without raw data, measured comparisons/curves, frozen test results, error analysis, attribution, browser evidence and disclosure. Verify its SHA-256 and embedded inventory. Public publication was explicitly authorized on October 27: submit the [repository](https://github.com/nikoo-dev/upgaming-pet-segmentation) and [release](https://github.com/nikoo-dev/upgaming-pet-segmentation/releases/tag/v1.0.0) links through the required channel. Credentials, caches, raw data and large weights stay outside ordinary Git history.
