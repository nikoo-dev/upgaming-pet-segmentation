# Observed prediction and annotation errors

This analysis uses the frozen experiment C checkpoint at 256 input resolution. It does not change training, model selection, the deployment choice, official test labels or reported scores. The Codex agent inspected the actual photographs, annotation displays and prediction overlays; the candidate's personal review is still required.

The official test run contains 3,669 images. Saved dataset-level scores are foreground IoU **0.939749**, background IoU **0.968331**, mean IoU **0.954040**, foreground Dice **0.968939**, and pixel accuracy **0.978802**. The confusion matrix includes 575,035,253 valid pixels; 79,778,469 boundary pixels are ignored, or 12.18% of all test pixels. These scores describe valid annotation pixels, not fine fur-edge accuracy.

Model SHA-256: `9a4a8d23e09c29479a6969a9f4fba4533c7998da71b35b82576a07e86ed4ce16`. Decision SHA-256: `db611c75e96f5dd20195aba19663e8345edfde7375e235839ae1dc4db8b7b07c`. The source metric/CSV hashes and exact example IDs are recorded in `error_examples.json` and `error_followup.json`.

## Selection and annotation limitations

The predeclared figure selects the six lowest defined saved per-image IoUs, the upper median and the best, using ascending `(IoU, image ID)` order. See `error_examples.png`. The first six all have saved IoU 0.0, but several annotations visibly contradict the photograph. They cannot all be described as model failures.

| Image ID | Raw label 1 pixels: pet | Raw label 2 pixels: background | Raw label 3 pixels: ignored outline | Observed annotation issue |
|---|---:|---:|---:|---|
| `Egyptian_Mau_20` | 0 | 3,110,400 | 0 | The visible sitting cat is labeled entirely background. The prediction follows much of the cat. |
| `Maine_Coon_268` | 35,195 | 142,420 | 9,885 | The target foreground lies mainly on the right background rather than the visible cat. |
| `Persian_259` | 0 | 165,910 | 0 | The close-up cat is labeled entirely background. |
| `Persian_268` | 22,031 | 48,161 | 12,058 | The target appears inverted: the pet interior is background while surrounding regions are pet. |
| `Persian_273` | 82,128 | 30,646 | 23,901 | The white cat interior is background while surrounding grass is pet. |
| `japanese_chin_199` | 0 | 187,500 | 0 | The visible dog is labeled entirely background. |

The files have valid dimensions and permitted raw label values; a structural integrity audit does not establish semantic annotation correctness. Additional all-background targets are visible in the next saved-ranked examples (`keeshond_7`, `saint_bernard_60`, `saint_bernard_78`, `wheaten_terrier_195`). No annotation was repaired or excluded after viewing test results, and no cleaned test score is reported. The number of examples inspected is too small to estimate the prevalence or overall score impact of annotation defects.

After identifying these inconsistencies, the next fixed fourteen saved-ranking cases, positions 7–20, were inspected. This descriptive follow-up rule and the unchanged saved IoUs are recorded in `error_followup.json`; figures are `error_followup_1.png` through `error_followup_4.png`. No scores were recomputed to choose settings.

## Specific model failures

| Image ID | Saved pet IoU | What is visible in the prediction |
|---|---:|---|
| `British_Shorthair_8` | 0.002972 | A complete dark gray cat is visible, but almost all of it is predicted background; only tiny foreground fragments remain. This is a near-total foreground miss. |
| `Bengal_24` | 0.175598 | The predicted region includes much of the large white furry cushion behind the cat. The cat is present in the mask, but the cushion creates substantial background leakage. |
| `miniature_pinscher_5` | 0.189539 | The small dog occupies a limited part of a hallway photograph. The mask contains disconnected body/leg fragments and misses large portions, including the head. |
| `Bombay_36` | 0.213440 | A cat's head is visible through a carrier opening. The target covers that visible pet region; the prediction also includes a large part of the carrier front and opening. This is background leakage in an occluded-pet example. |

Other inspected failures include `Bombay_75` (0.011711), which misses most of a black cat, and `pomeranian_193` (0.109519), which retains a patch near the head while missing most of the body. These are individual observations. They do not establish breed, dark-fur, small-pet, occlusion or clutter subgroup performance, and they do not prove a causal explanation for each error.

## Median, best and external context

The saved upper-median case, `Abyssinian_33` (0.975994), shows broad agreement on a side-on cat against a red background. The selected best case, `scottish_terrier_78` (1.0), matches the scored pixels of the visible dog on a chair. A score of 1.0 still excludes the ambiguous outline and therefore does not establish a perfect fur mask. These per-image scores also differ from the headline IoU, which comes from one accumulated pixel confusion matrix.

The two independently sourced external examples provide complementary qualitative evidence. Chelsea's face is retained but fine whiskers are lost at the binary contour; the no-pet astronaut image leaves small false foreground fragments. Source rights, dimensions, exact frozen-model identity and observations are in `docs/QUALITATIVE_SOURCES.md` and `artifacts/qualitative/sources.json`. Their foreground pixel fractions are mask occupancy, not accuracy or confidence.

The test foreground IoU is higher than the selected model's validation IoU (0.888393). Different split composition and annotation issues are possible considerations, but this run does not identify the cause of the gap. It is not a reason to tune on test or to claim that every difficult case is solved.
