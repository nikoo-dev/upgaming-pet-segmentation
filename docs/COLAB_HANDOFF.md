# Optional Colab reproduction and artifact return

The supplied final model was trained locally on CPU with the common 1,000-image A/B/C configuration family. All three runs and validation selection are complete; C was selected, and the measured CPU tradeoff retained input size 256. Frozen official-test evaluation also completed on all 3,669 images (pet IoU 0.939749). Actual execution and final evaluation status are recorded in `VERIFICATION.md` and `docs/REVIEW_AUDIT.md`. Running this notebook is optional for the existing trained application.

The notebook has valid JSON and compilable Python cells, with selected helper fragments checked offline. It has not been executed on Colab by this agent; no CUDA GPU or connected Colab session was available. It is an optional free-GPU reproduction route, not the source of the supplied training results. To reproduce the supplied experiment settings, select the reduced family below. The notebook full-family default defines a separate comparison. CPU and GPU results need not be bitwise identical. Do not replace frozen decisions in response to official test scores.

The supplied official test results are already known. Repeating the same protocol can check reproduction, but that test split is not a fresh holdout for later changes. Select new methods using validation and reserve genuinely new held-out data for a new final generalization claim. Keep reproduction outputs in separate directories so the supplied evidence remains intact.

Use free compute only. Colab does not guarantee GPU access or session lifetime; see its [FAQ](https://research.google.com/colaboratory/faq.html). Keep checkpoints in your own Drive and download the finished evidence.

## Run in Colab

From the root of your cloned or extracted project, create the current source archive:

```powershell
.\.venv\Scripts\python.exe scripts/package_source.py
```

Open `notebooks/train_colab.ipynb` in Colab, choose a free GPU runtime, and upload `artifacts/upgaming-petseg-source.zip` using the Files sidebar. Keep this exact ZIP for any resumed session. Run the seven code cells in order:

| Code cell | Action |
|---|---|
| 1: settings and source | Keep `DEVICE='cuda'`, `CONFIG_FAMILY='full'`, `RUN_NAME='full-v1'` for the full comparison. Source extraction validates paths and records the source ZIP hash. |
| 2: environment | Check a real LR-ASPP forward on CUDA, preserve the installed torch/torchvision pair, install helper dependencies and run the offline tests. Stop on failure. |
| 3: persistence and data | Leave `USE_DRIVE=True`, authorize your own Drive mount, and confirm the printed verified folder. The notebook checks writing and reading before training, downloads/audits data, and displays annotation conventions and audit overlays. |
| 4: diagnostics | Complete overfit and smoke checks. Inspect printed losses and displayed targets/predictions. Overfit predictions use the last checkpoint on the same two training images; their IoU is diagnostic only. |
| 5: review | After your own inspection, change `AUDIT_REVIEWED=False` and `DIAGNOSTICS_REVIEWED=False` to `True`, then run this cell. If outputs are wrong, return diagnostic artifacts first. |
| 6: A/B/C | Complete all three independent runs, select by validation pet IoU, and generate the measured training report. This cell does not evaluate official test or measure CPU deployment speed. |
| 7: export | Stop training before exporting. Download the artifact ZIP and its SHA-256 sidecar; save a copy of the notebook with its actual outputs. |

If free GPU time is restricted, choose the common reduced family **before starting A/B/C**. Change only these settings in code cell 1:

```python
DEVICE = 'cuda'
CONFIG_FAMILY = 'reduced'
RUN_NAME = 'reduced-v1'
```

The reduced family uses `configs/cpu_A.json`, `cpu_B.json`, `cpu_C.json`: 1,000 common breed-stratified training images, batch size 2, full validation/test, the same seed and stopping budget. The `cpu_` filenames do not force a CPU device. If no free GPU is available and you deliberately accept slower CPU training, set `DEVICE='cpu'` as well. There is no automatic device or family fallback.

For GPU memory errors with the full family, choose a new `RUN_NAME` and set `BATCH_SIZE=4`, then 2 if needed, before restarting **all three** experiments. Preserve older evidence. Changing one run or resuming with another config invalidates the comparison.

If dependency installation reports that installed versions differ from loaded modules, use **Runtime → Restart session** and rerun from the first cell. This clears old NumPy/Pillow modules after an upgrade.

After a disconnect, upload the same source ZIP in a fresh runtime, use the same settings and run from the beginning. Drive retains the complete run folders; the dataset and pretrained-weight cache must be recreated if the runtime lost them. Matching checkpoints/configs/split/history/hash records are checked before skipping completed runs or resuming. Work since the last committed epoch repeats. Resume requires the same Python, torch, torchvision and NumPy versions; cross-hardware bitwise equality is not promised.

## Return these files here

Return `upgaming-petseg-<RUN_NAME>-artifacts.zip`, its `.zip.sha256` file, and the saved notebook with output. Report any runtime error without changing logs to look successful. The export cell can also package partial runs after an interruption.

The ZIP has one top-level `artifacts/` folder containing:

- `transfer_inventory.json`: per-file SHA-256, sizes, source/split identity and validated completion state.
- `workflow/`: exact comparison configs, runtime, identity and your diagnostic-review record.
- `data/`: splits, audit, exclusions, annotation README and audit overlay; no raw dataset.
- `diagnostics/`: complete diagnostic run folders and labeled training/validation overlays.
- `runs/A`, `runs/B`, `runs/C`: **both** best/latest checkpoints and hash sidecars, status, config, environment, training IDs, JSON/CSV histories.
- `final/selected.pt`, its hash and `selection.json`, plus `reports/` when selection completed.

If a large ZIP cannot be attached, return its hash, `transfer_inventory.json`, `workflow/`, `data/`, A/B/C logs/status/config/environment/training IDs/hash sidecars, and `final/selection.json` first. The selected checkpoint is still required for local measurements. Keep complete best/latest folders on Drive for resume. Nothing is published automatically.

## Local continuation after return

The agent can verify/import the returned files and execute this part. For manual use, verify the ZIP hash before extracting into a **new** folder, preserving earlier runs. Replace the archive name below if using the reduced family:

```powershell
$archivePath = 'C:\path\to\downloads\upgaming-petseg-full-v1-artifacts.zip'
$expectedHash = ((Get-Content -LiteralPath "$archivePath.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLower() -ne $expectedHash.ToLower()) { throw 'ZIP hash mismatch' }
Expand-Archive -LiteralPath $archivePath -DestinationPath 'artifacts/colab-import/full-v1'
$returned = (Resolve-Path 'artifacts/colab-import/full-v1/artifacts').Path
```

Check the per-file inventory with this Python snippet, using the actual extraction path if different:

```python
from pathlib import Path
from petseg.utils import read_json, sha256
root = Path('artifacts/colab-import/full-v1/artifacts').resolve()
for item in read_json(root / 'transfer_inventory.json')['files']:
    path = (root.parent / item['path']).resolve()
    assert path.is_relative_to(root)
    assert path.stat().st_size == item['bytes'] and sha256(path) == item['sha256'], path
print('Inventory verified')
```

Recreate local data using a separate audit destination, then require the local split hash to match the returned manifest before benchmarking. These commands use the extracted folder without overwriting earlier local artifacts:

```powershell
.\.venv\Scripts\python.exe -m petseg prepare --download --root data --output artifacts/local-data-check
$localSplitHash = (Get-FileHash 'artifacts/local-data-check/splits.json' -Algorithm SHA256).Hash
$returnedSplitHash = (Get-FileHash "$returned/data/splits.json" -Algorithm SHA256).Hash
if ($localSplitHash -ne $returnedSplitHash) { throw 'Split hash mismatch; investigate before evaluating' }
.\.venv\Scripts\python.exe -m petseg select --runs "$returned/runs" --manifest "$returned/data/splits.json" --output "$returned/final"
.\.venv\Scripts\python.exe -m petseg --threads 1 benchmark --root data --manifest "$returned/data/splits.json" --checkpoint "$returned/final/selected.pt" --selection "$returned/final/selection.json" --output "$returned/final/decision.json"
.\.venv\Scripts\python.exe -m petseg evaluate --root data --manifest "$returned/data/splits.json" --checkpoint "$returned/final/selected.pt" --decision "$returned/final/decision.json" --split test --size 256 --output "$returned/evaluations/test_256"
$deploymentSize = (.\.venv\Scripts\python.exe -c "import json,sys; print(json.load(open(sys.argv[1]))['deployment_resolution'])" "$returned/final/decision.json").Trim()
if ($deploymentSize -ne '256') {
    .\.venv\Scripts\python.exe -m petseg evaluate --root data --manifest "$returned/data/splits.json" --checkpoint "$returned/final/selected.pt" --decision "$returned/final/decision.json" --split test --size $deploymentSize --output "$returned/evaluations/test_$deploymentSize"
}
.\.venv\Scripts\python.exe -m petseg report --artifacts "$returned"
$env:PETSEG_CHECKPOINT = "$returned/final/selected.pt"
$env:PETSEG_DECISION = "$returned/final/decision.json"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run these sequentially and stop if a command fails. The CPU benchmark freezes the deployment choice before test evaluation. Do not retune after inspecting test results. Keep all measurements, including an optimization attempt that retains 256. Inspect actual overlays and new appropriately sourced images, finish the report/model card, and verify trained-app upload/download before marking the submission complete. Ignored boundary pixels remain excluded from reported metrics.
