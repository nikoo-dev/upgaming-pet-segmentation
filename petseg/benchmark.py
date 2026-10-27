"""Validation-only selection and a fixed, reproducible CPU latency protocol."""
from pathlib import Path
import math
import random
import shutil
import time

import numpy as np
import torch

from .data import annotation_root, load_pair, stratified_cap
from .evaluate import evaluate_model
from .inference import load_model, predict_mask
from .transforms import preprocess
from .utils import environment, read_json, sha256, write_json


def inspect_comparison(runs, manifest):
    """Read and validate the current A/B/C comparison without copying artifacts."""
    from .train import inspect_run

    candidates, configs, training_ids, initializations = [], [], [], []
    for name in ["A", "B", "C"]:
        run = Path(runs) / name
        inspected = inspect_run(run, manifest=manifest, require_completed=True)
        status, state, config = inspected["status"], inspected["best"], inspected["config"]
        if not status["completed"] or status["diagnostic"] or config["diagnostic"]:
            raise ValueError(f"{name} is incomplete or diagnostic, not a final comparison")
        if sha256(run / "best.pt") != status["best_checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint hash differs from its run status")
        if state["manifest_sha256"] != sha256(manifest):
            raise ValueError(f"{name} used different splits")
        if config["experiment"] != name or config["frozen"] != (name == "A") or config["augment"] != (name == "C"):
            raise ValueError(f"{name} does not implement the specified ablation")
        if not isinstance(state["best_iou"], (int, float)) or not math.isfinite(state["best_iou"]) or not 0 <= state["best_iou"] <= 1:
            raise ValueError(f"{name} validation foreground IoU must be finite and within [0, 1]")
        configs.append({k: v for k, v in config.items() if k not in ["experiment", "frozen", "augment"]})
        training_ids.append(read_json(run / "training_ids.json"))
        initializations.append(state.get("initial_model_sha256"))
        candidates.append({"experiment": name, "validation_foreground_iou": state["best_iou"],
                           "checkpoint": str(run / "best.pt"), "epochs_completed": status["epochs_completed"],
                           "global_step": status["global_step"]})
    if not all(c == configs[0] for c in configs):
        raise ValueError("Comparisons require matching budgets, batch size, seed and preprocessing")
    if not all(ids == training_ids[0] for ids in training_ids):
        raise ValueError("Comparisons require the same actual training image IDs")
    if not initializations[0] or not all(identity == initializations[0] for identity in initializations):
        raise ValueError("Comparisons require matching recorded initial model tensors")
    winner = max(candidates, key=lambda r: r["validation_foreground_iou"])
    return {"candidates": candidates, "winner": winner, "initial_model_sha256": initializations[0]}


def select_checkpoint(runs, manifest, output):
    comparison = inspect_comparison(runs, manifest)
    winner = comparison["winner"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / "selected.pt"
    shutil.copy2(winner["checkpoint"], final)
    digest = sha256(final)
    final.with_suffix(".pt.sha256").write_text(digest + "\n", encoding="ascii")
    record = {"selected_experiment": winner["experiment"], "checkpoint_sha256": digest,
              "manifest_sha256": sha256(manifest), "candidates": comparison["candidates"],
              "initial_model_sha256": comparison["initial_model_sha256"],
              "selection": "maximum validation foreground IoU; ties A, B, C"}
    write_json(output / "selection.json", record)
    return record


def summarize_times(values):
    return {"median_ms": float(np.median(values) * 1000), "p95_ms": float(np.percentile(values, 95) * 1000)}


def choose_resolution(results):
    if not isinstance(results, list) or len(results) != 3 or {r.get("size") for r in results if isinstance(r, dict)} != {128, 192, 256}:
        raise ValueError("Resolution comparison requires exactly 256, 192 and 128")
    for row in results:
        iou = row.get("validation_foreground_iou")
        pipeline = row.get("end_to_end")
        latency = pipeline.get("median_ms") if isinstance(pipeline, dict) else None
        if (not isinstance(iou, (int, float)) or not math.isfinite(iou) or not 0 <= iou <= 1
                or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0):
            raise ValueError("Resolution comparison requires finite IoUs in [0, 1] and positive median latency")
    baseline = next(r for r in results if r["size"] == 256)
    acceptable = [r["size"] for r in results if r["size"] < 256
                  and r["validation_foreground_iou"] >= baseline["validation_foreground_iou"] - .02
                  and r["end_to_end"]["median_ms"] < baseline["end_to_end"]["median_ms"]]
    return min(acceptable) if acceptable else 256


def benchmark(root, manifest, checkpoint, selection_path, output, threads=1, calls=100):
    if calls < 100 or threads < 1:
        raise ValueError("Use at least 100 calls and a positive fixed CPU thread count")
    selection = read_json(selection_path)
    digest, manifest_digest = sha256(checkpoint), sha256(manifest)
    if selection["checkpoint_sha256"] != digest or selection["manifest_sha256"] != manifest_digest:
        raise ValueError("Selection, checkpoint and manifest identities differ")
    torch.set_num_threads(threads)
    model, state = load_model(checkpoint)
    if state.get("manifest_sha256") != manifest_digest:
        raise ValueError("Checkpoint training splits differ from the benchmark manifest")
    if state["config"]["diagnostic"]:
        raise ValueError("Cannot freeze deployment from a diagnostic run")
    if state["config"].get("experiment") != selection.get("selected_experiment"):
        raise ValueError("Checkpoint experiment differs from the selected experiment")
    records = read_json(manifest)["splits"]["val"]
    if not records:
        raise ValueError("Empty validation split")
    selected = sorted(stratified_cap(records, 100, seed=42), key=lambda r: r["id"])
    images = [load_pair(annotation_root(root), r)[0] for r in selected]
    sizes = [256, 192, 128]
    times = {size: {"forward": [], "end_to_end": []} for size in sizes}
    # Preload data and tensors before timing; do not precompute preprocessing for E2E.
    tensors = {s: [preprocess(im, s).unsqueeze(0) for im in images] for s in sizes}
    with torch.inference_mode():
        for size in sizes:
            for i in range(10):
                model(tensors[size][i % len(images)])
                predict_mask(model, images[i % len(images)], size)
        order_rng = random.Random(42)
        for i in range(calls):
            order = [(size, path) for size in sizes for path in ["forward", "end_to_end"]]
            order_rng.shuffle(order)
            for size, path in order:
                start = time.perf_counter()
                if path == "forward":
                    model(tensors[size][i % len(images)])
                else:
                    predict_mask(model, images[i % len(images)], size)
                times[size][path].append(time.perf_counter() - start)
    results = []
    for size in sizes:
        metrics, _ = evaluate_model(model, root, records, size)
        results.append({"size": size, "validation_foreground_iou": metrics["foreground_iou"],
                        "forward": summarize_times(times[size]["forward"]),
                        "end_to_end": summarize_times(times[size]["end_to_end"])})
    record = {"schema_version": 1, "checkpoint_sha256": digest, "manifest_sha256": manifest_digest,
              "deployment_resolution": choose_resolution(results), "baseline_resolution": 256,
              "policy": "smallest size within 0.02 absolute validation foreground IoU of 256 with strictly lower median end-to-end CPU latency; otherwise 256",
              "results": results, "parameters": sum(p.numel() for p in model.parameters()),
              "checkpoint_bytes": Path(checkpoint).stat().st_size,
              "environment": environment(), "timed_calls_per_candidate": calls,
              "warmup_calls_per_candidate_per_path": 10, "image_ids": [r["id"] for r in selected],
              "timing_order": "all six resolution/path pairs shuffled independently per image/call using seed 42",
              "image_selection": "up to 100 breed-stratified validation images, seed 42, then sorted by ID",
              "scope": "FP32 CPU, batch 1; model-only and in-memory preprocessing+forward+resize+argmax; no loading, rendering, or PNG encoding"}
    write_json(output, record)
    return record
