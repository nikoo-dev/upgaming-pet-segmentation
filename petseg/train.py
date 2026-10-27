"""Explicit epoch-boundary resumable training, with truthful diagnostic runs."""
import csv
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .data import PetDataset, stratified_cap
from .evaluate import evaluate_model
from .model import build_model, configure_training, make_optimizer, segmentation_loss, training_mode
from .transforms import CLASS_MAPPING, PREPROCESSING
from .utils import (environment, load_checkpoint, read_json, restore_rng, rng_state,
                    save_checkpoint, seed_all, sha256, tensor_state_sha256, write_json)

DEFAULT = {"seed": 42, "size": 256, "batch_size": 8, "epochs": 12, "patience": 3,
           "min_epochs": 3, "head_lr": .001, "backbone_lr": .0001,
           "weight_decay": .0001, "workers": 0, "train_cap": None,
           "max_steps": None, "val_limit": None, "pretrained": True}


def load_config(path):
    config = {**DEFAULT, **read_json(path)}
    if config.get("experiment") not in ["A", "B", "C", "diagnostic"]:
        raise ValueError("Experiment must be A, B, C, or diagnostic")
    if config["workers"] != 0:
        raise ValueError("Use workers=0 for exact epoch-boundary RNG restoration in this implementation")
    if min(config["batch_size"], config["epochs"], config["patience"], config["min_epochs"]) < 1:
        raise ValueError("Batch size, epochs and stopping settings must be positive")
    if config["min_epochs"] > config["epochs"]:
        raise ValueError("min_epochs cannot exceed epochs")
    config["frozen"] = config.get("frozen", config["experiment"] == "A")
    config["augment"] = config.get("augment", config["experiment"] == "C")
    config["diagnostic"] = bool(config["max_steps"] or config["val_limit"] or config["experiment"] == "diagnostic" or not config["pretrained"])
    return config


def _validate_training_state(state):
    if (state.get("schema_version") != 1 or state.get("preprocessing") != PREPROCESSING
            or state.get("class_mapping") != CLASS_MAPPING):
        raise ValueError("Training checkpoint schema, preprocessing or class mapping differs")
    try:
        history, config = state["history"], state["config"]
        if (not isinstance(history, list) or not isinstance(config, dict)
                or not history or len(history) != state["epoch"] + 1
                or len(history) > config["epochs"]):
            raise ValueError("Checkpoint history does not match its completed epoch")
        previous_step = 0
        best, best_epoch = -1., -1
        for epoch, row in enumerate(history):
            value = row["validation_foreground_iou"]
            if (row["epoch"] != epoch or row["global_step"] <= previous_step
                    or not math.isfinite(value) or not 0 <= value <= 1
                    or not math.isfinite(row["train_loss"]) or row["train_loss"] < 0):
                raise ValueError("Checkpoint history has invalid epoch, step, loss or validation IoU")
            previous_step = row["global_step"]
            if value > best:
                best, best_epoch = value, epoch
        if (state["global_step"] != previous_step or state["best_iou"] != best
                or state["bad_epochs"] != state["epoch"] - best_epoch
                or state.get("best_epoch", best_epoch) != best_epoch):
            raise ValueError("Checkpoint best/step/stopping state differs from its history")
        stopped = len(history) >= config["min_epochs"] and state["bad_epochs"] >= config["patience"]
        if state["stopped_early"] != stopped:
            raise ValueError("Checkpoint early-stopping flag differs from its history")
        if any(key not in state for key in ["model", "optimizer", "rng", "environment", "manifest_sha256"]):
            raise ValueError("Checkpoint is missing resumable training state")
        if (not isinstance(state["model"], dict) or not state["model"]
                or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                           for key, value in state["model"].items())
                or not all(isinstance(state[key], dict) for key in ["optimizer", "rng", "environment"])):
            raise ValueError("Checkpoint model, optimizer, RNG or environment state is malformed")
        expected_initialization = "ImageNet backbone; new binary head" if config["pretrained"] else "random diagnostic"
        if state.get("initialization") != expected_initialization:
            raise ValueError("Checkpoint initialization provenance differs from its config")
        return best_epoch
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete training checkpoint metadata") from exc


def _terminal(state):
    config = state["config"]
    return (state["epoch"] + 1 >= config["epochs"] or state["stopped_early"]
            or bool(config["max_steps"] and state["global_step"] >= config["max_steps"]))


def inspect_run(output, config=None, manifest=None, require_completed=False):
    """Validate a run before notebook reuse, resume, reporting or final selection.

    A committed latest checkpoint can precede best/log/status writes. Such a run
    returns completed=False and can be repaired by train(..., resume=info['resume']).
    Completed runs require all evidence to agree; status existence alone is not enough.
    """
    output = Path(output)
    latest = load_checkpoint(output / "latest.pt")
    best_epoch = _validate_training_state(latest)
    stored_config = read_json(output / "config.json")
    if latest["config"] != stored_config or (config is not None and stored_config != config):
        raise ValueError("Run config differs from the requested config or checkpoint")
    saved_environment = read_json(output / "environment.json")
    if any(saved_environment.get(key) != latest["environment"].get(key)
           for key in ["python", "torch", "torchvision", "numpy"]):
        raise ValueError("Run environment metadata differs from latest checkpoint")
    if manifest is not None:
        if latest["manifest_sha256"] != sha256(manifest):
            raise ValueError("Run split manifest differs from the requested manifest")
        records = stratified_cap(read_json(manifest)["splits"]["train"],
                                 stored_config["train_cap"], stored_config["seed"])
        if read_json(output / "training_ids.json") != [r["id"] for r in records]:
            raise ValueError("Run training IDs differ from its split/config")
    else:
        ids = read_json(output / "training_ids.json")
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Run training IDs are empty or duplicated")
    best = None
    try:
        candidate = load_checkpoint(output / "best.pt")
        _validate_training_state(candidate)
        if (candidate["config"] != stored_config
                or candidate["manifest_sha256"] != latest["manifest_sha256"]
                or candidate["epoch"] != best_epoch
                or candidate["history"] != latest["history"][:best_epoch + 1]
                or candidate["best_iou"] != latest["best_iou"]
                or candidate.get("initial_model_sha256") != latest.get("initial_model_sha256")):
            raise ValueError("Best checkpoint differs from latest checkpoint's best epoch/provenance")
        expected = latest.get("best_model_sha256")
        if expected is not None and tensor_state_sha256(candidate["model"]) != expected:
            raise ValueError("Best checkpoint tensor identity differs from latest checkpoint")
        best = candidate
    except (OSError, ValueError):
        # Current latest contains the best model itself, so replay can repair it.
        # An older missing best cannot be reconstructed from later model weights.
        if latest["epoch"] != best_epoch:
            raise
    status_path = output / "status.json"
    status = read_json(status_path) if status_path.exists() else None
    completed = False
    if status is not None:
        if (best is None or not _terminal(latest) or not status.get("completed")
                or status.get("diagnostic") != stored_config["diagnostic"]
                or status.get("epochs_completed") != len(latest["history"])
                or status.get("global_step") != latest["global_step"]
                or status.get("best_validation_foreground_iou") != latest["best_iou"]
                or status.get("best_checkpoint_sha256") != sha256(output / "best.pt")
                or status.get("latest_checkpoint_sha256", sha256(output / "latest.pt")) != sha256(output / "latest.pt")):
            raise ValueError("Run completion status differs from its checkpoints or stopping policy")
        if read_json(output / "history.json") != latest["history"]:
            raise ValueError("Run history JSON differs from latest checkpoint")
        with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        expected_rows = [{key: "" if value is None else str(value) for key, value in row.items()}
                         for row in latest["history"]]
        if rows != expected_rows:
            raise ValueError("Run history CSV differs from latest checkpoint")
        completed = True
    if require_completed and not completed:
        raise ValueError("Run is incomplete; resume latest.pt to finish checkpoint/log/status writes")
    return {"completed": completed, "resume": str(output / "latest.pt"),
            "config": stored_config, "latest": latest, "best": best, "status": status}


def _write_history(output, history):
    write_json(output / "history.json", history)
    temporary = output / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output / "history.csv")


def train(root, manifest, output, config, device="cpu", resume=None):
    if read_json(manifest).get("diagnostic_only") and not config["diagnostic"]:
        raise ValueError("Diagnostic subset cannot be used for final comparison experiments")
    output = Path(output)
    if any((output / name).exists() for name in ["latest.pt", "best.pt", "status.json"]) and resume is None:
        raise FileExistsError("Run directory already contains a checkpoint; use --resume or a new --output")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    dataset = PetDataset(root, manifest, "train", config["size"], config["augment"], config["train_cap"], config["seed"])
    if not len(dataset):
        raise ValueError("Empty training set")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, generator=generator)
    val = read_json(manifest)["splits"]["val"]
    if config["val_limit"]:
        val = val[:config["val_limit"]]
    model = build_model(pretrained=config["pretrained"] and resume is None).to(device)
    initial_digest = tensor_state_sha256(model.state_dict())
    configure_training(model, config["frozen"])
    optimizer = make_optimizer(model, config)
    start, step, best, best_epoch, best_digest, bad, history, elapsed_before = 0, 0, -1., -1, None, 0, [], 0.
    identity = sha256(manifest)
    if resume:
        if Path(resume).resolve() != (output / "latest.pt").resolve():
            raise ValueError("Resume must use latest.pt in the output run directory; copy the complete run directory")
        info = inspect_run(output, config=config, manifest=manifest)
        state = info["latest"]
        initial_digest = state.get("initial_model_sha256")
        if state["config"] != config or state["manifest_sha256"] != identity:
            raise ValueError("Resume config or split manifest differs; start a new named run")
        current_environment = environment()
        for key in ["python", "torch", "torchvision", "numpy"]:
            if state["environment"][key] != current_environment[key]:
                raise ValueError(f"Resume requires the same {key} version")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer tensors must follow parameters after CPU-portable loading.
        for values in optimizer.state.values():
            for k, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[k] = value.to(device)
        start, step, best, bad = state["epoch"] + 1, state["global_step"], state["best_iou"], state["bad_epochs"]
        best_epoch = _validate_training_state(state)
        if info["best"] is None:
            save_checkpoint(output / "best.pt", state)
        best_digest = state.get("best_model_sha256") or tensor_state_sha256(load_checkpoint(output / "best.pt")["model"])
        history, elapsed_before = state["history"], state["elapsed_seconds"]
        restore_rng(state["rng"], generator)
        if info["completed"]:
            return model, history
        # A terminal checkpoint may have committed immediately before a crash.
        # Rebuild the logs/status without training another epoch.
        if _terminal(state):
            start = config["epochs"]
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment())
    write_json(output / "training_ids.json", [r["id"] for r in dataset.records])
    started = time.perf_counter()
    for epoch in range(start, config["epochs"]):
        training_mode(model, config["frozen"])
        weighted_loss, valid_pixels, skipped = 0., 0, 0
        for images, targets in loader:
            if config["max_steps"] and step >= config["max_steps"]:
                break
            images, targets = images.to(device), targets.to(device)
            if not (targets != 255).any():
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = segmentation_loss(model(images)["out"], targets)
                if loss is None:
                    skipped += 1
                    continue
                loss.backward()
                optimizer.step()
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError("GPU out of memory. Restart ALL comparison runs with a common batch size 4, then 2. Use new run directories; do not resume with changed config.") from exc
            n = int((targets != 255).sum())
            weighted_loss += loss.item() * n
            valid_pixels += n
            step += 1
            if step % 100 == 0:
                print(f"Epoch {epoch+1}: steps={step}, running loss={weighted_loss / valid_pixels:.4f}", flush=True)
        if valid_pixels == 0:
            raise ValueError("Epoch had no valid training pixels/steps")
        metrics, _ = evaluate_model(model, root, val, config["size"], device)
        value = metrics["foreground_iou"]
        if value is None:
            raise ValueError("Validation foreground IoU is undefined")
        improved = value > best
        best, bad = (value, 0) if improved else (best, bad + 1)
        if improved:
            best_epoch = epoch
            best_digest = tensor_state_sha256(model.state_dict())
        stopped_early = epoch + 1 >= config["min_epochs"] and bad >= config["patience"]
        row = {"epoch": epoch, "global_step": step, "train_loss": weighted_loss / valid_pixels,
               "validation_foreground_iou": value, "validation_mean_iou": metrics["mean_iou"],
               "skipped_all_ignored_batches": skipped,
               "elapsed_seconds": elapsed_before + time.perf_counter() - started}
        history.append(row)
        state = {"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "global_step": step, "best_iou": best, "bad_epochs": bad,
                 "stopped_early": stopped_early, "config": config, "rng": rng_state(generator),
                 "manifest_sha256": identity, "preprocessing": PREPROCESSING, "class_mapping": CLASS_MAPPING,
                 "environment": environment(), "history": history, "elapsed_seconds": row["elapsed_seconds"],
                 "initialization": "ImageNet backbone; new binary head" if config["pretrained"] else "random diagnostic"}
        state.update(best_epoch=best_epoch, best_model_sha256=best_digest)
        state["initial_model_sha256"] = initial_digest
        # Latest is the epoch commit. If the following best/log writes fail,
        # resume can reconstruct a new best from this checkpoint's weights.
        save_checkpoint(output / "latest.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        _write_history(output, history)
        print(f"Epoch {epoch+1}: loss={row['train_loss']:.4f}, val pet IoU={value:.4f}, steps={step}", flush=True)
        if stopped_early or (config["max_steps"] and step >= config["max_steps"]):
            break
    _write_history(output, history)
    write_json(output / "status.json", {"completed": True, "diagnostic": config["diagnostic"],
               "epochs_completed": len(history), "global_step": step,
               "best_validation_foreground_iou": best,
               "best_checkpoint_sha256": sha256(output / "best.pt"),
               "latest_checkpoint_sha256": sha256(output / "latest.pt")})
    return model, history
