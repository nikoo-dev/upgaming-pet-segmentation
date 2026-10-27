import pytest
import torch
import csv

from petseg.benchmark import inspect_comparison, select_checkpoint
from petseg.train import DEFAULT
from petseg.transforms import CLASS_MAPPING, PREPROCESSING
from petseg.utils import environment, read_json, save_checkpoint, sha256, tensor_state_sha256, write_json


def make_runs(tmp_path, diagnostic=False):
    manifest = tmp_path / "splits.json"
    records = [{"id": "synthetic_fixture_only", "breed": 1, "content_sha256": "fixture"}]
    write_json(manifest, {"splits": {"train": records, "val": [], "test": []}})
    for name, iou in [("A", .6), ("B", .8), ("C", .8)]:
        path = tmp_path / "runs" / name
        # Synthetic run records exercise provenance checks, never model quality.
        config = {**DEFAULT, "experiment": name, "frozen": name == "A", "augment": name == "C",
                  "diagnostic": diagnostic, "epochs": 1, "min_epochs": 1}
        history = [{"epoch": 0, "global_step": 1, "train_loss": .1, "validation_foreground_iou": iou}]
        tensors = {"synthetic_fixture": torch.zeros(1)}
        state = {"schema_version": 1, "best_iou": iou, "manifest_sha256": sha256(manifest),
                 "preprocessing": PREPROCESSING, "class_mapping": CLASS_MAPPING, "config": config,
                 "epoch": 0, "global_step": 1, "bad_epochs": 0, "stopped_early": False,
                 "history": history, "model": tensors, "optimizer": {}, "rng": {},
                 "environment": environment(), "initialization": "ImageNet backbone; new binary head",
                 "initial_model_sha256": tensor_state_sha256(tensors),
                 "best_model_sha256": tensor_state_sha256(tensors)}
        save_checkpoint(path / "best.pt", state)
        save_checkpoint(path / "latest.pt", state)
        write_json(path / "config.json", config)
        write_json(path / "environment.json", state["environment"])
        write_json(path / "training_ids.json", [r["id"] for r in records])
        write_json(path / "history.json", history)
        with (path / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        write_json(path / "status.json", {"completed": True, "diagnostic": diagnostic,
                   "best_checkpoint_sha256": sha256(path / "best.pt"),
                   "latest_checkpoint_sha256": sha256(path / "latest.pt"),
                   "best_validation_foreground_iou": iou, "epochs_completed": 1, "global_step": 1})
    return manifest


def test_selection_tie_and_hash_identity(tmp_path):
    manifest = make_runs(tmp_path)
    selected = select_checkpoint(tmp_path / "runs", manifest, tmp_path / "final")
    assert selected["selected_experiment"] == "B"
    assert selected["checkpoint_sha256"] == sha256(tmp_path / "final/selected.pt")


def test_comparison_inspection_preserves_run_files(tmp_path):
    manifest = make_runs(tmp_path)
    before = {path.relative_to(tmp_path): sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    comparison = inspect_comparison(tmp_path / "runs", manifest)
    after = {path.relative_to(tmp_path): sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    assert comparison["winner"]["experiment"] == "B"
    assert [row["experiment"] for row in comparison["candidates"]] == ["A", "B", "C"]


def test_selection_rejects_diagnostics(tmp_path):
    manifest = make_runs(tmp_path, diagnostic=True)
    with pytest.raises(ValueError, match="diagnostic"):
        select_checkpoint(tmp_path / "runs", manifest, tmp_path / "final")


def test_selection_rejects_mixed_budgets(tmp_path):
    manifest = make_runs(tmp_path)
    run = tmp_path / "runs/B"
    for checkpoint in [run / "best.pt", run / "latest.pt"]:
        state = torch.load(checkpoint, weights_only=True)
        state["config"]["batch_size"] = 2
        save_checkpoint(checkpoint, state)
    write_json(run / "config.json", state["config"])
    status_path = run / "status.json"
    status = read_json(status_path)
    status["best_checkpoint_sha256"] = sha256(run / "best.pt")
    status["latest_checkpoint_sha256"] = sha256(run / "latest.pt")
    write_json(status_path, status)
    with pytest.raises(ValueError, match="matching budgets"):
        select_checkpoint(tmp_path / "runs", manifest, tmp_path / "final")


def test_selection_rejects_stale_history_and_wrong_training_ids(tmp_path):
    manifest = make_runs(tmp_path)
    path = tmp_path / "runs/B/history.json"
    history = read_json(path)
    history[0]["validation_foreground_iou"] = .99
    write_json(path, history)
    with pytest.raises(ValueError, match="history JSON"):
        select_checkpoint(tmp_path / "runs", manifest, tmp_path / "final")
    write_json(path, torch.load(tmp_path / "runs/B/latest.pt", weights_only=True)["history"])
    write_json(tmp_path / "runs/B/training_ids.json", ["other_image"])
    with pytest.raises(ValueError, match="training IDs"):
        select_checkpoint(tmp_path / "runs", manifest, tmp_path / "final")


def test_selection_rejects_different_initial_tensors(tmp_path):
    manifest = make_runs(tmp_path)
    run = tmp_path / "runs/C"
    for checkpoint in [run / "best.pt", run / "latest.pt"]:
        state = torch.load(checkpoint, weights_only=True)
        state["initial_model_sha256"] = "different-independent-initialization"
        save_checkpoint(checkpoint, state)
    status = read_json(run / "status.json")
    status["best_checkpoint_sha256"] = sha256(run / "best.pt")
    status["latest_checkpoint_sha256"] = sha256(run / "latest.pt")
    write_json(run / "status.json", status)
    with pytest.raises(ValueError, match="initial model tensors"):
        select_checkpoint(tmp_path / "runs", manifest, tmp_path / "final")
