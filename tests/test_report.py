from petseg.report import build_report
from petseg.utils import sha256, write_json


def test_report_excludes_old_metrics_and_removes_stale_curves(tmp_path):
    report = tmp_path / "reports"
    report.mkdir()
    (report / "learning_curves.png").write_bytes(b"old curve")
    write_json(tmp_path / "evaluations/test_256/metrics.json", {
        "split": "test", "checkpoint_sha256": "old checkpoint",
        "manifest_sha256": "old manifest", "input_resolution": 256,
        "foreground_iou": .9999,
    })
    evidence = build_report(tmp_path)
    text = (report / "results.md").read_text()
    assert not evidence["test_evaluation"]
    assert "0.9999" not in text
    assert "Excluded stale or unverified evaluation" in text
    assert not (report / "learning_curves.png").exists()


def test_report_incomplete_run_is_pending(tmp_path):
    write_json(tmp_path / "runs/A/status.json", {"completed": False, "diagnostic": False})
    evidence = build_report(tmp_path)
    assert not evidence["experiment_A"]
    assert "pending" in (tmp_path / "reports/results.md").read_text()


def test_report_accepts_current_results_and_requires_both_frozen_sizes(tmp_path, monkeypatch):
    import petseg.train as training
    import petseg.benchmark as benchmark

    manifest = tmp_path / "data/splits.json"
    write_json(manifest, {"splits": {"test": [{"id": "synthetic_test_image"}]}})
    identity = sha256(manifest)
    checkpoint = tmp_path / "final/selected.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic report fixture, not model weights")
    digest = sha256(checkpoint)
    history = [{"epoch": 0, "train_loss": .8, "validation_foreground_iou": .7}]
    for name in ["A", "B", "C"]:
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "best.pt").write_bytes(checkpoint.read_bytes())

    def inspected(run, **kwargs):
        config = {"experiment": run.name, "diagnostic": False,
                  "frozen": run.name == "A", "augment": run.name == "C"}
        return {"config": config, "latest": {"history": history},
                "best": {"config": config, "manifest_sha256": identity}}

    monkeypatch.setattr(training, "inspect_run", inspected)
    candidates = [{"experiment": name, "validation_foreground_iou": .7,
                   "epochs_completed": 1, "global_step": 2} for name in ["A", "B", "C"]]
    monkeypatch.setattr(benchmark, "inspect_comparison", lambda *_: {
        "candidates": candidates, "winner": candidates[0], "initial_model_sha256": "synthetic initialization"})
    selection_path = tmp_path / "final/selection.json"
    selection = {
        "selected_experiment": "A", "checkpoint_sha256": digest, "manifest_sha256": identity,
        "candidates": candidates, "initial_model_sha256": "synthetic initialization",
    }
    write_json(selection_path, selection)
    decision_path = tmp_path / "final/decision.json"
    write_json(decision_path, {
        "schema_version": 1, "checkpoint_sha256": digest, "manifest_sha256": identity,
        "baseline_resolution": 256, "deployment_resolution": 192,
        "results": [{"size": size, "validation_foreground_iou": iou,
                     "forward": {"median_ms": latency, "p95_ms": latency + 1},
                     "end_to_end": {"median_ms": latency + 2, "p95_ms": latency + 3}}
                    for size, iou, latency in [(256, .8, 20), (192, .79, 15), (128, .7, 10)]],
        "parameters": 1, "checkpoint_bytes": checkpoint.stat().st_size, "scope": "synthetic timing fixture",
        "environment": {"cpu": "synthetic", "platform": "synthetic", "torch": "synthetic", "threads": 1},
        "image_ids": ["synthetic_validation_image"], "warmup_calls_per_candidate_per_path": 10,
        "timed_calls_per_candidate": 100,
    })
    metrics = {"split": "test", "checkpoint_sha256": digest, "manifest_sha256": identity,
               "decision_sha256": sha256(decision_path), "images": 1,
               "foreground_iou": .81, "background_iou": .8, "mean_iou": .805,
               "foreground_dice": .895, "pixel_accuracy": .9}
    write_json(tmp_path / "evaluations/test_256/metrics.json", {**metrics, "input_resolution": 256})
    assert not build_report(tmp_path)["test_evaluation"]
    write_json(tmp_path / "evaluations/test_192/metrics.json", {**metrics, "input_resolution": 192})
    assert build_report(tmp_path)["test_evaluation"]
    text = (tmp_path / "reports/results.md").read_text()
    assert "0.8100" in text and "100 timed calls" in text
    # A current winning checkpoint must not let stale comparison scores through.
    stale = {**selection, "candidates": [{**candidates[0], "validation_foreground_iou": .1234}, *candidates[1:]]}
    write_json(selection_path, stale)
    assert not build_report(tmp_path)["optimization"]
    text = (tmp_path / "reports/results.md").read_text()
    assert "0.1234" not in text and "Excluded stale or incomparable selection" in text
    write_json(selection_path, selection)
    metrics["decision_sha256"] = "old policy"
    write_json(tmp_path / "evaluations/test_192/metrics.json", {**metrics, "input_resolution": 192})
    assert not build_report(tmp_path)["test_evaluation"]
