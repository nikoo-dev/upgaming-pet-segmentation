"""Reports are generated exclusively from existing measured artifacts."""
from pathlib import Path
import os

from .utils import read_json, sha256, write_json


def number(value):
    return "undefined" if value is None else f"{value:.4f}"


def build_report(artifacts):
    root = Path(artifacts)
    report = root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    lines = ["# Measured experiment results", "", "Generated from saved artifacts. Missing measurements remain pending.", ""]
    evidence = {}
    warnings = []
    checked_runs = {}
    os.environ.setdefault("MPLCONFIGDIR", str((root / "cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plotted = False
    for name in ["A", "B", "C"]:
        run = root / "runs" / name
        evidence[f"experiment_{name}"] = False
        if run.exists():
            from .train import inspect_run
            try:
                checked = inspect_run(run, manifest=root / "data/splits.json", require_completed=True)
                config = checked["config"]
                if (config["diagnostic"] or config["experiment"] != name
                        or config["frozen"] != (name == "A") or config["augment"] != (name == "C")):
                    raise ValueError("Not a completed final experiment")
                history = checked["latest"]["history"]
                checked_runs[name] = checked
                evidence[f"experiment_{name}"] = True
            except (OSError, ValueError, KeyError) as exc:
                warnings.append(f"Excluded run {name}: {exc}")
                continue
            axes[0].plot([r["epoch"] + 1 for r in history], [r["train_loss"] for r in history], label=name)
            axes[1].plot([r["epoch"] + 1 for r in history], [r["validation_foreground_iou"] for r in history], label=name)
            plotted = True
    if plotted:
        for axis, title in zip(axes, ["Training loss", "Validation foreground IoU"]):
            axis.set(xlabel="Epoch", title=title)
            axis.legend()
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(report / "learning_curves.png", dpi=150)
    plt.close(fig)
    if not plotted:
        (report / "learning_curves.png").unlink(missing_ok=True)
    selection = root / "final/selection.json"
    state = None
    if selection.exists():
        candidate = read_json(selection)
        checkpoint = root / "final/selected.pt"
        manifest = root / "data/splits.json"
        if (checkpoint.is_file() and manifest.is_file()
                and candidate.get("checkpoint_sha256") == sha256(checkpoint)
                and candidate.get("manifest_sha256") == sha256(manifest)
                and all(evidence[f"experiment_{n}"] for n in ["A", "B", "C"])
                and candidate.get("selected_experiment") in checked_runs
                and candidate["checkpoint_sha256"] == sha256(
                    root / "runs" / candidate["selected_experiment"] / "best.pt")):
            from .benchmark import inspect_comparison
            try:
                current = inspect_comparison(root / "runs", manifest)
                # Stored source paths may refer to Colab after artifact relocation.
                recorded_rows = [{k: v for k, v in row.items() if k != "checkpoint"}
                                 for row in candidate["candidates"]]
                measured_rows = [{k: v for k, v in row.items() if k != "checkpoint"}
                                 for row in current["candidates"]]
                if (recorded_rows != measured_rows
                        or candidate["selected_experiment"] != current["winner"]["experiment"]
                        or candidate.get("initial_model_sha256") != current["initial_model_sha256"]):
                    raise ValueError("Selection scores, winner or initialization differ from current runs")
                state = candidate
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f"Excluded stale or incomparable selection: {exc}")
        else:
            warnings.append("Excluded selection: checkpoint, manifest or completed comparison evidence does not match")
    if state is not None:
        lines += ["## Training comparisons", "", "| Experiment | Best validation pet IoU | Epochs | Training steps |", "|---|---:|---:|---:|"]
        for row in state["candidates"]:
            lines.append(f"| {row['experiment']} | {number(row['validation_foreground_iou'])} | {row['epochs_completed']} | {row['global_step']} |")
        lines += ["", f"Selected experiment: **{state['selected_experiment']}**. Single seed; no statistical significance claim.", ""]
    else:
        lines += ["Training comparison: **pending**. Run all experiments and `petseg select`.", ""]
    decision = root / "final/decision.json"
    d = None
    if decision.exists():
        candidate = read_json(decision)
        if (state is not None and candidate.get("checkpoint_sha256") == state["checkpoint_sha256"]
                and candidate.get("manifest_sha256") == state["manifest_sha256"]):
            from .inference import validate_deployment
            try:
                validate_deployment(candidate, state["checkpoint_sha256"],
                                    checked_runs[state["selected_experiment"]]["best"],
                                    state["manifest_sha256"])
                d = candidate
            except (ValueError, KeyError, TypeError) as exc:
                warnings.append(f"Excluded invalid CPU decision: {exc}")
        else:
            warnings.append("Excluded CPU decision: it does not match verified model selection")
    evidence["optimization"] = d is not None
    if d is not None:
        lines += ["## CPU resolution comparison", "", "| Resolution | Validation pet IoU | Forward median / p95 ms | Pipeline median / p95 ms |", "|---|---:|---:|---:|"]
        for r in d["results"]:
            lines.append(f"| {r['size']} | {number(r['validation_foreground_iou'])} | {r['forward']['median_ms']:.2f} / {r['forward']['p95_ms']:.2f} | {r['end_to_end']['median_ms']:.2f} / {r['end_to_end']['p95_ms']:.2f} |")
        lines += ["", f"Selected deployment resolution: **{d['deployment_resolution']}**.", "", f"Model parameters: {d['parameters']}; checkpoint bytes: {d['checkpoint_bytes']} (same weights at every resolution).", "", f"Protocol: {d['scope']}.", ""]
        env = d["environment"]
        lines += [f"Hardware: {env['cpu']}; {env['platform']}. Torch {env['torch']}; "
                  f"{env['threads']} CPU thread(s). Timing uses {len(d['image_ids'])} images, "
                  f"{d['warmup_calls_per_candidate_per_path']} warm-up calls per candidate/path, "
                  f"and {d['timed_calls_per_candidate']} timed calls per candidate/path.", ""]
    else:
        lines += ["Optimization: **pending**. Run the CPU benchmark after model selection.", ""]
    evaluations = []
    for path in sorted((root / "evaluations").glob("test_*/metrics.json")):
        m = read_json(path)
        if (d is not None and m.get("split") == "test"
                and m.get("checkpoint_sha256") == d["checkpoint_sha256"]
                and m.get("manifest_sha256") == d["manifest_sha256"]
                and m.get("decision_sha256") == sha256(decision)
                and m.get("input_resolution") in {256, d["deployment_resolution"]}
                and m.get("images") == len(read_json(root / "data/splits.json")["splits"]["test"])):
            evaluations.append(m)
        else:
            warnings.append(f"Excluded stale or unverified evaluation: {path.relative_to(root).as_posix()}")
    expected_sizes = {256, d["deployment_resolution"]} if d is not None else {256}
    evidence["test_evaluation"] = {m["input_resolution"] for m in evaluations} == expected_sizes
    if evaluations:
        lines += ["## Held-out test results", "", "| Resolution | Pet IoU | Background IoU | Mean IoU | Pet Dice | Pixel accuracy |", "|---|---:|---:|---:|---:|---:|"]
        for m in evaluations:
            fields = ["foreground_iou", "background_iou", "mean_iou", "foreground_dice", "pixel_accuracy"]
            lines.append(f"| {m['input_resolution']} | " + " | ".join(number(m[k]) for k in fields) + " |")
        lines += ["", "All reported metrics exclude ambiguous trimap boundary pixels. Inspect ranked_examples.png and per_image.csv for actual errors.", ""]
        if not evidence["test_evaluation"]:
            lines += ["One or more frozen-resolution test evaluations are **pending**.", ""]
    else:
        lines += ["Final test evaluation: **pending**. Freeze model and resolution decisions first.", ""]
    lines += ["## Interpretation still requiring human review", "", "Inspect the saved overlays and describe observed failures (clutter, occlusion, similar colors, small pets). Do not describe a failure as observed until inspecting the corresponding output.", "", "No automatic accuracy acceptance threshold is imposed. A generated table does not replace an engineering discussion or candidate code review."]
    if warnings:
        lines += ["", "## Evidence excluded", ""] + [f"- {warning}" for warning in warnings]
    (report / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report / "evidence_status.json", evidence)
    return evidence
