"""Package verified submission evidence without publishing or bundling raw data.

Run from the project environment after training, evaluation, qualitative review,
and trained-app QA. Include the selected checkpoint and complete final A/B/C
best/latest states so selection, run verification and reporting remain usable.
Source files are included directly beside artifacts/ for a single extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_METADATA_FILES = (
    "config.json", "environment.json", "training_ids.json", "history.json",
    "history.csv", "status.json", "best.pt.sha256", "latest.pt.sha256",
)
RUN_FILES = ("best.pt", "latest.pt", *RUN_METADATA_FILES)
CHECKPOINT_PATHS = {"artifacts/final/selected.pt"} | {
    f"artifacts/runs/{name}/{kind}.pt" for name in ["A", "B", "C"] for kind in ["best", "latest"]
}
EVALUATION_FILES = ("metrics.json", "per_image.csv", "ranked_examples.png", "example_selection.json")
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required submission evidence is missing or empty: {path}")
    return path


def evidence_tree(directory, suffixes):
    directory = Path(directory)
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes
                  and not any(part.startswith(".") for part in path.relative_to(directory).parts))


def app_qa_support_files(directory):
    """Keep small QA proof/harness files; large generated inputs are recreatable."""
    directory = Path(directory)
    suffixes = {".json", ".txt", ".log", ".py", ".cjs"}
    paths = [path for path in directory.iterdir()
             if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes]
    paths += evidence_tree(directory / "logs", suffixes)
    paths += evidence_tree(directory / "fixtures", suffixes | MEDIA_SUFFIXES)
    return sorted(path for path in paths if path.stat().st_size <= 1024 * 1024)


def collect_evidence(root=PROJECT_ROOT):
    """Validate measured identities and return an explicit allowlist of files."""
    from petseg.inference import validate_deployment
    from petseg.train import inspect_run

    root = Path(root).resolve()
    artifacts = root / "artifacts"
    manifest = require_file(artifacts / "data/splits.json")
    selected = require_file(artifacts / "final/selected.pt")
    checkpoint_digest = digest(selected)
    sidecar = require_file(artifacts / "final/selected.pt.sha256")
    if sidecar.read_text(encoding="ascii").strip() != checkpoint_digest:
        raise ValueError("Selected checkpoint differs from its SHA-256 sidecar")
    selection_path = require_file(artifacts / "final/selection.json")
    decision_path = require_file(artifacts / "final/decision.json")
    selection, decision = read_json(selection_path), read_json(decision_path)
    if (selection.get("checkpoint_sha256") != checkpoint_digest
            or selection.get("manifest_sha256") != digest(manifest)):
        raise ValueError("Selection does not identify the supplied checkpoint and split manifest")
    runs = {name: inspect_run(artifacts / "runs" / name, manifest=manifest, require_completed=True)
            for name in ["A", "B", "C"]}
    for name, run in runs.items():
        config = run["config"]
        if (config.get("diagnostic", True) or config.get("experiment") != name
                or config.get("frozen") != (name == "A") or config.get("augment") != (name == "C")):
            raise ValueError(f"Run {name} is not a final specified comparison experiment")
    expected_candidates = [{"experiment": name,
                            "validation_foreground_iou": runs[name]["best"]["best_iou"],
                            "epochs_completed": runs[name]["status"]["epochs_completed"],
                            "global_step": runs[name]["status"]["global_step"]}
                           for name in ["A", "B", "C"]]
    recorded_candidates = [{key: row.get(key) for key in expected_candidates[0]}
                           for row in selection.get("candidates", [])]
    winner = max(expected_candidates, key=lambda row: row["validation_foreground_iou"])["experiment"]
    comparable = [{key: value for key, value in run["config"].items()
                   if key not in {"experiment", "frozen", "augment"}}
                  for run in runs.values()]
    initializations = [run["best"].get("initial_model_sha256") for run in runs.values()]
    if (recorded_candidates != expected_candidates or selection.get("selected_experiment") != winner
            or not all(config == comparable[0] for config in comparable)
            or not initializations[0] or not all(value == initializations[0] for value in initializations)
            or digest(artifacts / "runs" / winner / "best.pt") != checkpoint_digest):
        raise ValueError("Selection is stale or its comparison runs are not comparable")
    deployment = validate_deployment(decision, checkpoint_digest, runs[winner]["best"], digest(manifest))
    protocol_path = require_file(artifacts / "execution_protocol.json")
    freeze_path = require_file(artifacts / "final/freeze.json")
    protocol, freeze = read_json(protocol_path), read_json(freeze_path)
    if protocol.get("manifest_sha256") != digest(manifest):
        raise ValueError("Execution protocol does not identify the supplied split manifest")
    for key, expected in [("checkpoint_sha256", checkpoint_digest),
                          ("manifest_sha256", digest(manifest)),
                          ("decision_sha256", digest(decision_path))]:
        if key in freeze and freeze[key] != expected:
            raise ValueError(f"Frozen test policy has a different {key}")
    files = [manifest, selected, sidecar, selection_path, decision_path, protocol_path, freeze_path]
    for name in ["audit.json", "exclusions.json", "ANNOTATION_README.txt", "audit_examples.png"]:
        files.append(require_file(artifacts / "data" / name))
    audit = read_json(artifacts / "data/audit.json")
    if audit.get("manifest_sha256") != digest(manifest):
        raise ValueError("Dataset audit does not identify the supplied split manifest")
    for name in ["A", "B", "C"]:
        files.extend(require_file(artifacts / "runs" / name / filename) for filename in RUN_FILES)
    for size in sorted({256, deployment}):
        evaluation = artifacts / "evaluations" / f"test_{size}"
        files.extend(require_file(evaluation / filename) for filename in EVALUATION_FILES)
        metrics = read_json(evaluation / "metrics.json")
        if (metrics.get("split") != "test" or metrics.get("input_resolution") != size
                or metrics.get("checkpoint_sha256") != checkpoint_digest
                or metrics.get("manifest_sha256") != digest(manifest)
                or metrics.get("decision_sha256") != digest(decision_path)
                or metrics.get("images") != len(read_json(manifest)["splits"]["test"])):
            raise ValueError(f"Test evaluation at {size} does not match the frozen model/data/decision")
    for name in ["results.md", "evidence_status.json", "learning_curves.png"]:
        files.append(require_file(artifacts / "reports" / name))
    files.extend(evidence_tree(artifacts / "reports", {".md", ".json", ".png"}))
    status = read_json(artifacts / "reports/evidence_status.json")
    if not all(status.get(name) is True for name in
               ["experiment_A", "experiment_B", "experiment_C", "optimization", "test_evaluation"]):
        raise ValueError("Generated report still marks required measured evidence pending")
    qualitative = artifacts / "qualitative"
    sources = require_file(qualitative / "sources.json")
    if not read_json(sources):
        raise ValueError("Qualitative source attribution is empty")
    inputs = evidence_tree(qualitative / "inputs", MEDIA_SUFFIXES)
    outputs = evidence_tree(qualitative / "outputs", MEDIA_SUFFIXES | {".json", ".txt", ".md"})
    metadata = [path for path in outputs if path.name == "metadata.json"]
    if not inputs or not metadata or not any(path.suffix.lower() in MEDIA_SUFFIXES for path in outputs):
        raise FileNotFoundError("Qualitative inputs, predicted images and prediction metadata are required")
    if any(read_json(path).get("checkpoint_sha256") != checkpoint_digest
           or read_json(path).get("decision_sha256") != digest(decision_path) for path in metadata):
        raise ValueError("Qualitative predictions used a different selected checkpoint or deployment decision")
    files.extend([sources, *inputs, *outputs])
    qa_root = artifacts / "app_qa"
    qa_path = require_file(qa_root / "qa.json")
    qa = read_json(qa_path)
    screenshots = evidence_tree(qa_root / "screenshots", MEDIA_SUFFIXES)
    downloads = evidence_tree(qa_root / "downloads", MEDIA_SUFFIXES | {".json"})
    if (qa.get("passed") is not True or qa.get("checkpoint_sha256") != checkpoint_digest
            or qa.get("decision_sha256") != digest(decision_path) or not screenshots or not downloads):
        raise ValueError("Successful trained-app QA, matching model/decision, screenshots and downloads are required")
    files.extend([qa_path, *screenshots, *downloads, *app_qa_support_files(qa_root)])
    # These explicitly labeled diagnostics are supporting pipeline evidence,
    # never required for final model selection and never include weight payloads.
    for name in ["final_overfit", "final_smoke"]:
        diagnostic = artifacts / "diagnostics" / name
        files.extend(diagnostic / filename for filename in RUN_METADATA_FILES
                     if (diagnostic / filename).is_file() and not (diagnostic / filename).is_symlink())
        files.extend(evidence_tree(diagnostic / "training_check", {".json", ".csv", ".png", ".md"}))
    return sorted(set(files)), checkpoint_digest


def source_entries(root, source_zip):
    """Reuse the source ZIP allowlist while placing runnable files at ZIP root."""
    root = Path(root).resolve()
    entries = []
    with zipfile.ZipFile(source_zip) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            path = (root / name).resolve()
            if ("\\" in name or Path(name).is_absolute() or ".." in Path(name).parts
                    or not path.is_relative_to(root)):
                raise ValueError(f"Unsafe source archive path: {name}")
            require_file(path)
            if hashlib.sha256(archive.read(info)).hexdigest() != digest(path):
                raise ValueError(f"Source changed after creating its ZIP; rerun packaging: {name}")
            entries.append((path, name))
    return entries


def write_bundle(output, entries, checkpoint_digest):
    """Write and verify every archived byte against an embedded SHA-256 inventory."""
    output = Path(output)
    if output.suffix.lower() != ".zip" or any(output.resolve() == Path(path).resolve() for path, _ in entries):
        raise ValueError("Submission output must be a .zip file separate from every input artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [name for _, name in entries]
    if len(names) != len(set(names)) or "SUBMISSION_INVENTORY.json" in names:
        raise ValueError("Submission archive contains duplicate file names")
    for path, name in entries:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts or "\\" in name:
            raise ValueError(f"Unsafe submission archive path: {name}")
        if Path(name).suffix.lower() in {".pt", ".pth"} and name not in CHECKPOINT_PATHS:
            raise ValueError(f"Only selected and final A/B/C best/latest checkpoints belong in the submission: {name}")
        require_file(path)
    records = [{"path": name, "bytes": Path(path).stat().st_size, "sha256": digest(path)}
               for path, name in sorted(entries, key=lambda entry: entry[1])]
    inventory = {"schema_version": 1, "selected_checkpoint_sha256": checkpoint_digest,
                 "inventory_excludes_itself": True, "files": records}
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, name in sorted(entries, key=lambda entry: entry[1]):
            archive.write(path, name)
        archive.writestr("SUBMISSION_INVENTORY.json", json.dumps(inventory, indent=2, ensure_ascii=False).encode("utf-8"))
    with zipfile.ZipFile(temporary) as archive:
        for record in records:
            data = archive.read(record["path"])
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise RuntimeError(f"Evidence changed during packaging: {record['path']}")
    temporary.replace(output)
    output.with_suffix(output.suffix + ".sha256").write_text(digest(output) + "\n", encoding="ascii")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/upgaming-petseg-submission.zip")
    args = parser.parse_args()
    files, checkpoint_digest = collect_evidence()
    # Build source last so documentation and QA fixes are included in this copy.
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/package_source.py")],
                   cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    source = PROJECT_ROOT / "artifacts/upgaming-petseg-source.zip"
    source_hash = source.with_suffix(".zip.sha256")
    if source_hash.read_text(encoding="ascii").strip() != digest(source):
        raise ValueError("Source archive SHA-256 does not match")
    entries = [(path, path.relative_to(PROJECT_ROOT).as_posix()) for path in files]
    entries += source_entries(PROJECT_ROOT, source)
    print(write_bundle(args.output, entries, checkpoint_digest))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Submission package not created: {exc}") from exc
