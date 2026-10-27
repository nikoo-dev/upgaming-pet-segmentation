"""Submission packaging integrity checks use synthetic bytes, never model claims."""
import hashlib
import json
import zipfile

import pytest

from scripts.package_submission import app_qa_support_files, collect_evidence, digest, source_entries, write_bundle


def test_submission_requires_measured_core_evidence(tmp_path):
    with pytest.raises(FileNotFoundError, match="splits.json"):
        collect_evidence(tmp_path)
    assert not list(tmp_path.rglob("*.zip"))


def test_submission_inventory_matches_archived_bytes(tmp_path):
    app = tmp_path / "app.py"
    app.write_text("# synthetic runnable source fixture")
    report = tmp_path / "docs/TECHNICAL_REPORT.md"
    report.parent.mkdir()
    report.write_text("Synthetic report fixture, not measured results")
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.write(app, "app.py")
        archive.write(report, "docs/TECHNICAL_REPORT.md")
    selected = tmp_path / "selected.pt"
    selected.write_bytes(b"synthetic checkpoint fixture; not a trained model")
    output = tmp_path / "submission.zip"
    entries = source_entries(tmp_path, source) + [(selected, "artifacts/final/selected.pt")]
    qa = tmp_path / "artifacts/app_qa"
    (qa / "fixtures").mkdir(parents=True)
    harness = qa / "run_browser_qa.cjs"
    harness.write_text("// synthetic browser harness fixture")
    proof = qa / "pet_downloads.json"
    proof.write_text('{"synthetic": true}')
    small_fixture = qa / "fixtures/image.png"
    small_fixture.write_bytes(b"small synthetic image fixture")
    omitted = qa / "fixtures/large_image.png"
    omitted.write_bytes(b"x" * (1024 * 1024 + 1))
    invalid_weights = qa / "fixtures/invalid.pt"
    invalid_weights.write_bytes(b"deliberately invalid synthetic checkpoint")
    support = app_qa_support_files(qa)
    assert set(support) == {harness, proof, small_fixture}
    entries += [(path, path.relative_to(tmp_path).as_posix()) for path in support]
    for name in ["A", "B", "C"]:
        for kind in ["best", "latest"]:
            state = tmp_path / "runs" / name / f"{kind}.pt"
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_bytes(f"synthetic {name}/{kind}; not a trained model".encode("ascii"))
            entries.append((state, f"artifacts/runs/{name}/{kind}.pt"))
    write_bundle(output, entries, digest(selected))
    with zipfile.ZipFile(output) as archive:
        inventory = json.loads(archive.read("SUBMISSION_INVENTORY.json"))
        assert inventory["selected_checkpoint_sha256"] == digest(selected)
        assert "app.py" in archive.namelist() and "docs/TECHNICAL_REPORT.md" in archive.namelist()
        assert not any(name.endswith(".zip") for name in archive.namelist())
        assert len([name for name in archive.namelist() if name.endswith(".pt")]) == 7
        assert len(inventory["files"]) == 12
        for record in inventory["files"]:
            data = archive.read(record["path"])
            assert record["bytes"] == len(data)
            assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert output.with_suffix(".zip.sha256").read_text().strip() == digest(output)


@pytest.mark.parametrize("archive_path", ["artifacts/diagnostics/final_overfit/best.pt", "cache/weights.pth", "../escape.txt"])
def test_submission_rejects_extra_weights_and_unsafe_paths(tmp_path, archive_path):
    source = tmp_path / "fixture.txt"
    source.write_text("synthetic fixture")
    output = tmp_path / "submission.zip"
    with pytest.raises(ValueError):
        write_bundle(output, [(source, archive_path)], "unused synthetic hash")
    assert not output.exists()


def test_submission_cannot_overwrite_an_input(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"source must survive")
    with pytest.raises(ValueError, match="separate"):
        write_bundle(source, [(source, source.name)], "unused synthetic hash")
    assert source.read_bytes() == b"source must survive"
