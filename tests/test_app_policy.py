from pathlib import Path
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from petseg.benchmark import choose_resolution
from petseg.report import build_report
from petseg.inference import validate_deployment
from petseg.utils import sha256, write_json


def row(size, iou, latency):
    return {"size": size, "validation_foreground_iou": iou, "end_to_end": {"median_ms": latency}}


def test_resolution_quality_and_speed_policy():
    assert choose_resolution([row(256, .8, 20), row(192, .79, 15), row(128, .75, 10)]) == 192
    assert choose_resolution([row(256, .8, 20), row(192, .79, 25), row(128, .78, 10)]) == 128
    assert choose_resolution([row(256, .8, 20), row(192, .79, 25), row(128, .70, 10)]) == 256


@pytest.mark.parametrize("results", [[], [row(256, .8, 20)] * 3,
                        [row(256, None, 20), row(192, .79, 15), row(128, .75, 10)],
                        [row(256, .8, 20), row(192, .79, float("nan")), row(128, .75, 10)]])
def test_resolution_policy_rejects_missing_or_undefined_measurements(results):
    with pytest.raises(ValueError, match="Resolution comparison"):
        choose_resolution(results)


def deployment_fixture(tmp_path):
    # Deliberately synthetic protocol data; this is never a saved model-quality result.
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic app test fixture, not a trained checkpoint")
    state = {"config": {"experiment": "A", "diagnostic": False}, "manifest_sha256": "synthetic-manifest"}
    decision = {"schema_version": 1, "checkpoint_sha256": sha256(checkpoint),
                "manifest_sha256": state["manifest_sha256"], "baseline_resolution": 256,
                "deployment_resolution": 192,
                "results": [row(256, .8, 20), row(192, .79, 15), row(128, .75, 10)]}
    return checkpoint, state, decision


def test_deployment_rejects_wrong_split_diagnostic_and_changed_policy(tmp_path):
    checkpoint, state, decision = deployment_fixture(tmp_path)
    assert validate_deployment(decision, sha256(checkpoint), state) == 192
    with pytest.raises(ValueError, match="manifest identities"):
        validate_deployment(decision, sha256(checkpoint), state, "other-manifest")
    state["config"]["diagnostic"] = True
    with pytest.raises(ValueError, match="diagnostic"):
        validate_deployment(decision, sha256(checkpoint), state)
    state["config"]["diagnostic"] = False
    decision["deployment_resolution"] = 128
    with pytest.raises(ValueError, match="measured"):
        validate_deployment(decision, sha256(checkpoint), state)


def test_app_missing_checkpoint_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("PETSEG_CHECKPOINT", str(tmp_path / "missing.pt"))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert "checkpoint missing" in app.error[0].value.lower()


def test_app_synthetic_upload_and_download_dimensions(tmp_path, monkeypatch):
    import streamlit as st
    import petseg.inference as inference

    checkpoint, state, decision = deployment_fixture(tmp_path)
    decision_path = tmp_path / "decision.json"
    write_json(decision_path, decision)
    monkeypatch.setenv("PETSEG_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("PETSEG_DECISION", str(decision_path))

    class SyntheticModel(torch.nn.Module):
        def forward(self, tensor):
            logits = torch.zeros((1, 2, tensor.shape[-2], tensor.shape[-1]))
            logits[:, 1, :, tensor.shape[-1] // 2:] = 2
            return {"out": logits}

    monkeypatch.setattr(inference, "load_model", lambda *_: (SyntheticModel().eval(), state))
    source = Image.new("RGB", (7, 11), "red")
    exif = source.getexif()
    exif[274] = 6
    upload = BytesIO()
    source.save(upload, "JPEG", exif=exif)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    downloads = {}
    original_download = st.download_button

    def capture_download(label, data, file_name, mime):
        downloads[file_name] = data
        return original_download(label, data, file_name, mime)

    monkeypatch.setattr(st, "download_button", capture_download)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception and not app.error
    assert set(downloads) == {"pet_mask.png", "pet_overlay.png", "pet_composite.png"}
    for data in downloads.values():
        with Image.open(BytesIO(data)) as image:
            assert image.size == (11, 7)
            assert image.format == "PNG"
    mask = np.array(Image.open(BytesIO(downloads["pet_mask.png"])))
    assert set(np.unique(mask)) == {0, 255}
    composite = np.array(Image.open(BytesIO(downloads["pet_composite.png"])))
    assert (composite[mask == 0] == [80, 160, 240]).all()

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: BytesIO(b"invalid image"))
    app.run(timeout=20)
    assert not app.exception and "could not process image" in app.error[0].value.lower()


def test_app_mismatched_decision_is_actionable(tmp_path, monkeypatch):
    import petseg.inference as inference

    checkpoint, state, decision = deployment_fixture(tmp_path)
    decision["manifest_sha256"] = "wrong-split"
    decision_path = tmp_path / "decision.json"
    write_json(decision_path, decision)
    monkeypatch.setenv("PETSEG_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("PETSEG_DECISION", str(decision_path))
    monkeypatch.setattr(inference, "load_model", lambda *_: (None, state))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception and "manifest identities" in app.error[0].value.lower()


def test_report_does_not_invent_results(tmp_path):
    status = build_report(tmp_path)
    assert not any(status.values())
    assert "pending" in (tmp_path / "reports/results.md").read_text()
