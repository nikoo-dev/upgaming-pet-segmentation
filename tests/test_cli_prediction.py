import sys

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.cli import main
from petseg.inference import load_upload, predict
from petseg.utils import read_json, sha256, write_json


@pytest.mark.parametrize("manual_size", [None, 64])
def test_cli_prediction_matches_shared_path_and_records_frozen_decision(tmp_path, monkeypatch, manual_size):
    import petseg.inference as inference

    class SyntheticModel(torch.nn.Module):
        def forward(self, tensor):
            logits = torch.zeros((1, 2, 2, 2))
            logits[:, 1, :, 1] = 2
            return {"out": logits}

    state = {"config": {"experiment": "A", "diagnostic": False}, "manifest_sha256": "synthetic-split",
             "epoch": 0, "global_step": 1}
    monkeypatch.setattr(inference, "load_model", lambda *_: (SyntheticModel().eval(), state))
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic CLI fixture, not a trained model")
    decision_path = tmp_path / "decision.json"
    decision = {"schema_version": 1, "checkpoint_sha256": sha256(checkpoint),
                "manifest_sha256": state["manifest_sha256"], "baseline_resolution": 256,
                "deployment_resolution": 192,
                "results": [{"size": size, "validation_foreground_iou": iou, "end_to_end": {"median_ms": ms}}
                            for size, iou, ms in [(256, .8, 20), (192, .79, 15), (128, .75, 10)]]}
    write_json(decision_path, decision)
    source = Image.new("RGB", (7, 11), "red")
    exif = source.getexif()
    exif[274] = 6
    image_path = tmp_path / "input.jpg"
    source.save(image_path, "JPEG", exif=exif)
    output = tmp_path / "prediction"
    args = ["petseg", "predict", str(image_path), "--checkpoint", str(checkpoint),
            "--decision", str(decision_path), "--output", str(output)]
    if manual_size is not None:
        args.extend(["--size", str(manual_size)])
        decision_path.unlink()  # A manual override must not need a frozen decision file.
    monkeypatch.setattr(sys, "argv", args)
    main()

    expected = predict(load_upload(image_path.read_bytes()), checkpoint, manual_size or 256,
                       decision=decision if manual_size is None else None)
    metadata = read_json(output / "metadata.json")
    assert metadata["checkpoint_sha256"] == sha256(checkpoint)
    assert [metadata["width"], metadata["height"]] == [11, 7]
    if manual_size is None:
        assert metadata["decision_sha256"] == sha256(decision_path)
        assert metadata["resolution_source"] == "frozen deployment decision"
        assert metadata["input_resolution"] == 192
    else:
        assert "decision_sha256" not in metadata
        assert metadata["resolution_source"] == "manual override"
        assert metadata["input_resolution"] == manual_size
    for name, expected_pixels in [("mask", expected["mask"] * 255),
                                  ("overlay", np.asarray(expected["overlay"])),
                                  ("composite", np.asarray(expected["composite"]))]:
        with Image.open(output / f"{name}.png") as actual:
            assert actual.size == (11, 7)
            assert np.array_equal(np.asarray(actual), expected_pixels)
