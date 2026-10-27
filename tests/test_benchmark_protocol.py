from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.benchmark import benchmark
from petseg.utils import read_json, sha256, write_json


def test_benchmark_uses_same_validation_images_and_balances_all_timed_paths(tmp_path, monkeypatch):
    import petseg.benchmark as module

    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic protocol fixture only")
    manifest = tmp_path / "splits.json"
    records = [{"id": "validation_fixture", "breed": 1}]
    write_json(manifest, {"splits": {"val": records, "test": [{"id": "must_not_be_loaded"}]}})
    selection = tmp_path / "selection.json"
    write_json(selection, {"checkpoint_sha256": sha256(checkpoint), "manifest_sha256": sha256(manifest),
                           "selected_experiment": "A"})
    state = {"manifest_sha256": sha256(manifest), "config": {"diagnostic": False, "experiment": "A"}}
    events, loaded = [], []

    class ProtocolModel(torch.nn.Module):
        def forward(self, tensor):
            assert tensor.dtype == torch.float32 and tensor.device.type == "cpu" and tensor.shape[0] == 1
            events.append((tensor.shape[-1], "forward"))

    monkeypatch.setattr(module, "load_model", lambda *_: (ProtocolModel(), state))

    def load_pair(base, record):
        loaded.append(record["id"])
        return Image.new("RGB", (17, 13)), None

    monkeypatch.setattr(module, "load_pair", load_pair)
    monkeypatch.setattr(module, "preprocess", lambda image, size: torch.zeros((3, size, size)))

    def predict_mask(model, image, size):
        assert image.size == (17, 13)
        events.append((size, "end_to_end"))
        return np.zeros((13, 17), dtype=np.uint8)

    monkeypatch.setattr(module, "predict_mask", predict_mask)
    evaluated = []

    def evaluate(model, root, evaluation_records, size):
        evaluated.append((evaluation_records, size))
        return {"foreground_iou": .8}, []

    monkeypatch.setattr(module, "evaluate_model", evaluate)
    output = tmp_path / "decision.json"
    result = benchmark("unused", manifest, checkpoint, selection, output)
    assert loaded == ["validation_fixture"]
    assert evaluated == [(records, size) for size in [256, 192, 128]]
    warmup, measured = events[:60], events[60:]
    pairs = {(size, path) for size in [256, 192, 128] for path in ["forward", "end_to_end"]}
    assert Counter(warmup) == {pair: 10 for pair in pairs}
    assert Counter(measured) == {pair: 100 for pair in pairs}
    assert all(set(measured[i:i + 6]) == pairs for i in range(0, 600, 6))
    assert len({tuple(measured[i:i + 6]) for i in range(0, 600, 6)}) > 1
    assert result["environment"]["threads"] == 1
    assert result["checkpoint_sha256"] == sha256(checkpoint)
    assert read_json(output) == result

    # Even a matching selection file cannot excuse a checkpoint trained on another split.
    state["manifest_sha256"] = "another-split"
    with pytest.raises(ValueError, match="training splits"):
        benchmark("unused", manifest, checkpoint, selection, output)
