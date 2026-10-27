import numpy as np
import pytest
import torch
from PIL import Image

from petseg.evaluate import evaluate_model
from petseg.inference import load_model


def test_original_pixel_aggregation_and_ignored_boundary(monkeypatch):
    import petseg.evaluate as evaluation

    pairs = {
        "large": (Image.new("RGB", (4, 2)), Image.fromarray(np.array([[1, 1, 1, 3], [2, 2, 2, 2]], dtype=np.uint8))),
        "small": (Image.new("RGB", (2, 2)), Image.fromarray(np.ones((2, 2), dtype=np.uint8))),
    }
    monkeypatch.setattr(evaluation, "load_pair", lambda base, record: pairs[record["id"]])

    class ForegroundModel(torch.nn.Module):
        def forward(self, tensor):
            # Low-resolution logits must still be restored to each original image.
            return {"out": torch.tensor([[[[0.]], [[1.]]]])}

    records = [{"id": name, "breed": 1, "species": 1} for name in pairs]
    metrics, rows = evaluate_model(ForegroundModel(), ".", records, 64)
    assert metrics["confusion_matrix"] == [[0, 4], [0, 7]]
    assert metrics["foreground_iou"] == pytest.approx(7 / 11)
    assert metrics["foreground_iou"] != pytest.approx(np.mean([row["foreground_iou"] for row in rows]))
    assert metrics["valid_pixels"] == 11
    assert metrics["ignored_pixels"] == 1
    assert metrics["total_pixels"] == 12
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(ForegroundModel(), ".", [], 64)


def test_non_dictionary_checkpoint_is_actionable(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    torch.save(torch.zeros(1), checkpoint)
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(checkpoint)
