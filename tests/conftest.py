from pathlib import Path
import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.utils import write_json


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    torch.set_num_threads(1)
    def no_download(*args, **kwargs):
        raise AssertionError("Offline tests must never download weights")
    monkeypatch.setattr(torch.hub, "download_url_to_file", no_download)


@pytest.fixture
def miniature_data(tmp_path):
    root = tmp_path / "data"
    base = root / "oxford-iiit-pet"
    (base / "images").mkdir(parents=True)
    (base / "annotations/trimaps").mkdir(parents=True)
    splits = {"train": [], "val": [], "test": []}
    i = 0
    for split, count in [("train", 4), ("val", 2), ("test", 2)]:
        for j in range(count):
            name = f"pet_{i}"
            rgb = np.full((48, 64, 3), 20 + i, dtype=np.uint8)
            rgb[10:40, 15:50] = [200, 100 + i, 80]
            raw = np.full((48, 64), 2, dtype=np.uint8)
            raw[10:40, 15:50] = 1
            raw[9, 15:50] = 3
            Image.fromarray(rgb).save(base / "images" / (name + ".jpg"))
            Image.fromarray(raw).save(base / "annotations/trimaps" / (name + ".png"))
            splits[split].append({"id": name, "breed": 1 + j % 2, "species": 1,
                                  "content_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
            i += 1
    manifest = tmp_path / "splits.json"
    write_json(manifest, {"schema_version": 1, "splits": splits, "seed": 42})
    return root, manifest
