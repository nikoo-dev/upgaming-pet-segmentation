import numpy as np
import pytest
import torch
from PIL import Image

from petseg.data import grouped_split, load_pair, stratified_cap, validate_splits
from petseg.metrics import confusion, scores
from petseg.transforms import PairTransform, map_trimap, paired_flip


def test_exact_mapping_and_reject_invalid():
    assert map_trimap(np.array([[1, 2, 3]], dtype=np.uint8)).tolist() == [[1, 0, 255]]
    with pytest.raises(ValueError):
        map_trimap(np.array([[0, 1]]))


def test_nearest_resize_and_pair_alignment():
    mask = Image.fromarray(np.array([[1, 2], [3, 2]], dtype=np.uint8))
    image = Image.fromarray(np.array([[[255, 0, 0], [0, 0, 0]], [[0, 255, 0], [0, 0, 0]]], dtype=np.uint8))
    im, ma = paired_flip(image, mask)
    assert np.asarray(im)[0, 1, 0] == 255
    assert np.asarray(ma)[0, 1] == 1
    tensor, target = PairTransform(32)(im, ma)
    assert set(target.unique().tolist()) == {0, 1, 255}
    assert tensor.shape == (3, 32, 32)
    assert target[0, -1] == 1


def test_grouped_split_and_test_duplicate_exclusion():
    records = [{"id": f"b{b}_{i}", "breed": b, "content_sha256": f"b{b}_{i}"} for b in [1, 2] for i in range(10)]
    records.append({"id": "duplicate", "breed": 1, "content_sha256": "b1_0"})
    test = [{"id": "official", "breed": 1, "content_sha256": "b1_9"}]
    splits, excluded = grouped_split(records, test)
    validate_splits(splits)
    assert any(r["id"] == "b1_9" for r in excluded)
    assert splits == grouped_split(records, test)[0]
    assert len(stratified_cap(splits["train"], 4)) == 4
    assert {r["breed"] for r in stratified_cap(splits["train"], 4)} == {1, 2}


def test_content_leakage_rejected():
    r = {"id": "a", "content_sha256": "same"}
    with pytest.raises(ValueError, match="Content leakage"):
        validate_splits({"train": [r], "val": [{**r, "id": "b"}]})


def test_pair_exif_orientation_preserves_annotation_geometry(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations/trimaps").mkdir(parents=True)
    image = Image.new("RGB", (5, 3), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(tmp_path / "images/pet.jpg", exif=exif)
    raw = np.array([[1, 2, 2, 2, 2], [1, 3, 2, 2, 2], [1, 1, 1, 2, 2]], dtype=np.uint8)
    Image.fromarray(raw).save(tmp_path / "annotations/trimaps/pet.png")
    oriented, mask = load_pair(tmp_path, {"id": "pet"})
    assert oriented.size == mask.size == (3, 5)
    assert np.array_equal(np.asarray(mask), np.rot90(raw, k=-1))
    assert oriented.getexif().get(274, 1) == 1


def test_metric_hand_calculation_and_ignored_pixels():
    target = np.array([[0, 0, 1, 1, 255]])
    prediction = np.array([[0, 1, 0, 1, 1]], dtype=np.uint8)
    m = confusion(prediction, target)
    assert m.tolist() == [[1, 1], [1, 1]]
    s = scores(m)
    assert s["foreground_iou"] == pytest.approx(1 / 3)
    assert s["foreground_dice"] == .5
    assert s["pixel_accuracy"] == .5
    assert scores(confusion(np.array([0, 1]), np.array([0, 1])))["mean_iou"] == 1
    assert scores(confusion(np.array([1, 0]), np.array([0, 1])))["foreground_iou"] == 0
    empty = scores(confusion(np.array([1]), np.array([255])))
    assert empty["foreground_iou"] is None and empty["mean_iou"] is None
    bg_only = scores(confusion(np.array([0]), np.array([0])))
    assert bg_only["foreground_iou"] is None and bg_only["mean_iou"] == 1
