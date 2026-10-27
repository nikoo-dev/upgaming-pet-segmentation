from io import BytesIO
import random

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.inference import load_model, load_upload, predict_mask, restore_logits
from petseg.model import build_model, configure_training, segmentation_loss, training_mode
from petseg.train import DEFAULT, inspect_run, train
from petseg.transforms import CLASS_MAPPING, PREPROCESSING
from petseg.utils import read_json, rng_state, restore_rng, save_checkpoint, seed_all, write_json


def checkpoint_state(model):
    return {"schema_version": 1, "model": model.state_dict(), "global_step": 1, "epoch": 0,
            "config": {"experiment": "synthetic_test", "diagnostic": True},
            "preprocessing": PREPROCESSING, "class_mapping": CLASS_MAPPING}


def test_forward_loss_gradient_and_frozen_batchnorm():
    model = build_model(False)
    configure_training(model, True)
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    training_mode(model, True)
    head_before = next(model.classifier.parameters()).detach().clone()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=.1)
    logits = model(torch.randn(2, 3, 64, 64))["out"]
    assert logits.shape == (2, 2, 64, 64)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    loss = segmentation_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(head_before, next(model.classifier.parameters()))
    assert all(torch.equal(v, before[k]) for k, v in model.backbone.state_dict().items())
    assert all(p.grad is None for p in model.backbone.parameters())
    assert segmentation_loss(logits, torch.full_like(target, 255)) is None
    with pytest.raises(ValueError):
        segmentation_loss(logits, torch.full_like(target, 9))


def test_checkpoint_parity_and_original_dimensions(tmp_path):
    model = build_model(False).eval()
    image = Image.new("RGB", (83, 49), "red")
    expected = predict_mask(model, image, 64)
    path = tmp_path / "test.pt"
    save_checkpoint(path, checkpoint_state(model))
    restored, _ = load_model(path)
    actual = predict_mask(restored, image, 64)
    assert actual.shape == (49, 83) and actual.dtype == np.uint8
    assert np.array_equal(expected, actual)
    assert path.with_suffix(".pt.sha256").exists()
    with pytest.raises(FileNotFoundError, match="Train"):
        load_model(tmp_path / "missing.pt")
    wrong = checkpoint_state(model)
    wrong["class_mapping"] = {}
    save_checkpoint(path, wrong)
    with pytest.raises(ValueError, match="mapping"):
        load_model(path)
    path.write_bytes(b"not a valid torch checkpoint")
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(path)


def encoded(image, fmt="PNG", **kwargs):
    stream = BytesIO()
    image.save(stream, fmt, **kwargs)
    return stream.getvalue()


def test_upload_modes_exif_limits_and_invalid(monkeypatch):
    import petseg.inference as module
    for mode in ["L", "RGBA"]:
        assert load_upload(encoded(Image.new(mode, (7, 11)))).mode == "RGB"
    image = Image.new("RGB", (7, 11))
    exif = image.getexif()
    exif[274] = 6
    assert load_upload(encoded(image, "JPEG", exif=exif)).size == (11, 7)
    with pytest.raises(ValueError, match="Invalid"):
        load_upload(b"not an image")
    monkeypatch.setattr(module, "MAX_BYTES", 2)
    with pytest.raises(ValueError, match="10 MiB"):
        load_upload(b"123")
    monkeypatch.setattr(module, "MAX_BYTES", 100000)
    monkeypatch.setattr(module, "MAX_PIXELS", 10)
    with pytest.raises(ValueError, match="20-megapixel"):
        load_upload(encoded(image))


def test_rng_restoration():
    seed_all(42)
    generator = torch.Generator().manual_seed(42)
    state = rng_state(generator)
    expected = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    restore_rng(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2]) and torch.equal(expected[3], actual[3])


def test_json_serialization_uses_portable_utf8_lf(tmp_path):
    path = tmp_path / "portable.json"
    value = {"label": "ცხოველი", "ids": ["cat", "dog"]}
    write_json(path, value)
    raw = path.read_bytes()
    assert raw == '{\n  "label": "ცხოველი",\n  "ids": [\n    "cat",\n    "dog"\n  ]\n}'.encode("utf-8")
    assert b"\r\n" not in raw
    assert read_json(path) == value


def test_logits_resize_precedes_argmax():
    # Hard-mask nearest resizing switches at x=2; confidence-aware logit
    # interpolation correctly keeps class 0 until x=3 in this example.
    logits = torch.tensor([[[[10., 0.]], [[0., 1.]]]])
    prediction = restore_logits(logits, (1, 4)).argmax(1)
    assert prediction.tolist() == [[[0, 0, 0, 1]]]


@pytest.mark.parametrize("pretrained", [False, True])
def test_resume_matches_uninterrupted_training(miniature_data, tmp_path, monkeypatch, pretrained):
    import petseg.train as module
    root, manifest = miniature_data
    constructor_paths, initial_models = [], []
    original_build = module.build_model

    def offline_build(pretrained=False):
        constructor_paths.append(pretrained)
        # Exercise different RNG consumption for new-pretrained and resume paths
        # without fetching external weights in an offline correctness test.
        if pretrained:
            torch.rand(17)
        model = original_build(False)
        initial_models.append({k: value.clone() for k, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(module, "build_model", offline_build)
    config = {**DEFAULT, "experiment": "diagnostic", "frozen": False, "augment": True,
              "diagnostic": True, "pretrained": pretrained, "size": 64, "batch_size": 2,
              "epochs": 2, "min_epochs": 1}
    train(root, manifest, tmp_path / "full", config)
    original_save = module.save_checkpoint

    class Interrupted(Exception):
        pass

    def interrupt_after_epoch(path, state):
        original_save(path, state)
        if path.name == "latest.pt" and state["epoch"] == 0:
            raise Interrupted()

    monkeypatch.setattr(module, "save_checkpoint", interrupt_after_epoch)
    with pytest.raises(Interrupted):
        train(root, manifest, tmp_path / "resumed", config)
    monkeypatch.setattr(module, "save_checkpoint", original_save)
    train(root, manifest, tmp_path / "resumed", config, resume=tmp_path / "resumed/latest.pt")
    full = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    assert full["global_step"] == resumed["global_step"] == 4
    assert full["bad_epochs"] == resumed["bad_epochs"]
    assert constructor_paths == [pretrained, pretrained, False]
    assert all(torch.equal(value, initial_models[1][key]) for key, value in initial_models[0].items())
    assert all(torch.equal(v, resumed["model"][k]) for k, v in full["model"].items())
    for a, b in zip(full["history"], resumed["history"]):
        assert a["train_loss"] == b["train_loss"]
        assert a["validation_foreground_iou"] == b["validation_foreground_iou"]
    assert inspect_run(tmp_path / "resumed", config, manifest, require_completed=True)["completed"]


@pytest.mark.parametrize("termination", ["epochs", "early_stopping", "max_steps"])
def test_terminal_checkpoint_recovers_missing_status(miniature_data, tmp_path, monkeypatch, termination):
    import petseg.train as module
    root, manifest = miniature_data
    config = {**DEFAULT, "experiment": "diagnostic", "frozen": True, "augment": False,
              "diagnostic": True, "pretrained": False, "size": 64, "batch_size": 2,
              "epochs": 1 if termination == "epochs" else 4, "min_epochs": 1,
              "patience": 1, "max_steps": 2 if termination == "max_steps" else None}
    monkeypatch.setattr(module, "evaluate_model", lambda *args: ({"foreground_iou": .5, "mean_iou": .5}, []))
    original_write = module.write_json

    def fail_status(path, value):
        if path.name == "status.json":
            raise InterruptedError("Session stopped before completion status commit")
        original_write(path, value)

    monkeypatch.setattr(module, "write_json", fail_status)
    output = tmp_path / termination
    with pytest.raises(InterruptedError):
        train(root, manifest, output, config)
    previous = torch.load(output / "latest.pt", weights_only=True)
    assert not inspect_run(output, config, manifest)["completed"]
    monkeypatch.setattr(module, "write_json", original_write)
    train(root, manifest, output, config, resume=output / "latest.pt")
    recovered = inspect_run(output, config, manifest, require_completed=True)
    assert recovered["latest"]["global_step"] == previous["global_step"]
    assert recovered["latest"]["history"] == previous["history"]


def test_run_validation_rejects_stale_status_and_repairs_best(miniature_data, tmp_path, monkeypatch):
    import petseg.train as module
    root, manifest = miniature_data
    config = {**DEFAULT, "experiment": "diagnostic", "frozen": True, "augment": False,
              "diagnostic": True, "pretrained": False, "size": 64, "batch_size": 2,
              "epochs": 1, "min_epochs": 1}
    output = tmp_path / "run"
    train(root, manifest, output, config)
    status = read_json(output / "status.json")
    write_json(output / "status.json", {**status, "global_step": status["global_step"] + 1})
    with pytest.raises(ValueError, match="completion status"):
        inspect_run(output, config, manifest, require_completed=True)
    # A complete atomic latest plus no status is recoverable even when best was
    # never committed. This is also the first-epoch bootstrap crash window.
    (output / "status.json").unlink()
    (output / "best.pt").unlink()
    (output / "best.pt.sha256").unlink()
    assert inspect_run(output, config, manifest)["best"] is None
    train(root, manifest, output, config, resume=output / "latest.pt")
    assert inspect_run(output, config, manifest, require_completed=True)["completed"]
    state = torch.load(output / "best.pt", weights_only=True)
    next(iter(state["model"].values())).add_(1)
    save_checkpoint(output / "best.pt", state)
    with pytest.raises(ValueError, match="completion status"):
        inspect_run(output, config, manifest, require_completed=True)
