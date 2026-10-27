"""Shared CPU/GPU prediction path. There are no automatic model downloads."""
from io import BytesIO
from pathlib import Path
import warnings
import pickle

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.nn import functional as F

from .model import build_model
from .transforms import CLASS_MAPPING, PREPROCESSING, orient_rgb, preprocess
from .utils import sha256

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def load_upload(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds the 10 MiB upload limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 20-megapixel decoded limit")
                image.load()
                return orient_rgb(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or unsafe image; upload a smaller valid JPEG or PNG") from exc


def load_model(checkpoint, device="cpu"):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train using petseg train or download your Colab artifacts.")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Incompatible checkpoint; expected a project checkpoint dictionary")
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema")
        if state.get("class_mapping") != CLASS_MAPPING or state.get("preprocessing") != PREPROCESSING:
            raise ValueError("Checkpoint preprocessing or class mapping does not match this application")
        if not isinstance(state.get("global_step"), int) or state["global_step"] < 1:
            raise ValueError("Checkpoint has no completed training steps")
        if not isinstance(state.get("config"), dict) or not isinstance(state.get("epoch"), int) or state["epoch"] < 0:
            raise ValueError("Incompatible checkpoint training metadata")
        model = build_model(pretrained=False)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        return model, state
    except (RuntimeError, KeyError, TypeError, pickle.UnpicklingError, EOFError, OSError) as exc:
        raise ValueError("Incompatible checkpoint; use a checkpoint produced by this project's train command") from exc


def validate_deployment(decision, checkpoint_digest, state, manifest_digest=None):
    """Require the measured policy and checkpoint to refer to the same final run."""
    from .benchmark import choose_resolution

    if not isinstance(decision, dict) or decision.get("schema_version") != 1:
        raise ValueError("Unsupported deployment decision; re-run petseg benchmark")
    if decision.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("Deployment decision does not match the checkpoint; re-run petseg benchmark")
    identity = state.get("manifest_sha256")
    if not identity or decision.get("manifest_sha256") != identity or (manifest_digest is not None and identity != manifest_digest):
        raise ValueError("Deployment decision, checkpoint and split manifest identities differ")
    config = state["config"]
    if config.get("diagnostic", True) or config.get("experiment") not in {"A", "B", "C"}:
        raise ValueError("A diagnostic checkpoint cannot be used as the final deployment model")
    if decision.get("baseline_resolution") != 256:
        raise ValueError("Deployment decision must use the 256 baseline")
    selected = choose_resolution(decision.get("results", []))
    if decision.get("deployment_resolution") != selected:
        raise ValueError("Deployment resolution differs from the measured validation/latency policy")
    return selected


def restore_logits(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


@torch.inference_mode()
def predict_mask(model, image, size=256, device="cpu"):
    tensor = preprocess(image, size).unsqueeze(0).to(device)
    logits = model(tensor)["out"]
    logits = restore_logits(logits, (image.height, image.width))
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def render_outputs(image, mask, color=(80, 160, 240)):
    rgb = np.array(image)
    overlay = rgb.astype(np.float32)
    overlay[mask == 1] = .55 * overlay[mask == 1] + .45 * np.array([255, 80, 120])
    composite = np.where(mask[..., None] == 1, rgb, np.array(color, dtype=np.uint8))
    return Image.fromarray(overlay.astype(np.uint8)), Image.fromarray(composite.astype(np.uint8))


def predict(image, checkpoint, size=256, color=(80, 160, 240), decision=None):
    model, state = load_model(checkpoint)
    digest = sha256(checkpoint)
    if decision is not None:
        size = validate_deployment(decision, digest, state)
    image = orient_rgb(image)
    mask = predict_mask(model, image, size)
    overlay, composite = render_outputs(image, mask, color)
    return {"mask": mask, "overlay": overlay, "composite": composite,
            "metadata": {"experiment": state["config"]["experiment"], "input_resolution": size,
                         "epoch": state["epoch"], "global_step": state["global_step"],
                         "checkpoint_sha256": digest, "width": image.width, "height": image.height,
                         "diagnostic": state["config"].get("diagnostic", True),
                         "resolution_source": "frozen deployment decision" if decision is not None else "manual override"}}
