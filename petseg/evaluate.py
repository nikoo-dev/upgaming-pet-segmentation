"""Original-resolution evaluation with ignored annotation boundaries."""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import annotation_root, load_pair
from .inference import predict_mask, render_outputs
from .metrics import confusion, scores
from .transforms import map_trimap
from .utils import read_json, write_json
from .visualize import contact_sheet


@torch.inference_mode()
def evaluate_model(model, root, records, size=256, device="cpu"):
    if not records:
        raise ValueError("Cannot evaluate an empty split")
    model.eval()
    matrix = np.zeros((2, 2), dtype=np.int64)
    ignored_pixels = 0
    rows = []
    base = annotation_root(root)
    for r in records:
        image, raw = load_pair(base, r)
        mask = predict_mask(model, image, size, device)
        target = map_trimap(raw)
        m = confusion(mask, target)
        ignored_pixels += int((target == 255).sum())
        matrix += m
        rows.append({"id": r["id"], "breed": r["breed"], "species": r["species"],
                     "foreground_iou": scores(m)["foreground_iou"]})
    valid_pixels = int(matrix.sum())
    return {**scores(matrix), "images": len(rows), "input_resolution": size,
            "valid_pixels": valid_pixels, "ignored_pixels": ignored_pixels,
            "total_pixels": valid_pixels + ignored_pixels}, rows


def save_evaluation(output, metrics, rows, model, root, records, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "breed", "species", "foreground_iou"])
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted([r for r in rows if r["foreground_iou"] is not None], key=lambda r: (r["foreground_iou"], r["id"]))
    if not ranked:
        return
    selected = [("worst", ranked[0]), ("median", ranked[len(ranked) // 2]), ("best", ranked[-1])]
    lookup = {r["id"]: r for r in records}
    views = []
    for name, row in selected:
        image, raw = load_pair(annotation_root(root), lookup[row["id"]])
        mask = predict_mask(model, image, metrics["input_resolution"], device)
        overlay, _ = render_outputs(image, mask)
        target = map_trimap(raw)
        rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
        rgb[target == 1] = [255, 100, 130]
        rgb[target == 255] = [255, 220, 0]
        views.append((f"{name}: {row['id']} IoU={row['foreground_iou']:.4f} | original / target / prediction", [image, Image.fromarray(rgb), overlay]))
    contact_sheet(views, output / "ranked_examples.png")
    write_json(output / "example_selection.json", {"rule": "ascending (foreground_iou, id); indices 0, n//2, n-1; undefined omitted", "examples": selected})
