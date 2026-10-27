"""Reproduce predeclared qualitative examples after frozen test evaluation.

Uses saved per-image IoUs only to select six worst cases, the upper median and
the best case. It makes no model/deployment changes and computes no new scores.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from petseg.data import annotation_root, load_pair
from petseg.inference import load_model, predict_mask, render_outputs, validate_deployment
from petseg.transforms import map_trimap
from petseg.utils import read_json, sha256, write_json
from petseg.visualize import contact_sheet


PROJECT = Path(__file__).resolve().parents[1]


def build_examples(root, artifacts):
    artifacts = Path(artifacts)
    checkpoint = artifacts / "final/selected.pt"
    decision_path = artifacts / "final/decision.json"
    manifest_path = artifacts / "data/splits.json"
    for path in [checkpoint, checkpoint.with_suffix(".pt.sha256"), decision_path, manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    digest = sha256(checkpoint)
    if checkpoint.with_suffix(".pt.sha256").read_text(encoding="ascii").strip() != digest:
        raise ValueError("Selected checkpoint differs from its SHA-256 sidecar")
    decision = read_json(decision_path)
    size = decision["deployment_resolution"]
    evaluation = artifacts / "evaluations" / f"test_{size}"
    metrics_path, csv_path = evaluation / "metrics.json", evaluation / "per_image.csv"
    for path in [metrics_path, csv_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Complete frozen test evaluation first; missing {path}")
    metrics = read_json(metrics_path)
    manifest = read_json(manifest_path)
    records = manifest["splits"]["test"]
    if (metrics.get("split") != "test" or metrics.get("input_resolution") != size
            or metrics.get("checkpoint_sha256") != digest
            or metrics.get("manifest_sha256") != sha256(manifest_path)
            or metrics.get("decision_sha256") != sha256(decision_path)
            or metrics.get("images") != len(records)
            or metrics.get("boundary_policy") != "raw label 3 ignored"):
        raise ValueError("Saved test metrics do not match the frozen checkpoint, decision and full test split")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != len(records) or len({row["id"] for row in rows}) != len(rows)
            or {row["id"] for row in rows} != {record["id"] for record in records}):
        raise ValueError("Saved per-image CSV does not contain the full test split exactly once")
    defined = []
    for row in rows:
        if row["foreground_iou"] == "":
            continue
        value = float(row["foreground_iou"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Saved per-image IoU must be defined within [0, 1], or empty")
        defined.append({"id": row["id"], "saved_foreground_iou": value})
    ranked = sorted(defined, key=lambda row: (row["saved_foreground_iou"], row["id"]))
    if len(ranked) < 6:
        raise ValueError("At least six defined saved test IoUs are required")
    selected = [(f"worst_{index + 1}", row) for index, row in enumerate(ranked[:6])]
    selected += [("upper_median", ranked[len(ranked) // 2]), ("best", ranked[-1])]

    torch.set_num_threads(1)
    model, state = load_model(checkpoint)
    validate_deployment(decision, digest, state, sha256(manifest_path))
    lookup = {record["id"]: record for record in records}
    views, examples = [], []
    for role, row in selected:
        record = lookup[row["id"]]
        image, raw = load_pair(annotation_root(root), record)
        target = map_trimap(raw)
        target_rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
        target_rgb[target == 1] = [255, 100, 130]
        target_rgb[target == 255] = [255, 220, 0]
        mask = predict_mask(model, image, size)
        overlay, _ = render_outputs(image, mask)
        title = f"{role}: {row['id']} IoU={row['saved_foreground_iou']:.4f} | original / target / overlay"
        views.append((title, [image, Image.fromarray(target_rgb), overlay]))
        examples.append({"role": role, **row, "breed": record["breed"], "species": record["species"]})
    output = artifacts / "reports"
    contact_sheet(views, output / "error_examples.png")
    record = {
        "schema_version": 1, "split": "test", "input_resolution": size,
        "checkpoint_sha256": digest, "manifest_sha256": sha256(manifest_path),
        "decision_sha256": sha256(decision_path), "metrics_sha256": sha256(metrics_path),
        "per_image_csv_sha256": sha256(csv_path),
        "selection_rule": "ascending (saved foreground IoU, image ID), undefined omitted; first six, index n//2, last",
        "boundary_policy": "Raw trimap label 3 ignored in saved IoUs and shown yellow in targets",
        "scope": "Qualitative examples selected by the saved test ranking; no subgroup metrics or tuning",
        "examples": examples,
    }
    write_json(output / "error_examples.json", record)
    return output / "error_examples.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "data", help="Local Oxford-IIIT Pet data root")
    parser.add_argument("--artifacts", type=Path, default=PROJECT / "artifacts")
    args = parser.parse_args()
    print(build_examples(args.root, args.artifacts))


if __name__ == "__main__":
    main()
