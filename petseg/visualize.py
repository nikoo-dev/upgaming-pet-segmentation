"""Deterministic evidence images; no invented examples or labels."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from .data import load_pair
from .inference import render_outputs
from .transforms import map_trimap


def contact_sheet(rows, output):
    canvas = Image.new("RGB", (768, len(rows) * 280), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, images) in enumerate(rows):
        draw.text((8, i * 280 + 4), title, fill="black")
        for j, image in enumerate(images):
            tile = image.convert("RGB")
            tile.thumbnail((250, 250))
            canvas.paste(tile, (j * 256, i * 280 + 25))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def audit_examples(base, records, output):
    rows = []
    for r in records:
        image, raw = load_pair(base, r)
        target = map_trimap(raw)
        view = np.zeros((*target.shape, 3), dtype=np.uint8)
        view[target == 1] = [255, 100, 130]
        view[target == 255] = [255, 220, 0]
        overlay, _ = render_outputs(image, (target == 1).astype(np.uint8))
        rows.append((r["id"] + " | original / annotation (yellow=ignored) / overlay", [image, Image.fromarray(view), overlay]))
    if rows:
        contact_sheet(rows, output)
