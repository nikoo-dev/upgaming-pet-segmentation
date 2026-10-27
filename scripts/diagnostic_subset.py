"""Extract a tiny *diagnostic only* trainval subset from a downloaded image archive.

Supports an incomplete streaming archive: stop after collecting ten complete images.
This is never a replacement for the checksum-verified full prepare workflow.
Annotations must already have been downloaded and checksum-verified by Torchvision.
"""
import argparse
import hashlib
from pathlib import Path
import tarfile

from PIL import Image

from petseg.data import load_pair, read_list
from petseg.transforms import map_trimap
from petseg.utils import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-archive", default="data/oxford-iiit-pet/images.tar.gz")
    p.add_argument("--root", default="artifacts/diagnostic_data")
    p.add_argument("--output", default="artifacts/diagnostics/data")
    args = p.parse_args()
    base = Path(args.root) / "oxford-iiit-pet"
    eligible = {r["id"]: r for r in read_list(base / "annotations/trainval.txt")}
    (base / "images").mkdir(parents=True, exist_ok=True)
    rows, used_hashes = [], set()
    with tarfile.open(args.image_archive, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).stem
            if not member.isfile() or not member.name.endswith(".jpg") or name not in eligible:
                continue
            stream = archive.extractfile(member)
            content = stream.read()
            path = base / "images" / (name + ".jpg")
            path.write_bytes(content)
            image, raw = load_pair(base, eligible[name])
            map_trimap(raw)
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest in used_hashes:
                continue
            used_hashes.add(digest)
            rows.append({**eligible[name], "content_sha256": digest})
            if len(rows) == 10:
                break
    if len(rows) < 10:
        raise RuntimeError("Not enough complete trainval images available yet; wait for more download progress")
    output = Path(args.output)
    write_json(output / "splits.json", {"schema_version": 1, "seed": 42, "diagnostic_only": True,
               "splits": {"train": rows[2:], "val": rows[:2], "test": []},
               "provenance": "First ten distinct complete official-trainval members in image archive order. No final test used. Image archive may be incomplete: its full checksum is NOT verified. Not a final stratified split."})
    from petseg.visualize import audit_examples
    audit_examples(base, rows[:6], output / "examples.png")
    print(output / "splits.json")


if __name__ == "__main__":
    main()
