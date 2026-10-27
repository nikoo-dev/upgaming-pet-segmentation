"""Official data preparation, content-grouped splits, and explicit audit records."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

from .transforms import PairTransform, map_trimap
from .utils import read_json, sha256, write_json


def annotation_root(root):
    return Path(root) / "oxford-iiit-pet"


def read_list(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        image_id, breed, species, species_breed = line.split()
        records.append({"id": image_id, "breed": int(breed), "species": int(species)})
    return records


def load_pair(base, record):
    with Image.open(Path(base) / "images" / (record["id"] + ".jpg")) as f:
        orientation = f.getexif().get(274, 1)
        image = f.convert("RGB")
    with Image.open(Path(base) / "annotations/trimaps" / (record["id"] + ".png")) as f:
        mask = f.copy()
    # Annotation coordinates refer to stored pixels. Apply EXIF to both together.
    method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
              4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
              6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
              8: Image.Transpose.ROTATE_90}.get(orientation)
    if method is not None:
        image, mask = image.transpose(method), mask.transpose(method)
    # Avoid a second EXIF transpose later in shared inference.
    image.info.pop("exif", None)
    if image.size != mask.size:
        raise ValueError("Image/mask sizes differ")
    return image, mask


def grouped_split(records, test_records, seed=42):
    """Approximately 80/20 within breed; never split identical decoded RGB content."""
    test_hashes = {r["content_sha256"] for r in test_records}
    excluded, by_breed = [], defaultdict(lambda: defaultdict(list))
    hash_breeds = defaultdict(set)
    for r in records:
        hash_breeds[r["content_sha256"]].add(r["breed"])
    for r in records:
        reason = None
        if r["content_sha256"] in test_hashes:
            reason = "duplicate_of_official_test"
        elif len(hash_breeds[r["content_sha256"]]) > 1:
            reason = "conflicting_breed_for_identical_content"
        if reason:
            excluded.append({"id": r["id"], "reason": reason})
        else:
            by_breed[r["breed"]][r["content_sha256"]].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for breed in sorted(by_breed):
        groups = list(by_breed[breed].values())
        groups.sort(key=lambda g: g[0]["id"])
        rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError(f"Breed {breed} cannot be split into train/validation content groups")
        # Choose a prefix closest to 20%, leaving at least one group for training.
        total = sum(len(g) for g in groups)
        cumulative = np.cumsum([len(g) for g in groups[:-1]])
        count = min(range(1, len(groups)), key=lambda k: (abs(cumulative[k-1] - total * .2), k))
        val.extend(r for g in groups[:count] for r in g)
        train.extend(r for g in groups[count:] for r in g)
    splits = {"train": sorted(train, key=lambda r: r["id"]),
              "val": sorted(val, key=lambda r: r["id"]),
              "test": sorted(test_records, key=lambda r: r["id"])}
    validate_splits(splits)
    return splits, excluded


def validate_splits(splits):
    for name, rows in splits.items():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate ID within {name}")
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if {r["id"] for r in splits[a]} & {r["id"] for r in splits[b]}:
                raise ValueError(f"ID leakage between {a} and {b}")
            if {r["content_sha256"] for r in splits[a]} & {r["content_sha256"] for r in splits[b]}:
                raise ValueError(f"Content leakage between {a} and {b}")


def stratified_cap(records, cap, seed=42):
    if cap is None:
        return list(records)
    if cap < 1:
        raise ValueError("Training cap must be positive")
    if cap >= len(records):
        return list(records)
    groups = defaultdict(list)
    for r in records:
        groups[r["breed"]].append(r)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    result = []
    # Round-robin strata provide deterministic broad breed coverage.
    while len(result) < cap:
        for breed in sorted(groups):
            if groups[breed] and len(result) < cap:
                result.append(groups[breed].pop())
    return result


def prepare(root, output, download=False):
    if download:
        OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=True)
    base, output = annotation_root(root), Path(output)
    readme = base / "annotations/README"
    if not readme.exists():
        raise FileNotFoundError("Official annotations/README missing; use prepare --download")
    text = readme.read_text(encoding="utf-8", errors="replace")
    # Keep the actual primary-source annotation description for manual inspection.
    output.mkdir(parents=True, exist_ok=True)
    (output / "ANNOTATION_README.txt").write_text(text, encoding="utf-8")
    if not all(word in text.lower() for word in ["trimap", "foreground", "background"]):
        raise ValueError("Unexpected annotation README: inspect label conventions before training")
    valid, exclusions, pixels, dimensions = {}, [], Counter(), []
    for partition in ["trainval", "test"]:
        valid[partition] = []
        for r in read_list(base / "annotations" / (partition + ".txt")):
            try:
                image, raw = load_pair(base, r)
                mapped = map_trimap(raw)
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                r = {**r, "content_sha256": digest, "width": image.width, "height": image.height}
                dimensions.append({"id": r["id"], "width": image.width, "height": image.height,
                                   "aspect_ratio": image.width / image.height})
                values, counts = np.unique(mapped, return_counts=True)
                pixels.update({int(v): int(c) for v, c in zip(values, counts)})
                valid[partition].append(r)
            except (OSError, ValueError) as exc:
                exclusions.append({"id": r["id"], "partition": partition, "reason": str(exc)})
        print(f"Audited {partition}: {len(valid[partition])} usable", flush=True)
    write_json(output / "exclusions.json", exclusions)
    # Never silently shrink the final official test set.
    if any(r["partition"] == "test" for r in exclusions):
        raise ValueError("Official test has corrupt/missing pairs; repair data before creating splits. See exclusions.json")
    splits, duplicates = grouped_split(valid["trainval"], valid["test"])
    exclusions.extend(duplicates)
    write_json(output / "exclusions.json", exclusions)
    manifest = {"schema_version": 1, "seed": 42, "splits": splits,
                "source": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
                "annotation_readme_sha256": sha256(readme)}
    write_json(output / "splits.json", manifest)
    total = sum(pixels.values())
    duplicates_within_test = len(splits["test"]) - len({r["content_sha256"] for r in splits["test"]})
    audit = {"counts": {k: len(v) for k, v in splits.items()}, "dimensions": dimensions,
             "pixel_proportions_before_duplicate_exclusion": {str(k): v / total for k, v in pixels.items()},
             "breed_distribution": {k: dict(Counter(r["breed"] for r in v)) for k, v in splits.items()},
             "species_distribution": {k: dict(Counter(r["species"] for r in v)) for k, v in splits.items()},
             "exclusions": exclusions, "duplicate_test_images_retained": duplicates_within_test,
             "manifest_sha256": sha256(output / "splits.json")}
    write_json(output / "audit.json", audit)
    from .visualize import audit_examples
    audit_records = random.Random(42).sample(splits["train"], min(6, len(splits["train"])))
    audit_examples(base, audit_records, output / "audit_examples.png")
    return audit


class PetDataset(Dataset):
    def __init__(self, root, manifest, split="train", size=256, augment=False, cap=None, seed=42):
        self.base = annotation_root(root)
        m = read_json(manifest)
        validate_splits(m["splits"])
        self.records = m["splits"][split]
        if split == "train":
            self.records = stratified_cap(self.records, cap, seed)
        self.transform = PairTransform(size, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = load_pair(self.base, self.records[index])
        return self.transform(image, mask)
