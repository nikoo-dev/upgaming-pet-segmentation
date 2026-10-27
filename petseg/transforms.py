"""All resizing and label conventions live here."""
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ColorJitter, functional as TF
from torchvision.transforms.functional import InterpolationMode

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREPROCESSING = {"mean": MEAN, "std": STD, "resize": "square_bilinear", "mask_resize": "nearest", "ignore_index": 255}
CLASS_MAPPING = {"1": 1, "2": 0, "3": 255}


def map_trimap(mask):
    a = np.asarray(mask)
    if a.ndim != 2 or not np.isin(a, [1, 2, 3]).all():
        raise ValueError("Trimap must be 2D with raw labels 1=pet, 2=background, 3=boundary")
    out = np.full(a.shape, 255, dtype=np.uint8)
    out[a == 1] = 1
    out[a == 2] = 0
    return out


def orient_rgb(image):
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess(image, size):
    if size < 32:
        raise ValueError("Input resolution must be at least 32")
    image = TF.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


def paired_flip(image, mask):
    return TF.hflip(image), TF.hflip(mask)


class PairTransform:
    def __init__(self, size=256, augment=False):
        self.size, self.augment = size, augment
        self.jitter = ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __call__(self, image, raw_mask):
        # Map before resizing to validate every original annotation pixel.
        mask = Image.fromarray(map_trimap(raw_mask))
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image, mask = paired_flip(image, mask)
            image = self.jitter(image)
        mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        return preprocess(image, self.size), torch.from_numpy(np.array(mask, dtype=np.int64))
