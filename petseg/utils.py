"""Small, explicit helpers for reproducibility and artifact provenance."""
import hashlib
import json
import os
import platform
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torchvision


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Bytes avoid Windows text-mode CRLF translation: manifests must hash the
    # same way when prepared independently on Windows and Colab/Linux.
    tmp.write_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(generator=None):
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state() if generator is not None else None}


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if generator is not None and state["loader"] is not None:
        generator.set_state(state["loader"].cpu())


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_hash = sidecar.with_name(sidecar.name + ".tmp")
    temporary_hash.write_text(sha256(path) + "\n", encoding="ascii")
    os.replace(temporary_hash, sidecar)


def load_checkpoint(path):
    """Check the saved file identity before loading a trusted project checkpoint."""
    path = Path(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    if expected != sha256(path):
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, KeyError, TypeError, pickle.UnpicklingError, EOFError, OSError) as exc:
        raise ValueError(f"Invalid training checkpoint: {path}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Invalid training checkpoint: {path}")
    return state


def tensor_state_sha256(state):
    """Hash tensor content independently of torch.save's serialization details."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(header.encode("utf-8") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
