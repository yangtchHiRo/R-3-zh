from __future__ import annotations
import hashlib, json, os, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, p)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def code_hashes():
    paths = list(ROOT.rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fps(x, n):
    x = np.asarray(x)
    n = min(int(n), len(x))
    if n < 1 or not np.isfinite(x).all():
        raise ValueError("Invalid FPS input")
    ids = np.empty(n, dtype=np.int64)
    ids[0] = np.square(x - x.mean(0)).sum(-1).argmax()
    d = np.full(len(x), np.inf)
    for k in range(1, n):
        d = np.minimum(d, np.square(x - x[ids[k - 1]]).sum(-1))
        d[ids[:k]] = -1
        ids[k] = d.argmax()
    return ids


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def atomic_torch(value, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_torch(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def finite_mean(x):
    a = np.asarray([v for v in x if v is not None], dtype=float)
    return float(a.mean()) if len(a) else None
