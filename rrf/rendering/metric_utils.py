import ast
import csv
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def cfg_read(p):
    node = ast.parse(Path(p).read_text().strip(), mode="eval").body
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "Namespace"
        or node.args
    ):
        raise ValueError("Expected cfg_args containing Namespace(keyword=literal, ...)")
    if any(k.arg is None for k in node.keywords):
        raise ValueError("Unsupported cfg_args expansion")
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}


def csv_write(p, rows):
    with Path(p).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows):
    keys = ["iou_pct", "psnr", "ssim", "lpips"]
    return {
        s: dict(frames=len(rr), **{k: sum(r[k] for r in rr) / len(rr) for k in keys})
        for s in ["E1", "E2"]
        if (rr := [r for r in rows if r["split"] == s])
    }
