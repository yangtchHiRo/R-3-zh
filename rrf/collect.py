"""Collect per-scene/per-pair metrics; macro averages never replace pair rows."""

import argparse
from pathlib import Path
import numpy as np
from .io import read, write
from .evaluate import csv_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for filename in a.inputs:
        r = read(filename)
        if r.get("dataset") == "pgnd":
            if not r["complete_split"]:
                raise ValueError("Incomplete episode partition")
            rows.append({k: r[k] for k in ["task", "step", "mde_m", "cd_m", "emd_m"]})
            continue
        for s in r.get("scenes", [r]):
            for split in ("E1", "E2", "ALL"):
                if split not in s:
                    continue
                q = s[split]
                visual = {k: None for k in ("iou_pct", "psnr", "ssim", "lpips")}
                image_path = Path(filename).parent / s["scene"] / "images/metrics.json"
                if image_path.exists():
                    im = read(image_path)
                    if im.get("status") != "complete" or im.get("smoke_only"):
                        raise ValueError("Incomplete image metrics")
                    if im["request"]["checkpoint_sha256"] != s["checkpoint_sha256"]:
                        raise ValueError("Image/geometry checkpoints differ")
                    values = im["metrics"]
                    if split in values:
                        visual = {k: values[split][k] for k in visual}
                    elif split == "ALL":
                        total = sum(v["frames"] for v in values.values())
                        visual = {
                            k: sum(v[k] * v["frames"] for v in values.values()) / total
                            for k in visual
                        }
                rows.append(
                    dict(
                        scene=s["scene"],
                        source_scene=r.get("source_scene"),
                        target_scene=r.get("target_scene", s["scene"]),
                        source_checkpoint=s.get("checkpoint", r.get("checkpoint")),
                        step=s.get("step", r.get("step")),
                        split=split,
                        frames=q["frames"],
                        prediction_complete=q["prediction_complete"],
                        cd_l1_m=q["cd_l1_m"],
                        track_l2_m=q["track_l2_m"],
                        pseudo_mde_m=q.get("pseudo_mde_m"),
                        **visual,
                    )
                )
    if not rows:
        raise ValueError("No results")
    if len({tuple(r) for r in rows}) != 1:
        raise ValueError(
            "Collect PhysTwin and PGND into separate tables because their metric definitions differ"
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / "per_scene.csv", rows)
    write(out / "per_scene.json", rows)
    if "task" in rows[0]:
        if len({r["task"] for r in rows}) != len(rows):
            raise ValueError(
                "Duplicate PGND category (provide one evaluated checkpoint per category)"
            )
        mean = {
            k: float(np.mean([r[k] for r in rows])) for k in ("mde_m", "cd_m", "emd_m")
        }
        write(
            out / "macro_mean.json", dict(categories=[r["task"] for r in rows], **mean)
        )
    else:
        means = {}
        for split in ("E1", "E2", "ALL"):
            rr = [r for r in rows if r["split"] == split]
            if not rr:
                continue
            if any(r["frames"] and not r["prediction_complete"] for r in rr):
                raise ValueError("Incomplete rollouts; refusing a truncated mean")
            keys = (
                "cd_l1_m",
                "track_l2_m",
                "pseudo_mde_m",
                "iou_pct",
                "psnr",
                "ssim",
                "lpips",
            )
            means[split] = {
                k: (
                    float(np.mean(v))
                    if (v := [r[k] for r in rr if r[k] is not None])
                    else None
                )
                for k in keys
            }
            means[split]["valid_scenes"] = {
                k: sum(r[k] is not None for r in rr) for k in keys
            }
        write(out / "macro_mean.json", means)
    print(out / "per_scene.csv")


if __name__ == "__main__":
    main()
