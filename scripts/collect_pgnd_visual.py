"""Join saved geometry and visual curves, averaging episodes then categories."""

import argparse, json, csv
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--runs",
    required=True,
    help="JSON: task -> {geometry: directory, visual: directory}",
)
p.add_argument("--out", required=True)
a = p.parse_args()
jobs = json.loads(Path(a.runs).read_text())
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
categories = []
by_step = []
for task, paths in jobs.items():
    summary = json.loads((Path(paths["geometry"]) / "summary.json").read_text())
    if not summary["complete_split"] or summary["task"] != task:
        raise ValueError("Incomplete/wrong category")
    metrics = []
    for ep in summary["episodes"]:
        vd = Path(paths["visual"]) / f"episode_{ep:04d}"
        gd = Path(paths["geometry"]) / f"episode_{ep:04d}"
        visual = json.loads((vd / "frames.json").read_text())
        protocol = json.loads((vd / "protocol.json").read_text())
        from rrf.io import sha

        if protocol["trajectory_sha256"] != sha(gd / "trajectories_world.npz"):
            raise ValueError("Visual and geometry trajectories differ")
        if [v["prediction_step"] for v in visual] != list(range(1, 38)):
            raise ValueError("Missing/duplicate visual steps")
        with np.load(gd / "metrics.npz") as z:
            geo = z["official"]
        metrics.append(
            np.c_[
                geo,
                np.array(
                    [
                        [v[k] for k in ["j", "f", "psnr", "ssim", "lpips"]]
                        for v in visual
                    ]
                ),
            ]
        )
    mean = np.stack(metrics).mean(0)
    keys = [
        "mde_m",
        "cd_m",
        "emd_m",
        "iou",
        "boundary_f",
        "psnr_official",
        "ssim",
        "lpips",
    ]
    by_step.append(mean)
    index = summary["official_array_index"]
    categories.append(
        dict(
            category=task,
            physical_horizon=index + 1,
            **dict(zip(keys, mean[index].tolist())),
        )
    )
with (out / "per_category.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(categories[0]))
    w.writeheader()
    w.writerows(categories)
macro = np.stack(by_step).mean(0)
with (out / "temporal_macro.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["prediction_step", "time_s", *keys])
    w.writerows([[i + 1, (i + 1) * 0.1, *row] for i, row in enumerate(macro.tolist())])
(out / "coverage.json").write_text(
    json.dumps(
        {
            "categories": list(jobs),
            "complete_six_categories": set(jobs)
            == {"rope", "bread", "sloth", "box", "cloth", "paperbag"},
        },
        indent=2,
    )
)
