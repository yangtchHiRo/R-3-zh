"""Evaluate frozen checkpoints; geometry never consumes future object states."""

import argparse, csv
from pathlib import Path
import numpy as np
import torch
from .io import read, write, sha, atomic_npz, atomic_torch
from .checkpoints import load, add_arguments, from_args


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def phystwin(a):
    from .data.phystwin import Case
    from .dynamics import rollout
    from .metrics.phystwin import curves, aggregate
    from .rendering.assets import assets

    p = read(Path(a.data) / "manifest.json")
    g = p["group"]
    if a.scenes:
        names = a.scenes
    elif a.partition == "all":
        names = [e["id"] for e in p["cases"]]
    else:
        names = g[a.partition]
    if not names:
        raise ValueError("Empty partition; use --scenes for source temporal validation")
    model, ck, checkpoint = load(a.checkpoint, a.device)
    results = []
    for name in names:
        c = Case(p, name)
        out = Path(a.out) / name
        out.mkdir(parents=True, exist_ok=True)
        from .models.regions import reference_pack

        good = c.vis[0] & np.isfinite(c.x[0]).all(-1)
        c.patch_reference = reference_pack(
            c.x[0, good], ck["model_config"]["settings"]["patches"]
        )
        b = c.entry.get("official_train_end", c.b)
        request = dict(
            checkpoint_sha256=sha(checkpoint), cache_sha256=c.entry["cache_sha256"]
        )
        if (out / "request.json").exists() and read(out / "request.json") != request:
            raise ValueError("Evaluation output belongs to different inputs")
        write(out / "request.json", request)
        pred, failure = rollout(model, c, c.eval_ids, a.device)
        rows, matching = curves(pred, c, c.eval_ids)
        summary = dict(
            scene=name,
            step=ck.get("step", 0),
            checkpoint=str(checkpoint.resolve()),
            failure=failure,
            E1=aggregate(rows, 3, b),
            E2=aggregate(rows, b, c.T),
            ALL=aggregate(rows, 3, c.T),
            **request,
        )
        atomic_npz(
            out / "geometry/trajectory.npz",
            pred=pred,
            initial_ids=c.eval_ids,
            controls=c.ctrl,
            dt=c.entry["dt"],
            train_end=b,
            raw_frame_indices=np.arange(c.T),
        )
        csv_write(out / "geometry/per_frame.csv", rows)
        write(out / "geometry/matching.json", matching)
        asset = (
            assets(Path(p["dataset_root"]), g["source"], name, range(c.T))
            if a.prepare_render
            else dict(
                ready=False,
                missing=[
                    "Rerun with --prepare-render to resolve external appearance assets"
                ],
            )
        )
        if a.prepare_render and g["source"] == "additional" and not asset["ready"]:
            from .rendering.asset_fix import resolve

            asset = resolve(
                dict(assets=asset, cache=c.entry["cache"], raw_frames=list(range(c.T)))
            )
        entry = dict(
            case_name=name,
            case=name,
            step=ck.get("step", 0),
            checkpoint=str(checkpoint.resolve()),
            assets=asset,
            checkpoint_sha256=request["checkpoint_sha256"],
            score_start=3,
        )
        write(out / "render_input.json", entry)
        write(
            out / "geometry/done.json",
            dict(
                evaluation=entry, trajectory_sha256=sha(out / "geometry/trajectory.npz")
            ),
        )
        write(out / "summary.json", summary)
        results.append(summary)
        print(
            "[EVAL]",
            name,
            "E1",
            summary["E1"]["cd_l1_m"],
            "E2",
            summary["E2"]["cd_l1_m"],
            flush=True,
        )
    write(
        Path(a.out) / "summary.json",
        dict(
            dataset="phystwin",
            group=g["name"],
            checkpoint=str(checkpoint.resolve()),
            step=ck.get("step", 0),
            scenes=results,
        ),
    )


@torch.no_grad()
def particle_rollout(model, ref, device):
    from .dynamics import predict_sequence

    return predict_sequence(
        model,
        ref["sequence"],
        ref["actions"],
        ref["next_grippers"],
        ref["initial_ref"],
        ref["initial_ctrl_ref"],
        ref["patch"],
        device,
        ref.get("reference_valid"),
        ref.get("ctrl_available"),
    )


def pgnd(a):
    from omegaconf import OmegaConf
    from .references import reference
    from .metrics.pgnd import run_official_metrics, official_aggregate

    data = Path(a.data)
    paths = read(data / "paths.json")
    manifest = read(data / "manifest.json")
    model, ck, checkpoint = load(a.checkpoint, a.device)
    eps = [e["episode"] for e in manifest["eval"]]
    if a.episodes:
        if not set(a.episodes) <= set(eps):
            raise ValueError("Requested episode is not held out")
        eps = a.episodes
    cfg = OmegaConf.create(read(data / "pgnd_config.json"))
    OmegaConf.set_struct(cfg, False)
    cfg.iteration = int(ck.get("step", 0))
    cfg.train.source_dataset_name = paths["raw_source"]
    cfg.train.dataset_name = str(Path(paths["native_state"]).parent)
    all_metrics = []
    mses = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep in eps:
        ref, rpath = reference(data, ep, a.references)
        dest = out / f"episode_{ep:04d}"
        dest.mkdir(parents=True, exist_ok=True)
        request = dict(checkpoint_sha256=sha(checkpoint), reference_sha256=sha(rpath))
        if (dest / "request.json").exists() and read(dest / "request.json") != request:
            raise ValueError("Evaluation output belongs to different inputs")
        write(dest / "request.json", request)
        pred = particle_rollout(model, ref, a.device)
        official = run_official_metrics(
            paths["pgnd_root"],
            cfg,
            ep,
            ref["indices"],
            ref["x0"],
            pred,
            dest,
            "rrf",
            a.max_cost_gib,
        )
        world = pred @ ref["linear"] + ref["shift"]
        err = world.double() - ref["world_gt"].double()
        mde = err.norm(dim=-1).mean(1).numpy()
        mse = err.square().mean((1, 2)).numpy()
        np.testing.assert_allclose(mde, official[:, 0], rtol=0, atol=3e-6)
        atomic_torch(dict(pred=pred, **request), dest / "prediction.pt")
        atomic_npz(
            dest / "trajectories_world.npz",
            pred=world.numpy(),
            gt=ref["world_gt"].numpy(),
            x0=ref["world_x0"].numpy(),
            particle_ids=ref["indices"].numpy(),
            dt=0.1,
        )
        atomic_npz(dest / "metrics.npz", official=official, mse_xyz_m2=mse)
        write(
            dest / "done.json",
            dict(status="complete", episode=ep, step=ck.get("step", 0), **request),
        )
        all_metrics.append(official)
        mses.append(mse)
        print("[EVAL]", manifest["task"], ep, flush=True)
    values = np.stack(all_metrics)
    mean, std, index, program = official_aggregate(values, paths["pgnd_root"])
    summary = dict(
        dataset="pgnd",
        task=manifest["task"],
        step=ck.get("step", 0),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha(checkpoint),
        episodes=eps,
        complete_split=eps == [e["episode"] for e in manifest["eval"]],
        official_array_index=index,
        physical_horizon=index + 1,
        mde_m=float(mean[0]),
        cd_m=float(mean[1]),
        emd_m=float(mean[2]),
        std=std.tolist(),
        official_program=program,
    )
    write(out / "summary.json", summary)
    csv_write(
        out / "temporal_metrics.csv",
        [
            dict(
                prediction_step=k + 1,
                time_s=(k + 1) * 0.1,
                mde_m=float(values[:, k, 0].mean()),
                cd_m=float(values[:, k, 1].mean()),
                emd_m=float(values[:, k, 2].mean()),
                mse_xyz_m2=float(np.stack(mses)[:, k].mean()),
            )
            for k in range(37)
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["phystwin", "pgnd"], required=True)
    p.add_argument("--data", required=True)
    add_arguments(p)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument(
        "--partition", choices=["train", "validation", "test", "all"], default="all"
    )
    p.add_argument("--scenes", nargs="+")
    p.add_argument("--prepare-render", action="store_true")
    p.add_argument("--references", help="Fixed PGND particle reference directory")
    p.add_argument("--episodes", nargs="+", type=int)
    p.add_argument("--max-cost-gib", type=float, default=4.0)
    a = p.parse_args()
    a.checkpoint = from_args(a)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    globals()[a.dataset](a)


if __name__ == "__main__":
    main()
