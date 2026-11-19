"""Freeze official PGND particle identities and world-space alignment once."""

from pathlib import Path
import sys
import numpy as np
import torch
from .io import read, write, sha, seed_all, atomic_torch
from .data.pgnd_prepare import frame_files, load_pt
from .data.pgnd_config import SPECS
from .metrics.external import extract_functions


def interval_actions(grips, dt=0.1):
    from kornia.geometry.conversions import (
        quaternion_to_rotation_matrix,
        rotation_matrix_to_axis_angle,
    )

    a = grips[2:-1].clone()
    future = grips[3:].clone()
    a[..., 3:6] = (future[..., :3] - a[..., :3]) / dt
    r0 = quaternion_to_rotation_matrix(a[..., 6:10].reshape(-1, 4))
    r1 = quaternion_to_rotation_matrix(future[..., 6:10].reshape(-1, 4))
    a[..., 10:13] = (
        rotation_matrix_to_axis_angle(r0.bmm(r1.transpose(1, 2))).reshape(
            len(a), a.shape[1], 3
        )
        / dt
    )
    return a, future


def reference(data, episode, root=None):
    from omegaconf import OmegaConf

    data = Path(data)
    paths = read(data / "paths.json")
    m = read(data / "manifest.json")
    task = m["task"]
    cfg = OmegaConf.create(read(data / "pgnd_config.json"))
    if episode not in [e["episode"] for e in m["eval"]]:
        raise ValueError("Episode not in held-out partition")
    dest = (
        Path(root) / task if root else data / "references"
    ) / f"episode_{episode:04d}"
    path = dest / "reference.pt"
    files = frame_files(Path(paths["native_state"]) / f"episode_{episode:04d}")[::3]
    ss = [load_pt(p) for p in files]
    xs = torch.stack([s["x"] for s in ss])
    grips = torch.stack([s["grippers"] for s in ss])
    if len(xs) != 40 or xs.dtype != torch.float32:
        raise ValueError("Expected 40 FP32 states")
    ns = {"torch": torch, "np": np}
    pgnd = Path(paths["pgnd_root"])
    extract_functions(pgnd / "experiments/train/metric.py", ["inverse_preprocess"], ns)
    with np.load(
        Path(paths["raw_source"]) / f"episode_{episode:04d}/traj.npz",
        allow_pickle=False,
    ) as z:
        raw = torch.from_numpy(z["xyz"].copy()).float()
    zero = torch.zeros(4, 3)
    zero[1:] = torch.eye(3)
    tr, full_world = ns["inverse_preprocess"](cfg, zero[None], raw.clone())
    shift = tr[0, 0]
    linear = tr[0, 1:] - shift
    if path.exists():
        ref = torch.load(path, map_location="cpu", weights_only=False)
        ids = ref["indices"]
        if ref["task"] != task or ref["episode"] != episode or ref["horizon"] != 37:
            raise ValueError("Frozen reference identity mismatch")
    else:
        sys.path.insert(0, str(pgnd))
        from pgnd.data import RealTeleopBatchDataset

        seed_all(list(SPECS).index(task) * 10000 + episode)
        ds = RealTeleopBatchDataset(
            cfg,
            Path(paths["native_state"]),
            Path(paths["raw_source"]),
            torch.device("cpu"),
            37,
            train=False,
            eval_episode_name=f"episode_{episode:04d}",
            dataset_non_overwrite=True,
        )
        if (
            len(ds.traj_list) != 1
            or Path(ds.traj_list[0][0]).parent.resolve()
            != (Path(paths["native_state"]) / f"episode_{episode:04d}").resolve()
        ):
            raise ValueError("Official Dataset episode alias mismatch")
        init, actions, gt, ids = ds[0]
        ids = ids.long().cpu()
        actions = actions.cpu()
        x, v, xh, vh, *_ = init
        torch.testing.assert_close(x.cpu(), xs[2, ids], rtol=0, atol=2e-5)
        torch.testing.assert_close(gt[0].cpu(), xs[3:, ids], rtol=0, atol=2e-5)
        expected, future = interval_actions(grips)
        torch.testing.assert_close(actions, expected, rtol=1e-5, atol=2e-5)
        ref = dict(
            task=task,
            episode=episode,
            sequence=torch.cat([xs[:3, ids], grips[:3, :, :3]], 1).permute(1, 0, 2)[
                None
            ],
            actions=actions,
            next_grippers=future,
            indices=ids,
            x0=xs[2, ids],
            gt_x=xs[3:, ids],
            world_gt=xs[3:, ids] @ linear + shift,
            world_x0=xs[2, ids] @ linear + shift,
            linear=linear,
            shift=shift,
            native_init=dict(x=x.cpu(), v=v.cpu(), x_his=xh.cpu(), v_his=vh.cpu()),
            dt=0.1,
            horizon=37,
            origin="Official named-episode Dataset FPS sampled once and frozen",
        )
        atomic_torch(ref, path)
        write(
            dest / "ready.json",
            dict(
                reference_sha256=sha(path),
                native_files={p.name: sha(p) for p in files},
                metric_source_sha256=sha(pgnd / "experiments/train/metric.py"),
            ),
        )
    if ids.shape != (SPECS[task][2],) or len(ids.unique()) != len(ids):
        raise ValueError("Invalid frozen particle IDs")
    for actual, expected in [
        (
            ref["sequence"],
            torch.cat([xs[:3, ids], grips[:3, :, :3]], 1).permute(1, 0, 2)[None],
        ),
        (ref["gt_x"], xs[3:, ids]),
        (ref["x0"], xs[2, ids]),
        (ref["next_grippers"], grips[3:]),
        (ref["world_gt"], full_world[9::3][:, ids]),
        (ref["linear"], linear),
        (ref["shift"], shift),
    ]:
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-6)
    from .models.regions import reference_pack

    ref = dict(
        ref,
        initial_ref=xs[0, ids],
        initial_ctrl_ref=grips[0, :, :3],
        patch=reference_pack(xs[0].numpy(), 64),
    )
    return ref, path
