import random, inspect, importlib, sys, tempfile
from pathlib import Path
import numpy as np
from ..io import read, write, sha, digest


def load_pt(p):
    import torch

    return torch.load(p, map_location="cpu", weights_only=False)


def official_preprocessor(pgnd):
    pgnd = Path(pgnd).resolve()
    sys.path.insert(0, str(pgnd))
    data = importlib.import_module("pgnd.data")
    fn = data.RealTeleopBatchDataset.__init__.__globals__.get("preprocess")
    if not callable(fn):
        raise RuntimeError(
            "Local PGND Dataset does not expose its preprocessing function"
        )
    path = Path(inspect.getfile(fn)).resolve()
    if not path.is_relative_to(pgnd):
        raise RuntimeError("Imported PGND outside selected root: " + str(path))
    return fn, {"file": str(path), "sha256": sha(path)}


def raw_signature(ep, g):
    with np.load(ep / "traj.npz", allow_pickle=False) as z:
        x = z["xyz"]
        shape = x.shape
        if x.ndim != 3 or x.shape[2] != 3 or x.shape[0] < 2 or not np.isfinite(x).all():
            raise ValueError("Invalid xyz trajectory: " + str(ep))
    eef = np.loadtxt(ep / "eef_traj.txt", ndmin=2)
    if eef.shape != (shape[0], g * 3) or not np.isfinite(eef).all():
        raise ValueError(f"{ep}: eef_traj shape {eef.shape}, expected {(shape[0],g*3)}")
    files = ["traj.npz", "eef_traj.txt"] + [
        s for s in ("eef_rot.txt", "eef_gripper.txt") if (ep / s).exists()
    ]
    return {"raw_shape": list(shape), "files": {s: sha(ep / s) for s in files}}


def frame_files(ep):
    files = sorted(ep.glob("*.pt"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(len(files))):
        raise ValueError("Missing/duplicate raw-frame cache files: " + str(ep))
    return files


def verify_states(ep, n_frames, n_min, g):
    import torch

    files = frame_files(ep)
    if len(files) != n_frames or not (ep / "info.yaml").is_file():
        raise ValueError("Incomplete native state: " + str(ep))
    hashes = {}
    particle_count = None
    for p in files:
        s = load_pt(p)
        x, grip = s["x"], s["grippers"]
        if x.dtype != torch.float32 or grip.dtype != torch.float32:
            raise ValueError("Native cache must remain FP32")
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < n_min:
            raise ValueError("Insufficient/invalid native particles: " + str(p))
        if (
            grip.shape != (g, 15)
            or not torch.isfinite(x).all()
            or not torch.isfinite(grip).all()
        ):
            raise ValueError("Invalid native state: " + str(p))
        if particle_count is not None and len(x) != particle_count:
            raise ValueError("Particle identity/count changes across frames")
        particle_count = len(x)
        hashes[p.name] = sha(p)
    hashes["info.yaml"] = sha(ep / "info.yaml")
    return {"native_particles": particle_count, "files_sha256": hashes}


def ensure_native(ep, dst, cfg, fn, fn_info, n, g):
    from omegaconf import OmegaConf

    raw = raw_signature(ep, g)
    contract = digest({"config": cfg, "preprocessor": fn_info, "raw": raw})
    if dst.exists():
        if not (dst / "preprocess_done.json").is_file():
            raise FileExistsError(
                "Unmanaged or incomplete native directory; not overwriting: " + str(dst)
            )
        done = read(dst / "preprocess_done.json")
        if done["contract"] != contract:
            raise ValueError(
                "Source/config changed; use another output root: " + str(dst)
            )
        if set(done["files_sha256"]) != {p.name for p in frame_files(dst)} | {
            "info.yaml"
        }:
            raise ValueError("Native frame list changed: " + str(dst))
        for name, h in done["files_sha256"].items():
            if sha(dst / name) != h:
                raise ValueError("Native output changed: " + str(dst / name))
        return done
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build_", dir=dst.parent) as tmp:
        work = Path(tmp) / dst.name
        work.mkdir()
        c = OmegaConf.create(cfg)
        dt_raw = (
            float(c.sim.dt)
            * int(c.sim.interval)
            / int(c.train.dataset_skip_frame)
            / int(c.train.dataset_load_skip_frame)
        )
        ok = fn(c, work, ep, dt_raw, c.sim.num_grids[-1])
        if ok is False:
            raise RuntimeError("Official preprocessing failed: " + str(ep))
        result = verify_states(work, raw["raw_shape"][0], n, g)
        result.update(
            contract=contract, raw=raw, source=str(ep), raw_frames=raw["raw_shape"][0]
        )
        write(work / "preprocess_done.json", result)
        work.rename(dst)
    return result


def fixed_fps(x, n, seed, device):
    """Deterministic calibration subset, not the random Full-5 training sample."""
    import torch

    if len(x) < n:
        raise ValueError(f"{len(x)} points < configured {n}")
    if len(x) == n:
        return torch.arange(n)
    start = random.Random(seed).randint(0, len(x) - 1)
    x = x.to(device)
    ids = torch.empty(n, dtype=torch.long, device=device)
    ids[0] = start
    dist = torch.full((len(x),), float("inf"), device=device)
    for j in range(1, n):
        dist = torch.minimum(dist, (x - x[ids[j - 1]]).square().sum(-1))
        dist[ids[:j]] = -1  # enforce unique IDs even if points overlap
        ids[j] = dist.argmax()
    return ids.cpu()


def compute_statistics(train_x, task, cfg, policy="official", train_ids=None):
    """Exact Training calibration rule; caller must provide TRAIN tensors only."""
    import torch
    from .pgnd_config import split_ids

    expected = train_ids if train_ids is not None else split_ids(task, policy)[0]
    n = int(cfg["sim"]["n_particles"])
    vs = []
    acc = []
    radii = []
    seen = []
    for eid, x in train_x:
        if eid not in expected:
            raise ValueError("Excluded/TEST episode passed to normalization")
        seen.append(eid)
        v = x[1:] - x[:-1]
        if len(v) >= 2:
            a = v[1:] - v[:-1]
            sub = max(1, n // 200)
            vs.append(v[:, ::sub].reshape(-1, 3))
            acc.append(a[:, ::sub].reshape(-1, 3))
        if len(radii) < 24:
            ref = min(2, len(x) - 1)
            for f in (ref, min(ref + 5, len(x) - 1)):
                radii.append(
                    torch.cdist(x[f], x[f]).topk(21, largest=False).values[:, -1]
                )
    if seen != expected:
        raise ValueError("Incomplete TRAIN-only calibration")
    v = torch.cat(vs)
    a = torch.cat(acc)
    radius = float(torch.cat(radii).median())
    if not radius > 0:
        raise ValueError("Nonpositive connectivity radius")
    meta = dict(
        task=task,
        n_particles=n,
        num_grippers=int(cfg["sim"]["num_grippers"]),
        dataset_skip_frame=1,
        load_skip_frame=3,
        connectivity_radius=radius,
        target_degree=20,
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        vel_mean=v.mean(0).tolist(),
        vel_std=v.std(0).clamp_min(1e-6).tolist(),
        acc_mean=a.mean(0).tolist(),
        acc_std=a.std(0).clamp_min(1e-6).tolist(),
        primary_history_len=3,
        recommended_noise_std_0p1_vel=float(0.1 * v.std(0).clamp_min(1e-6).mean()),
        note="Training TRAIN-only FP32 calibration; compact FPS cache is FP16. Full-5 resamples native FP32 particles.",
        normalization_train_ids=seen,
        radius_frame_cap=24,
    )
    if policy != "official":
        meta["split_policy"] = policy
    return meta
