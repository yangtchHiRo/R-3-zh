import numpy as np
import pickle, re, copy
from pathlib import Path
from scipy.spatial import cKDTree
from ..io import read, sha, digest, fps
from .phystwin import normalize_motion_mask, window_info


def array(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def bool_array(a, shape, name):
    a = array(a)
    if a.shape != shape or not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        raise ValueError(f"Invalid {name}: {a.shape}, expected {shape}")
    return a.astype(bool)


def case_id(root, path):
    rel = str(path.parent.relative_to(root))
    name = re.sub("[^A-Za-z0-9_.-]+", "__", rel)
    return (
        (name[-100:] if name not in ("", ".", "..") else path.parent.name)
        + "_"
        + digest(rel)[:8]
    )


def training_metadata(x, vis, motion, b, particles, min_particles):
    good = vis[:b] & np.isfinite(x[:b]).all(-1)
    v = np.diff(x[:b], axis=0)
    vm = good[1:] & good[:-1] & motion[: b - 1]
    acc = np.diff(v, axis=0)
    am = vm[1:] & vm[:-1]
    if vm.sum() < 3 or am.sum() < 3:
        raise ValueError("Too few valid training displacements/accelerations")
    initial = np.flatnonzero(good[0])
    if len(initial) < min_particles:
        raise ValueError("Too few visible first-frame object particles")
    ids = initial[fps(x[0, initial], min(particles, len(initial)))]
    p = x[0, ids]
    k = min(21, len(p))
    nn = cKDTree(p).query(p, k=k)[0][:, -1]
    radius = max(float(np.median(nn)), 1e-4)
    points = x[:b][good]
    lo = points.min(0) - 2 * radius
    hi = points.max(0) + 2 * radius
    return (
        dict(
            n_particles=len(ids),
            connectivity_radius=radius,
            target_degree=20,
            bounds=np.stack([lo, hi], axis=1).tolist(),
            vel_mean=v[vm].mean(0).tolist(),
            vel_std=np.maximum(v[vm].std(0), 1e-6).tolist(),
            acc_mean=acc[am].mean(0).tolist(),
            acc_std=np.maximum(acc[am].std(0), 1e-6).tolist(),
            normalization="training-only discrete displacement and second difference; world meters; original coordinate system",
        ),
        ids,
    )


def read_case(path, a):
    # Only unpickle the user's trusted official dataset, never arbitrary network files.
    with open(path, "rb") as f:
        d = pickle.load(f)
    x = array(d["object_points"]).astype(np.float32)
    ctrl = array(d["controller_points"]).astype(np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("object_points must be [T,N,3]")
    T, N, _ = x.shape
    if T < 9 or N < a.min_particles:
        raise ValueError("Insufficient frames/particles")
    if (
        ctrl.ndim != 3
        or ctrl.shape[0] != T
        or ctrl.shape[2] != 3
        or ctrl.shape[1] < 1
        or not np.isfinite(ctrl).all()
    ):
        raise ValueError(
            "controller_points must be finite [T,M,3]; no fake hand padding allowed"
        )
    vis = bool_array(d["object_visibilities"], (T, N), "object_visibilities")
    raw = array(d["object_motions_valid"])
    if not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError("Invalid motion validity values")
    motion = normalize_motion_mask(raw, T, N)
    split_path = path.parent / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"{split_path}; refusing to guess official split")
    split = read(split_path)
    b = int(split["train"][1])
    if (
        split["train"][0] != 0
        or split["test"] != [b, T]
        or int(split["frame_len"]) != T
        or not 6 <= b < T
    ):
        raise ValueError("Expected contiguous native [0,b),[b,T) split")
    metadata_path = path.parent / "metadata.json"
    rawmeta = read(metadata_path) if metadata_path.exists() else {}
    fps_value = a.fps
    fps_source = "explicit --fps" if a.fps else None
    if fps_value is None:
        for key in ("fps", "FPS", "frame_rate"):
            if key in rawmeta and np.isscalar(rawmeta[key]):
                fps_value = float(rawmeta[key])
                fps_source = "metadata." + key
                break
    if fps_value is None:
        fps_value = 30.0
        fps_source = "official default 30Hz; no rate field found"
    if not 0 < fps_value <= 1000:
        raise ValueError("Invalid FPS")
    gt_path = path.parent / "gt_track_3d.pkl"
    gt = np.empty((T, 0, 3), np.float32)
    gt_note = "missing independent manual track GT"
    if gt_path.is_file():
        with open(gt_path, "rb") as f:
            gt = array(pickle.load(f)).astype(np.float32)
        if gt.ndim != 3 or gt.shape[0] != T or gt.shape[-1] != 3:
            raise ValueError("gt_track_3d expected [T,K,3]")
        gt_note = "independent gt_track_3d.pkl; not dense training pseudo-track"
    meta, ids = training_metadata(x, vis, motion, b, a.particles, a.min_particles)
    horizon_windows = {}
    for horizon in (5, 10):
        wins = []
        for t in range(2, max(2, b - horizon)):
            _, good, label = window_info(x, vis, motion, t, b, horizon=horizon)
            if good.sum() >= a.min_particles and (label[:, good].sum(1) > 0).all():
                wins.append(t)
        horizon_windows[str(horizon)] = wins
    distances = cKDTree(x[0, ids]).query(ctrl[0])[0]
    source_files = {
        str(p): sha(p) for p in [path, split_path, metadata_path, gt_path] if p.exists()
    }
    e = dict(
        id=case_id(Path(a.root).resolve(), path),
        name=path.parent.name,
        source_dir=str(path.parent),
        source_files=source_files,
        T=T,
        N=N,
        M=ctrl.shape[1],
        train_end=b,
        dt=1 / fps_value,
        fps=fps_value,
        fps_source=fps_source,
        metadata=meta,
        initial_ids=ids.tolist(),
        training_windows=len(horizon_windows["5"]),
        training_windows_h10=len(horizon_windows["10"]),
        fit_ready=bool(horizon_windows["5"]),
        manual_gt_points=gt.shape[1],
        manual_gt_note=gt_note,
        manual_gt_finite_fraction=(
            float(np.isfinite(gt).all(-1).mean()) if gt.shape[1] else None
        ),
        controller_proxy="all persistent official controller points; radius0; identity quaternion; no contact labels",
        controller_initial_nearest_distance_m=distances.tolist(),
        initial_near_controls=int(
            (distances < 0.5 * meta["connectivity_radius"]).sum()
        ),
        object_initial_span_m=np.ptp(x[0, ids], axis=0).tolist(),
        initial_shape_extra_nodes={
            key: int(len(d[key])) if key in d else 0
            for key in ("surface_points", "interior_points")
        },
        warning="Native preprocessed/offline tracks; initial zero controls edges can be a valid approaching-hand state.",
    )
    return e, dict(x=x, vis=vis, motion=motion, ctrl=ctrl, gt=gt)


def pooled_metadata(items, particles):
    # Count-weighted population moments, computed ONLY from each source fit interval.
    local = []
    stats = {k: [0, np.zeros(3), np.zeros(3)] for k in ["vel", "acc"]}
    for ar, b in items:
        m, _ = training_metadata(ar["x"], ar["vis"], ar["motion"], b, particles, 32)
        local.append(m)
        x = ar["x"][:b]
        good = ar["vis"][:b] & np.isfinite(x).all(-1)
        v = np.diff(x, axis=0)
        ok = good[1:] & good[:-1] & ar["motion"][: b - 1]
        for k, z in [("vel", v[ok]), ("acc", np.diff(v, axis=0)[ok[1:] & ok[:-1]])]:
            n, s, q = stats[k]
            z = z.astype(np.float64)
            stats[k] = [n + len(z), s + z.sum(0), q + (z * z).sum(0)]
    meta = copy.deepcopy(local[0])
    meta["n_particles"] = particles
    for k, (n, s, q) in stats.items():
        if n < 3:
            raise ValueError("Insufficient training statistics")
        meta[k + "_mean"] = (s / n).tolist()
        meta[k + "_std"] = np.maximum(
            np.sqrt(np.maximum(q / n - (s / n) ** 2, 0)), 1e-6
        ).tolist()
    bounds = np.array([m["bounds"] for m in local])
    meta["bounds"] = np.stack(
        [bounds[:, :, 0].min(0), bounds[:, :, 1].max(0)], 1
    ).tolist()
    meta["connectivity_radius"] = float(
        np.median([m["connectivity_radius"] for m in local])
    )
    meta["normalization"] = (
        "pooled training scenes/fit intervals only; discrete displacement; world meters"
    )
    return meta
