"""Explicit additional-data camera and hand-mask adapters; no fabricated assets."""

import copy
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from ..io import read, sha


def array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def camera_from_raw(raw, ply, initial):
    raw = Path(raw)
    meta_path = raw / "metadata.json"
    cal_path = raw / "calibrate.pkl"
    meta = read(meta_path)
    with cal_path.open("rb") as f:
        c2ws = array(pickle.load(f)).astype(float)
    ks = np.array(meta["intrinsics"], float)
    if (
        c2ws.ndim != 3
        or c2ws.shape[1:] != (4, 4)
        or len(c2ws) < 2
        or ks.shape != (len(c2ws), 3, 3)
    ):
        raise ValueError("Expected calibrated [C,4,4] poses and [C,3,3] intrinsics")
    if not np.isfinite(c2ws).all() or not np.isfinite(ks).all():
        raise ValueError("Nonfinite calibration")
    if not np.allclose(c2ws[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid c2w homogeneous row")
    # Retain the previous official virtual-view convention: view0 pose, dummy_cam_id=1 intrinsics.
    c2w = c2ws[0]
    K = ks[1]
    if (
        not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-4)
        or np.linalg.det(c2w[:3, :3]) < 0
    ):
        raise ValueError("Camera rotation is not a proper orthogonal matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid focal lengths")
    with Image.open(raw / "color/0/0.png") as im:
        W, H = im.size
    if (W, H) != (848, 480) or list(meta["WH"]) != [W, H]:
        raise ValueError("Unexpected native image size")
    # cameras.json is used to cross-check the GS coordinate frame, not to guess a view by list index.
    cp = Path(ply).parents[2] / "cameras.json"
    cams = read(cp)
    if not isinstance(cams, list):
        raise ValueError("Expected Graphdeco cameras.json list")
    matches = []
    for i, c in enumerate(cams):
        if not all(k in c for k in ["rotation", "position"]):
            continue
        rot = np.asarray(c["rotation"], float)
        pos = np.asarray(c["position"], float)
        if rot.shape != (3, 3) or pos.shape != (3,):
            continue
        err = max(
            float(np.max(np.abs(rot - c2w[:3, :3]))),
            float(np.max(np.abs(pos - c2w[:3, 3]))),
        )
        if err < 1e-4:
            matches.append(dict(index=i, img_name=c.get("img_name"), pose_error=err))
    if not matches:
        raise ValueError(
            "Raw camera0 pose not found in GS cameras.json; no guessed transform"
        )
    pts = np.asarray(initial, float)[:: max(1, len(initial) // 2000)]
    cc = pts @ np.linalg.inv(c2w)[:3, :3].T + np.linalg.inv(c2w)[:3, 3]
    positive = cc[:, 2] > 1e-7
    uv = cc @ K.T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-7)
    inside = (
        positive & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    audit = dict(
        positive_fraction=float(positive.mean()),
        in_image_fraction=float(inside.mean()),
        gs_camera_matches=matches,
        camera0_pose_intrinsics1_policy=True,
        intrinsics_1_vs_0_max_abs=float(np.max(np.abs(ks[1] - ks[0]))),
    )
    if audit["positive_fraction"] < 0.95 or audit["in_image_fraction"] < 0.8:
        raise ValueError("Initial particles do not project into camera0; " + str(audit))
    return dict(
        K=K.tolist(),
        c2w=c2w.tolist(),
        width=W,
        height=H,
        source="raw_calibrate_camera0_metadata_intrinsics1",
        audit=audit,
    ), [meta_path, cal_path, cp]


def resolve(entry):
    a = copy.deepcopy(entry["assets"])
    raw = Path(a["raw"])
    frames = entry["raw_frames"]
    # Preserve all existing main-data paths and evaluation semantics exactly.
    if a.get("ready"):
        return a
    required = []
    issues = []
    sources = []
    if not a.get("ply") or not Path(a["ply"]).is_file():
        issues.append("Missing unique existing PLY")
    if not a.get("cfg") or not Path(a["cfg"]).is_file():
        issues.append("Missing GS cfg_args")
    masks = read(raw / "mask/mask_info_0.json")
    hands = sorted(int(k) for k, v in masks.items() if v == "hand")
    objects = [int(k) for k, v in masks.items() if v != "hand"]
    if len(objects) != 1:
        issues.append("Ambiguous object label")
    else:
        a["object_masks"] = str(raw / "mask/0" / str(objects[0]))
    human = Path(a["human"])
    if all((human / f"{t}.png").is_file() for t in frames):
        a["human_mode"] = "official_separate"
    elif hands:
        a["human_mode"] = "raw_hand_union"
        a["hand_dirs"] = [str(raw / "mask/0" / str(k)) for k in hands]
        sources.append(raw / "mask/mask_info_0.json")
    else:
        issues.append("No original human mask and no explicitly labelled hand instance")
    for t in frames:
        required.append(raw / f"color/0/{t}.png")
        if a.get("object_masks"):
            required.append(Path(a["object_masks"]) / f"{t}.png")
        required += (
            [Path(d) / f"{t}.png" for d in a.get("hand_dirs", [])]
            if a.get("human_mode") == "raw_hand_union"
            else [human / f"{t}.png"]
        )
    g = Path(a["gsdata"])
    if not all((g / f).is_file() for f in ["camera_meta.pkl", "interp_poses.pkl"]):
        try:
            with np.load(entry["cache"], allow_pickle=False) as z:
                pts = z["x"][0]
                valid = z["vis"][0] & np.isfinite(pts).all(-1)
                initial = pts[valid]
            a["camera"], ss = camera_from_raw(raw, a["ply"], initial)
            sources += ss
        except Exception as exc:
            issues.append("Camera fallback: " + str(exc))
    else:
        sources += [g / "camera_meta.pkl", g / "interp_poses.pkl"]
    issues += [str(p) for p in required if not p.is_file()]
    a["missing"] = issues
    a["ready"] = not issues
    a["identity_files"] = {str(p): sha(p) for p in sources if p.is_file()}
    for k in ["ply", "cfg"]:
        if a.get(k) and Path(a[k]).is_file():
            a["identity_files"][a[k]] = sha(a[k])
    a["mask_protocol"] = (
        "union of raw view0 hand instances"
        if a.get("human_mode") == "raw_hand_union"
        else "original separate human masks"
    )
    if a["ready"]:
        with Image.open(raw / "color/0/0.png") as im:
            W, H = im.size
        for d in a.get("hand_dirs", []):
            with Image.open(Path(d) / "0.png") as im:
                m = np.array(im)
            if m.shape != (H, W) or not np.isin(m, [0, 1, 255]).all():
                raise ValueError("Hand mask must be binary single-channel")
    return a


def hand_mask(asset, frame):
    if asset.get("human_mode") != "raw_hand_union":
        return np.array(Image.open(Path(asset["human"]) / f"{frame}.png"))
    acc = None
    for d in asset["hand_dirs"]:
        with Image.open(Path(d) / f"{frame}.png") as im:
            m = np.array(im)
        if m.ndim != 2 or not np.isin(m, [0, 1, 255]).all():
            raise ValueError("Non-binary hand mask")
        if acc is not None and acc.shape != m.shape:
            raise ValueError("Hand mask sizes disagree")
        acc = (m > 0) if acc is None else acc | (m > 0)
    if acc is None:
        raise ValueError("No hand mask sources")
    return acc.astype(np.uint8) * 255
