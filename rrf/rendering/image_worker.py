"""Stream saved GNS object motion through the installed PhysTwin LBS and renderer."""

import argparse
import ast
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from .metric_utils import read, write, sha, cfg_read, csv_write, aggregate
from .asset_fix import hand_mask


def main(a):
    official = Path(a.official).resolve()
    dataset = Path(a.dataset).resolve()
    evaluation = Path(a.evaluation).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(official))
    import numpy as np
    import torch
    from PIL import Image
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.dynamic_utils import (
        get_topk_indices,
        knn_weights,
        interpolate_motions,
    )
    from gaussian_splatting.utils.loss_utils import ssim
    from gaussian_splatting.utils.image_utils import psnr
    from gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(0)
    np.random.seed(0)

    # Upstream debugging branches must fail visibly instead of opening an interactive shell.
    def no_debugger(*args, **kwargs):
        raise RuntimeError(
            "Official LBS entered its numerical-error debugger branch; inspect worker log"
        )

    import pdb

    pdb.set_trace = no_debugger
    try:
        import ipdb

        ipdb.set_trace = no_debugger
    except ImportError:
        pass

    entry = read(evaluation / "render_input.json")
    if entry["case_name"] != a.case:
        raise ValueError("Evaluation scene mismatch")
    traj = evaluation / "geometry/trajectory.npz"
    geometry_done = read(evaluation / "geometry/done.json")
    if geometry_done["evaluation"] != entry or geometry_done[
        "trajectory_sha256"
    ] != sha(traj):
        raise ValueError("Trajectory does not match the evaluation metadata")
    asset = entry["assets"]
    if not asset["ready"]:
        raise FileNotFoundError("Missing image assets: " + str(asset["missing"][:5]))
    for path, h in asset["identity_files"].items():
        if sha(path) != h:
            raise ValueError("Appearance asset changed: " + path)
    raw = Path(asset["raw"])
    gsdata = Path(asset["gsdata"])
    human = Path(asset["human"])
    ply = Path(asset["ply"])
    cfgpath = Path(asset["cfg"])
    cfg = cfg_read(cfgpath)
    split = read(raw / "split.json")
    z = np.load(traj, allow_pickle=False)
    pred = np.asarray(z["pred"], dtype=np.float32)
    ids = np.asarray(z["initial_ids"])
    T, N, xyz = pred.shape
    b = int(z["train_end"])
    raw_frames = z["raw_frame_indices"].astype(int)
    if (
        raw_frames.shape != (T,)
        or raw_frames[0] != 0
        or not np.all(np.diff(raw_frames) > 0)
    ):
        raise ValueError("Invalid raw frame map")
    if xyz != 3 or N < 17 or not np.isfinite(pred).all():
        raise ValueError("Invalid object predictions")
    if (
        b != int(np.searchsorted(raw_frames, split["train"][1]))
        or raw_frames[-1] >= split["frame_len"]
    ):
        raise ValueError("Trajectory and raw frame timeline do not match")
    with (raw / "final_data.pkl").open("rb") as f:
        original = pickle.load(f)
    original_x = original["object_points"]
    if hasattr(original_x, "detach"):
        original_x = original_x.detach().cpu().numpy()
    if ids.shape != (N,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Missing original particle IDs")
    initial_error = float(np.max(np.abs(pred[0] - np.asarray(original_x)[0, ids])))
    if initial_error > 3e-6:
        raise ValueError(f"Initial coordinates differ by {initial_error} m")
    del original, original_x
    info = read(raw / "mask/mask_info_0.json")
    obj_ids = [int(k) for k, label in info.items() if label != "hand"]
    if len(obj_ids) != 1:
        raise ValueError("Expected one foreground object label")
    object_masks = raw / "mask/0" / str(obj_ids[0])
    stop = min(T, 5) if a.smoke else T
    for t in raw_frames[:stop]:
        for p in [raw / f"color/0/{t}.png", object_masks / f"{t}.png"] + (
            [Path(d) / f"{t}.png" for d in asset["hand_dirs"]]
            if asset.get("human_mode") == "raw_hand_union"
            else [human / f"{t}.png"]
        ):
            if not p.is_file():
                raise FileNotFoundError(p)

    if asset.get("camera"):
        cam = asset["camera"]
        c2w = np.asarray(cam["c2w"])
        K = np.asarray(cam["K"])
        camera_error = max(m["pose_error"] for m in cam["audit"]["gs_camera_matches"])
        intrinsic_difference = cam["audit"]["intrinsics_1_vs_0_max_abs"]
    else:
        with (gsdata / "camera_meta.pkl").open("rb") as f:
            camera_meta = pickle.load(f)
        with (gsdata / "interp_poses.pkl").open("rb") as f:
            poses = pickle.load(f)
        c2w = np.asarray(poses[0])
        K = np.asarray(camera_meta["intrinsics"][1])
        camera_error = float(np.max(np.abs(c2w - np.asarray(camera_meta["c2ws"][0]))))
        intrinsic_difference = float(
            np.max(np.abs(K - np.asarray(camera_meta["intrinsics"][0])))
        )
        if camera_error > 1e-5:
            raise ValueError("Official interpolated view0 differs from camera0")
    with Image.open(raw / "color/0/0.png") as im:
        W, H = im.size
    if (W, H) != (848, 480):
        raise ValueError("Official evaluation expects 848x480; do not silently resize")
    camera = SimpleNamespace(
        K=K,
        world_view_transform=torch.as_tensor(
            np.linalg.inv(c2w).T.copy(), dtype=torch.float32, device="cuda"
        ),
        image_width=W,
        image_height=H,
    )
    if cfg.get("white_background", False):
        raise ValueError("Expected official black-background appearance config")
    code_paths = [
        official / p
        for p in [
            "gaussian_splatting/gaussian_renderer/__init__.py",
            "gaussian_splatting/dynamic_utils.py",
            "gaussian_splatting/scene/gaussian_model.py",
            "gaussian_splatting/utils/loss_utils.py",
            "gaussian_splatting/utils/image_utils.py",
            "gaussian_splatting/evaluate_render.py",
        ]
    ]
    code_paths += list((official / "gaussian_splatting/lpipsPyTorch").rglob("*.py"))
    request = dict(
        version=1,
        case=a.case,
        case_id=entry["case"],
        step=entry["step"],
        checkpoint_sha256=entry["checkpoint_sha256"],
        trajectory_sha256=sha(traj),
        ply_sha256=sha(ply),
        cfg_sha256=sha(cfgpath),
        cameras_sha256=asset["identity_files"],
        camera_adapter=asset.get("camera"),
        mask_protocol=asset.get("mask_protocol", "original separate human masks"),
        official_sources={str(p.relative_to(official)): sha(p) for p in code_paths},
        addon_sources={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        dataset=str(dataset),
        evaluation=str(evaluation),
        split=split,
        raw_frames=raw_frames.tolist(),
        smoke=a.smoke,
        chunk=a.chunk,
        save_all=a.save_all,
        initial_error_m=initial_error,
        camera_pose_error=camera_error,
        intrinsics_1_vs_0_max_abs=intrinsic_difference,
        protocol="official view0, intrinsics1 / original PNG quantization / documented hand-mask source / alpha>0 / LPIPS alex v0.1",
        motion="official spatial LBS K16; continuous saved predictions; no temporal smoothing or trajectory clipping",
        official_temporal_smoothing_applied=False,
    )
    if (out / "request.json").exists() and read(out / "request.json") != request:
        raise ValueError("Inputs changed: use fresh IMAGE_OUT")
    write(out / "request.json", request)
    if (out / "done.json").exists():
        if read(out / "done.json")["request"] != request:
            raise ValueError("Completion identity mismatch")
        if (
            not (out / "per_frame.csv").is_file()
            or not (out / "metrics.json").is_file()
        ):
            raise ValueError("Image completion marker exists but outputs are missing")
        done = read(out / "done.json")
        if any(
            sha(out / n) != done.get("output_hashes", {}).get(n)
            for n in ["per_frame.csv", "metrics.json"]
        ):
            raise ValueError("Cached image metrics changed")
        print("[SKIP COMPLETE]", a.case, flush=True)
        return

    # Reuse official helper definitions instead of a separately guessed metric implementation.
    tree = ast.parse((official / "gaussian_splatting/evaluate_render.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ["img2tensor", "compute_iou"]
    ]
    if len(nodes) != 2:
        raise ValueError("Official metric helper interface changed")
    env = dict(np=np, torch=torch)
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "official_metric_helpers", "exec"
        ),
        env,
    )
    img2tensor, compute_iou = env["img2tensor"], env["compute_iou"]
    # Use existing cached weights only; never download or substitute random weights.
    from .offline_weights import block_downloads

    block_downloads(torch)
    criterion = LPIPS(net_type="alex", version="0.1").cuda().eval()
    gaussians = GaussianModel(int(cfg.get("sh_degree", 3)))
    gaussians.isotropic = bool(cfg.get("isotropic", False))
    gaussians.load_ply(str(ply), False)
    with torch.no_grad():
        keep = gaussians.get_opacity.squeeze(-1) > 0.1
        for attr in [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
        ]:
            setattr(
                gaussians, attr, getattr(gaussians, attr).detach()[keep].contiguous()
            )
        if not len(gaussians._xyz):
            raise ValueError("No Gaussians survive official opacity filter")
        pos = gaussians.get_xyz.clone()
        quat = gaussians.get_rotation.clone()
        bones = torch.as_tensor(pred, device="cuda")
        relations = get_topk_indices(bones[0], K=16)
        background = torch.zeros(3, device="cuda")
        pipe = SimpleNamespace(
            debug=False,
            antialiasing=False,
            convert_SHs_python=False,
            compute_cov3D_python=False,
        )
        rows = []
        probes = sorted(set([0, 1, b - 1, b, T - 1]))
        zero_motion_steps = 0
        (out / "previews").mkdir(exist_ok=True)
        if a.save_all:
            (out / "renders/0").mkdir(parents=True, exist_ok=True)
        for t in range(stop):
            raw_t = int(raw_frames[t])
            if t >= int(entry.get("score_start", 3)):
                motions = bones[t] - bones[t - 1]
                if torch.count_nonzero(motions).item() == 0:
                    zero_motion_steps += 1
                for start in range(0, len(pos), a.chunk):
                    end = min(start + a.chunk, len(pos))
                    weights = knn_weights(bones[t - 1], pos[start:end], K=16)
                    pp, qq, _ = interpolate_motions(
                        bones=bones[t - 1],
                        motions=motions,
                        relations=relations,
                        weights=weights,
                        xyz=pos[start:end],
                        quat=quat[start:end],
                    )
                    pos[start:end] = pp
                    quat[start:end] = qq
                    del weights, pp, qq
            if not torch.isfinite(pos).all() or not torch.isfinite(quat).all():
                raise ValueError(
                    f"Nonfinite official LBS output at frame {t}; no particles removed"
                )
            gaussians._xyz = pos
            gaussians._rotation = torch.nn.functional.normalize(quat, dim=-1)
            override = (
                gaussians.get_features_dc.squeeze(1)
                if cfg.get("disable_sh", False)
                else None
            )
            rgba = render(
                camera,
                gaussians,
                pipe,
                background,
                override_color=override,
                use_gsplat=True,
            )["render"]
            if rgba.ndim != 3 or rgba.shape[0] != 4 or tuple(rgba.shape[1:]) != (H, W):
                raise ValueError(
                    "Renderer must return 4-channel RGBA at native resolution"
                )
            if not torch.isfinite(rgba).all():
                raise ValueError("Nonfinite rendered pixels")
            pixels = (
                (rgba.clamp(0, 1) * 255 + 0.5)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            gt = np.array(Image.open(raw / f"color/0/{raw_t}.png"))
            mask = np.array(Image.open(object_masks / f"{raw_t}.png"))
            hand = hand_mask(asset, raw_t)
            if gt.shape != (H, W, 3) or mask.shape != (H, W) or hand.shape != (H, W):
                raise ValueError(
                    "Unexpected RGB/mask shape; do not silently reshape evaluation data"
                )
            gt_mask = mask.astype(np.float32) / 255.0
            inv_hand = (1.0 - hand / 255.0).astype(np.float32)
            gt = gt.astype(np.float32) * gt_mask[..., None]
            gt[gt_mask == 0] = [0, 0, 0]
            gt = gt * inv_hand[..., None]
            rendered = pixels[:, :, :3].astype(np.float32) * inv_hand[..., None]
            render_mask = pixels[:, :, 3] * inv_hand
            if t >= int(entry.get("score_start", 3)):
                x, y = img2tensor(rendered), img2tensor(gt)
                row = dict(
                    frame=t,
                    raw_frame=raw_t,
                    split="E1" if t < b else "E2",
                    iou_pct=100 * float(compute_iou(gt_mask > 0, render_mask > 0)),
                    psnr=float(psnr(x, y).item()),
                    ssim=float(ssim(x, y).item()),
                    lpips=float(criterion(x, y).item()),
                )
                if not all(
                    np.isfinite(row[k]) for k in ["iou_pct", "psnr", "ssim", "lpips"]
                ):
                    raise ValueError(
                        "Nonfinite image metric; inspect rather than truncate"
                    )
                rows.append(row)
            if a.save_all:
                Image.fromarray(pixels).save(out / f"renders/0/{t:05d}.png")
            if t in probes:
                Image.fromarray(pixels).save(out / f"previews/{t:05d}_rgba.png")
                joined = np.concatenate(
                    [
                        np.clip(gt, 0, 255).astype(np.uint8),
                        np.clip(rendered, 0, 255).astype(np.uint8),
                    ],
                    axis=1,
                )
                Image.fromarray(joined).save(out / f"previews/{t:05d}_gt_pred.png")
            if t % 10 == 0 or t == stop - 1:
                print(f"[FRAME] {a.case} {t+1}/{stop}", flush=True)
        csv_write(out / "per_frame.csv", rows)
        result = dict(
            status="complete",
            request=request,
            metrics=aggregate(rows),
            gaussians=len(pos),
            zero_motion_steps=zero_motion_steps,
            rendered_frames=stop,
            smoke_only=a.smoke,
        )
        write(out / "metrics.json", result)
        write(
            out / "done.json",
            dict(
                result,
                output_hashes={
                    n: sha(out / n) for n in ["metrics.json", "per_frame.csv"]
                },
            ),
        )
        print("[DONE]", a.case, result["metrics"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["official", "dataset", "evaluation", "out", "case"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--reuse-root", action="append", default=[])
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--save-all", action="store_true")
    a = p.parse_args()
    if a.chunk < 1:
        p.error("--chunk must be positive")
    main(a)
