"""Render saved PGND rollouts using externally installed official components."""

import argparse, os
from pathlib import Path
import numpy as np
from ..io import read, write, sha


def save_official_png(image, white, rgb_path, mask_path):
    """Exact conversion order from official gs.py do_gs, then metric_eval.py reads."""
    import cv2

    im = (image * 255).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    im[:, :, 3] = (white * 255).mean(-1).astype(np.uint8)
    mask = ((white > 0.1).astype(np.float32) * 255).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    for path, value in ((rgb_path, im), (mask_path, mask)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), value):
            raise IOError("Cannot write " + str(path))
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2GRAY) > 0
    return (rgb, mask)


def score(metric, model, rgb, mask, gt, gt_mask):
    if not gt_mask.any():
        raise ValueError("Empty GT mask: official PSNR undefined")
    j = metric.compute_j(mask, gt_mask)
    f = metric.compute_f(mask, gt_mask)
    result = dict(
        j=float(j),
        f=float(f),
        jf=float((j + f) / 2),
        lpips=float(metric.compute_lpips(model, rgb, gt)),
        psnr=float(metric.calc_psnr(rgb, gt, gt_mask)),
        ssim=float(metric.calc_ssim(rgb, gt, gt_mask)),
    )
    if not all((np.isfinite(v) for v in result.values())):
        raise ValueError("Nonfinite official metric: " + str(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pgnd-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    pgnd = Path(a.pgnd_root).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trajectory = Path(a.trajectory).resolve()
    from .pgnd_backend import load_official, install_camera, fp32, camera
    import torch, cv2, lpips
    from omegaconf import OmegaConf

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Official Gaussian rendering needs CUDA")
    config = read(a.config)
    assets = read(a.assets)
    frames = assets["frames"]
    if (
        assets["camera"] != 1
        or len(frames) != 38
        or frames[0] < 6
        or not np.all(np.diff(frames) == 3)
    ):
        raise ValueError(
            "Expected camera1, 38 source frames at stride3; preserve subepisode offset"
        )
    for key in ("rgb", "mask"):
        if len(assets[key]) != 38 or not all(Path(x).is_file() for x in assets[key]):
            raise ValueError("Need 38 explicit existing " + key + " paths")
    with np.load(trajectory) as z:
        pred = z["pred"]
        x0 = z["x0"]
        dt = float(z["dt"])
    if (
        len(pred) != 37
        or pred.shape[1:] != x0.shape
        or not np.isfinite(pred).all()
        or abs(dt - 0.1) > 1e-6
    ):
        raise ValueError("Invalid trajectory")
    # Resolve absolute paths before changing cwd for the upstream renderer.
    os.chdir(pgnd)
    gs, metric = load_official(pgnd)
    install_camera(gs)
    renderer = gs.GSRenderer(OmegaConf.create(config).render, device="cuda:0")
    renderer.load_params(Path(assets["splat"]))
    renderer.params = fp32(renderer.params, renderer.device)
    renderer.set_camera(**camera(assets))
    loss = lpips.LPIPS(net="alex").eval()
    rows = []
    with torch.no_grad():
        bones = torch.tensor(
            np.concatenate([x0[None], pred]), dtype=torch.float32, device="cuda:0"
        )
        render_vars, _ = renderer.rollout_and_render(bones, with_bg=False)
        if len(render_vars) != 38:
            raise ValueError("Unexpected render count")
        for h, var in enumerate(render_vars):
            var = fp32(var, renderer.device)
            image, _ = renderer.render(var, 0)
            var["colors_precomp"] = torch.ones_like(var["colors_precomp"])
            white, _ = renderer.render(var, 0)
            if not torch.isfinite(image).all() or not torch.isfinite(white).all():
                raise ValueError("Nonfinite render")
            rgb, mask = save_official_png(
                image.cpu().numpy().transpose(1, 2, 0),
                white.cpu().numpy().transpose(1, 2, 0),
                out / "rgb" / f"{h:04d}.png",
                out / "mask" / f"{h:04d}.png",
            )
            gt = cv2.cvtColor(cv2.imread(assets["rgb"][h]), cv2.COLOR_BGR2RGB)
            gm = cv2.cvtColor(cv2.imread(assets["mask"][h]), cv2.COLOR_BGR2GRAY) > 0
            gt = gt * gm[..., None]
            if rgb.shape != gt.shape or mask.shape != gm.shape:
                raise ValueError("Render/GT resolution mismatch")
            if h:
                rows.append(
                    dict(
                        prediction_step=h,
                        time_s=h * 0.1,
                        raw_frame=assets["frames"][h],
                        **score(metric, loss, rgb, mask, gt, gm),
                    )
                )
    write(out / "frames.json", rows)
    write(
        out / "protocol.json",
        dict(
            official_code={
                str(p.relative_to(pgnd)): sha(p)
                for p in [
                    pgnd / "experiments/train/gs.py",
                    pgnd / "experiments/train/metric.py",
                ]
            },
            background="object-only black",
            alpha_threshold=0.1,
            psnr="upstream uint8 arithmetic, not standard float PSNR",
            lpips="upstream alex CPU; RGB uint8/255; no additional [-1,1] normalization",
            trajectory_sha256=sha(trajectory),
        ),
    )


if __name__ == "__main__":
    main()
