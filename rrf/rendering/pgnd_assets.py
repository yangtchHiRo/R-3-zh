import argparse
from pathlib import Path
import numpy as np
from ..io import read, write, sha


def sequence(pattern, root, frames, cam):
    return [
        str((root / str(pattern).format(frame=int(f), cam=cam)).resolve())
        for f in frames
    ]


def find_sequence(kind, root, frames, cam, override):
    if override:
        paths = sequence(override, root, frames, cam)
        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{kind}: missing {len(missing)} frames, first: {missing[0]}"
            )
        return paths
    labels = {
        "rgb": ["color", "rgb", "images"],
        "mask": ["mask", "masks", "seg", "segmentation"],
        "ignore": ["robot_mask", "gripper_mask", "human_mask", "ignore_mask"],
    }[kind]
    patterns = []
    for label in labels:
        for stem in ("{frame:06d}", "{frame:04d}", "{frame:05d}", "{frame}"):
            for ext in ("png", "jpg") if kind == "rgb" else ("png",):
                patterns += [
                    f"{label}/{{cam}}/{stem}.{ext}",
                    f"{label}/camera_{{cam}}/{stem}.{ext}",
                    f"camera_{{cam}}/{label}/{stem}.{ext}",
                    f"camera_{{cam}}/{stem}_{label}.{ext}",
                    f"{label}/{{cam}}/0/{stem}.{ext}",
                    f"camera_{{cam}}/{label}_{stem}.{ext}",
                ]
    hits = {
        tuple(sequence(s, root, frames, cam))
        for s in patterns
        if all((Path(p).is_file() for p in sequence(s, root, frames, cam)))
    }
    if len(hits) == 1:
        return list(next(iter(hits)))
    if len(hits) > 1:
        raise ValueError(f"Ambiguous {kind} paths; specify a pattern in --assets-json")
    if kind == "ignore":
        return None
    samples = [str(p.relative_to(root)) for p in sorted(root.rglob("*.png"))[:20]]
    raise FileNotFoundError(
        f"{kind}: cannot resolve exact camera {cam}, frames {frames[0]}..{frames[-1]} in {root}; PNG samples={samples}. Set rgb_pattern/mask_pattern in --assets-json."
    )


def resolve(e, ep, pgnd, cam, asset_root, override):
    raw = Path(e["registry"]["raw_source"])
    meta_path = raw / f"episode_{ep:04d}" / "meta.txt"
    meta = np.loadtxt(meta_path).reshape(-1)
    if len(meta) < 3 or not np.all(meta[:3] == np.round(meta[:3])):
        raise ValueError("Invalid meta.txt")
    mapping = read(raw / "metadata.json")
    entry = (
        mapping[ep]
        if isinstance(mapping, list)
        else mapping.get(str(ep), mapping.get(f"episode_{ep:04d}"))
    )
    original = Path(entry["path"])
    source_ep = int(meta[0])
    start = int(meta[1])
    cfg = e["config"]
    if (
        int(cfg["sim"]["n_history"]) != 2
        or int(cfg["train"]["dataset_load_skip_frame"]) != 3
        or int(cfg["train"]["dataset_skip_frame"]) != 1
    ):
        raise ValueError(
            "Unsupported timeline: expected history=2, load_stride=3, stride=1"
        )
    frames = (start + 6 + 3 * np.arange(38)).tolist()
    name = f"episode_{source_ep:04d}"
    dataset = original.parent.name
    roots = [
        (pgnd / original).parent / name,
        pgnd / "experiments/log/data" / dataset / name,
        pgnd / "data" / dataset / name,
        pgnd
        / f"data_{e['config']['train'].get('material', 'unused')}"
        / dataset
        / name,
    ]
    if asset_root:
        roots.insert(0, Path(asset_root) / dataset / name)
    roots += [parent / dataset / name for parent in raw.parents[:3]]
    if override.get("raw_dir"):
        roots = [Path(override["raw_dir"])]
    roots = list(dict.fromkeys((r.resolve() for r in roots)))
    hits = [r for r in roots if (r / "gs" / f"{frames[0]:06d}.splat").is_file()]
    if not hits:
        raise FileNotFoundError(
            "Exact initial splat missing; candidates: "
            + ", ".join((str(r / "gs" / f"{frames[0]:06d}.splat") for r in roots))
        )
    if (
        len(hits) > 1
        and len({sha(r / "gs" / f"{frames[0]:06d}.splat") for r in hits}) > 1
    ):
        raise ValueError("Different initial splats found; specify raw_dir")
    root = hits[0]
    paths = {
        k: str(root / "calibration" / f"{k}.npy")
        for k in ("intrinsics", "rvecs", "tvecs")
    }
    for p in paths.values():
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    intr = np.load(paths["intrinsics"])
    rv = np.load(paths["rvecs"])
    tv = np.load(paths["tvecs"])
    if not 0 <= cam < min(len(intr), len(rv), len(tv)):
        raise ValueError("Camera absent from calibration")
    if intr[cam].shape != (3, 3):
        raise ValueError("Expected intrinsics[C,3,3]")
    rgb = find_sequence("rgb", root, frames, cam, override.get("rgb_pattern"))
    mask = find_sequence("mask", root, frames, cam, override.get("mask_pattern"))
    ignore = find_sequence("ignore", root, frames, cam, override.get("ignore_pattern"))
    from PIL import Image

    sizes = set()
    for i, (r, m) in enumerate(zip(rgb, mask)):
        with Image.open(r) as im:
            size = im.size
        sizes.add(size)
        if binary_image(m).shape != size[::-1]:
            raise ValueError("GT RGB/mask resolution differs")
        if ignore and binary_image(ignore[i]).shape != size[::-1]:
            raise ValueError("Ignore mask resolution differs")
    if len(sizes) != 1:
        raise ValueError("Variable image resolution")
    width, height = next(iter(sizes))
    calib_size = override.get("calibration_size", [848, 480])
    if [width, height] != list(calib_size):
        raise ValueError(
            f"RGB {width}x{height} != calibration_size {calib_size}; confirm native calibration resolution in assets JSON. No automatic intrinsics guessing/resizing."
        )
    return dict(
        raw_dir=str(root),
        source_episode=source_ep,
        frames=frames,
        camera=cam,
        width=width,
        height=height,
        splat=str(root / "gs" / f"{frames[0]:06d}.splat"),
        calibration=paths,
        rgb=rgb,
        mask=mask,
        ignore=ignore,
        occlusion_policy=(
            "exclude_explicit_masks" if ignore else "no_occluder_exclusion"
        ),
    )


def binary_image(path):
    from PIL import Image

    a = np.array(Image.open(path))
    if a.ndim == 3:
        if not np.array_equal(a[..., 0], a[..., 1]) or not np.array_equal(
            a[..., 0], a[..., 2]
        ):
            raise ValueError(
                "Colored/instance mask needs explicit binary conversion: " + str(path)
            )
        a = a[..., 0]
    if a.ndim != 2 or not set(np.unique(a)).issubset({0, 1, 255, False, True}):
        raise ValueError("Expected binary object/ignore mask: " + str(path))
    return a > 0


def main():
    p = argparse.ArgumentParser(
        description="Resolve original RGB/GS/calibration via subepisode meta.txt and metadata.json"
    )
    p.add_argument("--data", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--asset-root")
    p.add_argument(
        "--override",
        help="JSON with raw_dir/rgb_pattern/mask_pattern/calibration_size overrides",
    )
    a = p.parse_args()
    data = Path(a.data)
    paths = read(data / "paths.json")
    config = read(data / "pgnd_config.json")
    assets = resolve(
        dict(registry=paths, config=config),
        a.episode,
        Path(paths["pgnd_root"]),
        1,
        a.asset_root,
        read(a.override) if a.override else {},
    )
    if assets["ignore"]:
        raise ValueError(
            "Official visual protocol has no ignore-mask exclusion; inspect assets explicitly"
        )
    write(a.out, assets)
    print(a.out)


if __name__ == "__main__":
    main()
