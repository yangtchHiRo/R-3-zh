from pathlib import Path
from ..io import read, sha


def assets(dataset, source, name, frames):
    base = dataset if source == "main" else dataset / "additional_data"
    raw = base / "data/different_types" / name
    a = dict(
        raw=str(raw),
        gsdata=str(base / "data/gaussian_data" / name),
        human=str(base / "data/different_types_human_mask" / name / "mask/0/0"),
    )
    candidates = list(
        (base / "gaussian_output" / name).glob(
            "*/point_cloud/iteration_10000/point_cloud.ply"
        )
    )
    # Some installations store additional appearances in a shared gaussian_output.
    if source == "additional":
        candidates += list(
            (dataset / "gaussian_output" / name).glob(
                "*/point_cloud/iteration_10000/point_cloud.ply"
            )
        )
    candidates = sorted(set(p.resolve() for p in candidates))
    missing = []
    if len(candidates) != 1:
        missing.append(
            "Expected exactly one iteration_10000 PLY; found " + str(candidates)
        )
    else:
        a["ply"] = str(candidates[0])
        a["cfg"] = str(candidates[0].parents[2] / "cfg_args")
    info = raw / "mask/mask_info_0.json"
    if info.exists():
        ids = [int(k) for k, v in read(info).items() if v != "hand"]
        if len(ids) == 1:
            a["object_masks"] = str(raw / "mask/0" / str(ids[0]))
        else:
            missing.append("Ambiguous object mask labels")
    else:
        missing.append(str(info))
    required = [Path(a["gsdata"]) / n for n in ["camera_meta.pkl", "interp_poses.pkl"]]
    if a.get("cfg"):
        required.append(Path(a["cfg"]))
    for t in frames:
        required += [raw / f"color/0/{t}.png", Path(a["human"]) / f"{t}.png"]
        if a.get("object_masks"):
            required.append(Path(a["object_masks"]) / f"{t}.png")
    missing += [str(p) for p in required if not p.is_file()]
    a["missing"] = missing
    a["ready"] = not missing
    a["identity_files"] = {str(p): sha(p) for p in required[:2] if p.is_file()}
    if a.get("ply"):
        a["identity_files"][a["ply"]] = sha(a["ply"])
        if Path(a["cfg"]).exists():
            a["identity_files"][a["cfg"]] = sha(a["cfg"])
    return a
