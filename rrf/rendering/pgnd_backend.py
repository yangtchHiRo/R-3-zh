import importlib.util, inspect, sys
from pathlib import Path


def load_official(pgnd):
    pgnd = Path(pgnd).resolve()
    marker = pgnd / "experiments/train"
    while not (marker / ".root").is_file():
        if marker == marker.parent:
            raise FileNotFoundError(
                "PGND .root marker missing; restore the official repository layout"
            )
        marker = marker.parent
    sys.path[:0] = [
        str(pgnd),
        str(pgnd / "experiments"),
        str(pgnd / "experiments/train"),
    ]
    from pgnd.utils import get_root

    modules = {}
    for stem in ("render_utils", "helpers", "convert", "gs", "metric"):
        name = "_rrf_official_" + stem
        paths = {
            "render_utils": "experiments/real_world/utils/render_utils.py",
            "helpers": "experiments/real_world/gs/helpers.py",
            "convert": "experiments/real_world/gs/convert.py",
            "gs": "experiments/train/gs.py",
            "metric": "experiments/train/metric.py",
        }
        path = pgnd / paths[stem]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.root = get_root(pgnd / "experiments/train/gs.py")
        modules[stem] = module
    # Bind dependencies from the explicitly selected external repository.
    modules["gs"].interpolate_motions = modules["render_utils"].interpolate_motions
    modules["gs"].setup_camera = modules["helpers"].setup_camera
    modules["gs"].read_splat = modules["convert"].read_splat
    modules["gs"].save_to_splat = modules["convert"].save_to_splat
    return modules["gs"], modules["metric"]


def fp32(values, device):
    import torch

    return {
        k: (
            v.to(device=device, dtype=torch.float32).contiguous()
            if torch.is_tensor(v) and v.is_floating_point()
            else v.to(device=device).contiguous() if torch.is_tensor(v) else v
        )
        for k, v in values.items()
    }


def install_camera(module):
    import torch

    original = module.setup_camera

    def setup(*args, **kwargs):
        cam = original(*args, **kwargs)
        if not hasattr(cam, "_asdict") or not hasattr(cam, "_replace"):
            raise TypeError("Unsupported rasterizer camera settings")
        fields = cam._asdict()
        device = fields["viewmatrix"].device
        tensors = fp32({k: v for k, v in fields.items() if torch.is_tensor(v)}, device)
        if "bg" not in fields:
            raise TypeError("Cannot establish a black raster background")
        tensors["bg"] = torch.zeros(3, device=device, dtype=torch.float32)
        return cam._replace(**tensors)

    module.setup_camera = setup


def camera(assets):
    import numpy as np, cv2

    c = assets["camera"]
    p = assets["calibration"]
    intr = np.load(p["intrinsics"])
    rv = np.load(p["rvecs"])
    tv = np.load(p["tvecs"])
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = cv2.Rodrigues(rv[c])[0]
    w2c[:3, 3] = tv[c].reshape(3)
    return dict(
        w=assets["width"],
        h=assets["height"],
        intr=np.asarray(intr[c], dtype=np.float32),
        w2c=w2c,
    )
