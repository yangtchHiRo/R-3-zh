from pathlib import Path
import torch
from .model import build_model
from .io import read, sha


def resolve(path):
    """Resolve a model file or the final checkpoint of a completed training run."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        model_dir = path / "model" if (path / "model").is_dir() else path
        marker = model_dir / "complete.json"
        if not marker.is_file():
            raise ValueError(f"Training is not complete: {model_dir}")
        done = read(marker)
        checkpoint = model_dir / done.get("checkpoint", f"step_{done['step']:07d}.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if (
            done.get("checkpoint_sha256")
            and sha(checkpoint) != done["checkpoint_sha256"]
        ):
            raise ValueError("Final checkpoint differs from its training record")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state.get("step") != done["step"]
            or state.get("identity") != done["identity"]
        ):
            raise ValueError("Final checkpoint does not match completed training")
        return checkpoint
    if path.suffix != ".pt":
        raise ValueError("Pass a training run directory or a .pt checkpoint")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Completed training run directory")
    group.add_argument("--checkpoint", help="Model .pt file")


def from_args(args):
    return resolve(args.checkpoint or args.run)


def load(path, device="cpu"):
    path = resolve(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck.get("model_config", ck.get("config"))
    if c is None or "metadata" not in c or "model" not in ck:
        raise ValueError("Expected RRF model/config state dictionary")
    model = build_model(c).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ck, path
