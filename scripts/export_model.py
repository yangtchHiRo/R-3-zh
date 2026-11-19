"""Export inference weights without optimizer, sampler state or machine paths."""

import argparse
import torch
from rrf.io import read, atomic_torch
from rrf.model import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--plan",
        help="Original experiment plan, only for an old checkpoint without model_config",
    )
    p.add_argument(
        "--key", help="Metadata key in that plan, e.g. phystwin/double_stretch_zebra"
    )
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    config = ck.get("model_config")
    if config is None:
        if not a.plan or not a.key:
            p.error("Old weights require --plan and --key")
        plan = read(a.plan)
        config = dict(
            architecture="RRF",
            metadata=plan["metadata"][a.key],
            settings={
                "patches": plan["settings"].get("patches", 64),
                "candidates": plan["settings"].get("candidates", 8),
                "coarse_loss_weight": plan["settings"].get("coarse_loss_weight", 0.1),
            },
        )
    # Keep only shape, time and normalization statistics used by the network.
    meta = config["metadata"]
    config = dict(
        architecture="RRF",
        metadata={
            k: meta[k]
            for k in (
                "dt",
                "scale",
                "n_particles",
                "connectivity_radius",
                "normalization",
                "control_availability_mask",
            )
            if k in meta
        },
        settings=dict(config["settings"]),
    )
    network = build_model(config)
    network.load_state_dict(ck["model"], strict=True)
    atomic_torch(
        dict(model=network.state_dict(), model_config=config, step=ck.get("step", 0)),
        a.out,
    )
    print(a.out)


if __name__ == "__main__":
    main()
