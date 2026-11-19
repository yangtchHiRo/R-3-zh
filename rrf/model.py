"""Shared model factory; no wrapper prefixes in checkpoint parameter names."""

from .models.network import RRF
from .data.adapters import metadata as physical_metadata


def model_config(metadata, experiment):
    m = experiment["model"]
    return dict(
        architecture="RRF",
        metadata=metadata,
        settings=dict(
            candidates=m["candidates"],
            patches=m["regions"],
            coarse_loss_weight=m["coarse_loss_weight"],
        ),
    )


def build_model(config, metadata=None):
    if "schema_version" in config:
        if metadata is None:
            raise ValueError("Source metadata is required")
        config = model_config(metadata, config)
    if config.get("architecture") != "RRF":
        raise ValueError("Expected an RRF configuration")
    return RRF(config["metadata"], config["settings"])
