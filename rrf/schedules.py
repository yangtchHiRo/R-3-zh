"""Task-independent optimizer and learning-rate schedule."""

import math


def learning_rate(step, config):
    if step < 1:
        raise ValueError("Optimizer updates are indexed from one")
    base = config["optimizer"]["lr"]
    s = config["scheduler"]
    if s["name"] == "constant":
        return base
    if s["name"] == "exponential":
        return s["min_lr"] + (base - s["min_lr"]) * s["decay_rate"] ** (
            (step - 1) / s["decay_steps"]
        )
    raise ValueError("Unknown learning rate schedule")


def build_optimizer(model, config):
    import torch

    o = config["optimizer"]
    if o["name"] != "adam":
        raise ValueError("Unsupported optimizer")
    return torch.optim.Adam(
        model.parameters(),
        lr=o["lr"],
        betas=tuple(o["betas"]),
        eps=o["eps"],
        weight_decay=o["weight_decay"],
    )
