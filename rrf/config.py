"""Shared task configuration, inheritance, overrides and validation."""

import copy
import json
import math
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
TASKS = tuple(sorted(p.stem for p in (CONFIG_DIR / "tasks").glob("*.json")))


def merge(base, patch, path=""):
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {path}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path}{key} must be an object")
            result[key] = merge(base[key], value, path + key + ".")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError("Cyclic config inheritance: " + str(path))
    value = json.loads(path.read_text())
    parent = value.pop("extends", None)
    if parent:
        return merge(_load(path.parent / parent, (*stack, path)), value)
    return value


def load_config(task=None, path=None, overrides=()):
    if task and path:
        raise ValueError("Choose either --task or --config")
    if task and task not in TASKS:
        raise ValueError("Unknown task: " + task)
    path = (
        Path(path)
        if path
        else CONFIG_DIR / "tasks" / f'{task or "phystwin_scene"}.json'
    )
    cfg = merge(_load(CONFIG_DIR / "base.json"), _load(path))
    for assignment in overrides:
        key, sep, raw = assignment.partition("=")
        if not sep:
            raise ValueError("Override must be dotted.key=value")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError("Unknown override: " + key)
            node = node[part]
        if parts[-1] not in node or isinstance(node[parts[-1]], dict):
            raise ValueError("Override an existing scalar/list field: " + key)
        node[parts[-1]] = value
    validate(cfg)
    return cfg


def validate(c):
    if c["dataset"] not in ("phystwin", "pgnd"):
        raise ValueError("Unknown dataset")
    if (
        c["model"]["history"] != 3
        or c["data"]["horizon"] != 5
        or c["data"]["batch_size"] != 2
    ):
        raise ValueError("RRF uses true H3, H5, and two serial streams")
    if not 0 <= c["data"]["continue_probability"] <= 1:
        raise ValueError("Invalid continuation probability")
    for value in [
        c["model"]["regions"],
        c["model"]["candidates"],
        c["data"]["frame_cache_size"],
        *[
            c["train"][k]
            for k in ("steps", "checkpoint_interval", "resume_interval", "log_interval")
        ],
    ]:
        if type(value) is not int or value < 1:
            raise ValueError("Sizes/intervals must be positive integers")
    if c["optimizer"]["name"] != "adam":
        raise ValueError("Expected Adam")
    if c["train"]["loss_scale"] != 100.0:
        raise ValueError("RRF position/coarse loss uses scale 100")
    if c["optimizer"]["lr"] <= 0 or c["train"]["grad_clip"] <= 0:
        raise ValueError("Invalid optimization settings")
    if c["seed"] < 0:
        raise ValueError("Invalid seed")


def resolve_args(args):
    """Resolve the task configuration and command-line overrides."""
    task = args.task
    if not task and not args.config:
        task = "pgnd" if args.dataset == "pgnd" else "phystwin_scene"
    overrides = list(args.set or [])
    if getattr(args, "seed", None) is not None:
        overrides.append(f"seed={args.seed}")
    c = load_config(task, args.config, overrides)
    if args.dataset and args.dataset != c["dataset"]:
        raise ValueError("--dataset conflicts with the task configuration")
    return c


def add_config_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", choices=TASKS)
    group.add_argument(
        "--config", help="JSON config, optionally extending another JSON"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--dataset",
        choices=["phystwin", "pgnd"],
        help="Dataset shorthand for the default task",
    )
    parser.add_argument("--seed", type=int, help="Shorthand for --set seed=...")
