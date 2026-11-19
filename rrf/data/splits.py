"""User-defined train/test splits with source-only normalization."""

from pathlib import Path
from ..io import read


def scene_groups(args, preset_file):
    custom = bool(
        args.split_file
        or args.train_scenes
        or args.test_scenes
        or args.validation_scenes
    )
    if not custom:
        groups = read(preset_file)
        result = (
            groups
            if args.groups == ["all"]
            else [g for g in groups if g["name"] in args.groups]
        )
        if args.groups != ["all"] and set(args.groups) != {g["name"] for g in result}:
            raise ValueError("Unknown predefined group")
        return result
    if args.groups != ["all"]:
        raise ValueError("Use custom splits or --groups, not both")
    if args.split_file:
        if args.train_scenes or args.test_scenes or args.validation_scenes:
            raise ValueError("Use --split-file or inline scene lists, not both")
        value = read(args.split_file)
        groups = value.get("groups", [value]) if isinstance(value, dict) else value
    else:
        groups = [
            dict(
                name=args.group_name,
                train=args.train_scenes,
                test=args.test_scenes,
                validation=args.validation_scenes or [],
                kind=args.generalization,
                source=args.source,
            )
        ]
    if not isinstance(groups, list) or not groups:
        raise ValueError("Provide at least one custom group")
    result = []
    for item in groups:
        group = dict(
            experiment="generalization",
            kind="scene",
            source="main",
            material="deformable",
            validation=[],
        )
        group.update(item)
        if group["experiment"] != "generalization" or group["kind"] not in (
            "scene",
            "action",
        ):
            raise ValueError("Custom groups require scene or action generalization")
        if group["source"] not in ("main", "additional"):
            raise ValueError("source must be main or additional")
        name = group.get("name")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise ValueError("Provide a simple output group name")
        sets = []
        for part in ("train", "validation", "test"):
            names = group.get(part)
            if not isinstance(names, list) or (part != "validation" and not names):
                raise ValueError(
                    f"{part} must be a list; train and test must be nonempty"
                )
            if any(
                not isinstance(n, str) or not n or Path(n).name != n or n in (".", "..")
                for n in names
            ):
                raise ValueError(
                    "Scene names must be directory names under the data root"
                )
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate scene in {part}")
            sets.append(set(names))
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(
                "Custom generalization train, validation and test scenes must be disjoint"
            )
        result.append(group)
    if len({g["name"] for g in result}) != len(result):
        raise ValueError("Duplicate output group name")
    return result


def episode_ids(train, test):
    for name, ids in (("train_ids", train), ("test_ids", test)):
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i < 0 for i in ids)
        ):
            raise ValueError(f"{name} must be a nonempty list of nonnegative integers")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate episode in {name}")
    if set(train) & set(test):
        raise ValueError("Training and test episodes must be disjoint")
    return sorted(train), sorted(test)
