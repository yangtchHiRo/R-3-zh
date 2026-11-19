"""Prepare official processed observations; never fit appearance or download data."""

import argparse, copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .io import read, write, sha, fps, atomic_npz

CONFIG = Path(__file__).resolve().parent / "configs"


def phystwin(a):
    from .data.phystwin_prepare import read_case, pooled_metadata
    from .data.splits import scene_groups

    selected = scene_groups(a, CONFIG / "phystwin_groups.json")
    out = Path(a.out).resolve()
    for g in selected:
        dataset = Path(g.get("data_root", a.data_root)).expanduser().resolve()
        dest = out / g["name"]
        entries = []
        items = {}
        if (dest / "manifest.json").exists():
            raise FileExistsError(
                "Prepared group exists; reuse it or choose a new --out: " + str(dest)
            )
        for name in sorted(set(g["train"] + g["validation"] + g["test"])):
            prefix = dataset if g["source"] == "main" else dataset / "additional_data"
            p = prefix / "data/different_types" / name / "final_data.pkl"
            args = SimpleNamespace(
                root=str(dataset), min_particles=32, particles=1000, fps=None
            )
            e, ar = read_case(p, args)
            e.update(id=name, name=name)
            b = e["train_end"]
            single = g["experiment"] == "main22"
            fit_b = (
                b
                if single or (name in g["train"] and g["kind"] == "action")
                else (e["T"] if name in g["train"] else 0)
            )
            e.update(
                train_end=fit_b,
                official_train_end=b,
                validation_start=fit_b,
                raw_frames=list(range(e["T"])),
            )
            items[name] = (ar, fit_b)
            cache = dest / "data" / f"{name}.npz"
            atomic_npz(cache, **ar)
            e.update(cache=str(cache), cache_sha256=sha(cache))
            entries.append(e)
        if len({round(e["dt"], 9) for e in entries}) != 1:
            raise ValueError("Mixed native frame rates in group")
        meta = (
            entries[0]["metadata"]
            if single
            else pooled_metadata([items[n] for n in g["train"]], 1000)
        )
        for e in entries:
            e["metadata"] = meta
        manifest = dict(
            dataset="phystwin",
            dataset_root=str(dataset),
            group=g,
            particles=1000,
            min_particles=32,
            metadata=meta,
            cases=entries,
        )
        write(dest / "manifest.json", manifest)
        print("[PREPARED]", dest / "manifest.json", flush=True)


def pgnd(a):
    import torch
    from .data.pgnd_config import SPECS, split_ids, make_config
    from .data.pgnd_prepare import (
        official_preprocessor,
        ensure_native,
        frame_files,
        load_pt,
        fixed_fps,
        compute_statistics,
    )

    tasks = list(SPECS) if a.tasks == ["all"] else a.tasks
    paths = read(a.paths)
    out = Path(a.out).resolve()
    for task in tasks:
        e = paths[task]
        raw = Path(e["raw_source"]).expanduser().resolve()
        dest = out / task
        if (dest / "manifest.json").exists():
            raise FileExistsError(
                "Prepared task exists; reuse it or choose a new --out: " + str(dest)
            )
        native = (
            Path(e["native_state"]).expanduser().resolve()
            if e.get("native_state")
            else dest / "native/state"
        )
        cfg = make_config(a.pgnd_root, task, raw, native.parent)
        policy = "cloth_train_dedup_v1" if task == "cloth" else "official"
        if "train_ids" in e or "test_ids" in e:
            from .data.splits import episode_ids

            tr, ev = episode_ids(e.get("train_ids"), e.get("test_ids"))
            policy = "custom"
            if task == "cloth":
                from .data.pgnd_config import CLOTH_DUPLICATE_PAIRS

                if any(
                    (a in tr and b in ev) or (b in tr and a in ev)
                    for a, b in CLOTH_DUPLICATE_PAIRS.items()
                ):
                    raise ValueError(
                        "Cloth split crosses known duplicate episode pairs"
                    )
        else:
            tr, ev = split_ids(task, policy)
        n = SPECS[task][2]
        G = SPECS[task][3]
        if a.preprocess:
            fn, info = official_preprocessor(a.pgnd_root)
            for eid in tr + ev:
                ensure_native(
                    raw / f"episode_{eid:04d}",
                    native / f"episode_{eid:04d}",
                    cfg,
                    fn,
                    info,
                    n,
                    G,
                )
        entries = {"train": [], "eval": []}

        def train_items():
            for eid in tr:
                ep = native / f"episode_{eid:04d}"
                files = frame_files(ep)[::3]
                if len(files) < 8:
                    raise ValueError("No complete Full-5 window: " + str(ep))
                first = load_pt(files[2])
                ids = fixed_fps(first["x"], n, eid * 104729, a.device)
                x = torch.stack([load_pt(p)["x"][ids] for p in files])
                entries["train"].append(
                    dict(episode=eid, source_directory=str(ep), num_frames=len(files))
                )
                yield eid, x

        # Source statistics are computed only from the training partition.
        if e.get("reuse_cache"):
            import copy
            from .data.pgnd_config import validate_manifest

            old = Path(e["reuse_cache"])
            meta = read(old / "metadata.json")
            oldm = read(old / "manifest.json")
            validate_manifest(oldm, meta, task)
            if [v["episode"] for v in oldm["train"]] != tr or [
                v["episode"] for v in oldm["eval"]
            ] != ev:
                raise ValueError("Cached split differs from requested split")
            if oldm.get("split_policy", "official") != policy:
                raise ValueError("Archived split differs from release split")
            for eid in tr:
                ep = native / f"episode_{eid:04d}"
                files = frame_files(ep)[::3]
                previous = next(x for x in oldm["train"] if x["episode"] == eid)
                if len(files) != previous["num_frames"]:
                    raise ValueError("Archived native timeline changed")
                entries["train"].append(
                    dict(episode=eid, source_directory=str(ep), num_frames=len(files))
                )
        else:
            meta = compute_statistics(train_items(), task, cfg, policy, train_ids=tr)
        for eid in ev:
            ep = native / f"episode_{eid:04d}"
            files = frame_files(ep)[::3]
            if len(files) != 40:
                raise ValueError(
                    "Paper evaluation needs 3 history + 37 future states: " + str(ep)
                )
            entries["eval"].append(
                dict(episode=eid, source_directory=str(ep), num_frames=len(files))
            )
        m = dict(
            task=task,
            dataset_skip_frame=1,
            load_skip_frame=3,
            split_policy=policy,
            **entries,
        )
        if task == "cloth" and policy != "custom":
            m["excluded_train_ids"] = list(range(528, 558))
        write(dest / "manifest.json", m)
        write(dest / "metadata.json", meta)
        write(dest / "pgnd_config.json", cfg)
        write(
            dest / "paths.json",
            dict(
                raw_source=str(raw),
                native_state=str(native),
                pgnd_root=str(Path(a.pgnd_root).resolve()),
            ),
        )
        print("[PREPARED]", dest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="dataset", required=True)
    s = sub.add_parser("phystwin")
    s.add_argument("--data-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--groups", nargs="+", default=["all"])
    s.add_argument("--split-file", help="JSON custom scene groups")
    s.add_argument("--train-scenes", nargs="+")
    s.add_argument("--test-scenes", nargs="+")
    s.add_argument("--validation-scenes", nargs="+")
    s.add_argument("--group-name", default="custom")
    s.add_argument("--generalization", choices=["scene", "action"], default="scene")
    s.add_argument("--source", choices=["main", "additional"], default="main")
    s = sub.add_parser("pgnd")
    s.add_argument("--pgnd-root", required=True)
    s.add_argument("--paths", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tasks", nargs="+", default=["all"])
    s.add_argument("--preprocess", action="store_true")
    s.add_argument("--device", default="cpu")
    a = p.parse_args()
    globals()[a.dataset](a)


if __name__ == "__main__":
    main()
