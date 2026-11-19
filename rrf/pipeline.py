"""Train RRF, evaluate the trained model, and collect results."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from .config import add_config_arguments, resolve_args
from .io import read, write

STAGES = ("train", "evaluate", "collect")


def planned_commands(
    config,
    data,
    out,
    device,
    threads,
    references=None,
    prepare_render=False,
    checkpoint=None,
    eval_data=None,
    eval_partition=None,
):
    data, out = Path(data).resolve(), Path(out).resolve()
    eval_data = Path(eval_data).expanduser().resolve() if eval_data else data
    eval_manifest = read(eval_data / "manifest.json")
    if config["dataset"] == "pgnd" and "task" not in eval_manifest:
        raise ValueError(
            "PGND evaluation needs PGND prepared data"
        )
    if config["dataset"] == "phystwin" and "group" not in eval_manifest:
        raise ValueError("PhysTwin evaluation needs PhysTwin prepared data")
    partition = eval_partition or config["evaluation"]["report_partition"]
    if config["dataset"] == "pgnd" and partition != "test":
        raise ValueError("PGND evaluation uses the test episode list")
    checkpoint = (
        Path(checkpoint).expanduser().resolve() if checkpoint else out / "model"
    )
    if references and config["dataset"] != "pgnd":
        raise ValueError("--references applies only to PGND evaluation")
    if prepare_render and config["dataset"] != "phystwin":
        raise ValueError("Use the separate PGND asset resolver for rendering")
    prefix = [sys.executable, "-m"]
    commands = {
        "train": [
            prefix
            + [
                "rrf.train",
                "--config",
                str(out / "task.json"),
                "--data",
                str(data),
                "--out",
                str(out / "model"),
                "--device",
                device,
                "--threads",
                str(threads),
            ]
        ]
    }
    model_argument = "--checkpoint" if checkpoint.suffix == ".pt" else "--run"
    evaluation = prefix + [
        "rrf.evaluate",
        "--dataset",
        config["dataset"],
        "--data",
        str(eval_data),
        model_argument,
        str(checkpoint),
        "--out",
        str(out / "eval"),
        "--device",
        device,
        "--threads",
        str(threads),
        "--partition",
        partition,
    ]
    if references:
        evaluation += ["--references", str(Path(references).resolve())]
    if prepare_render:
        evaluation.append("--prepare-render")
    commands["evaluate"] = [evaluation]
    commands["collect"] = [
        prefix
        + [
            "rrf.collect",
            "--inputs",
            str(out / "eval/summary.json"),
            "--out",
            str(out / "tables"),
        ]
    ]
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--eval-data", help="Separate prepared evaluation dataset; defaults to --data"
    )
    parser.add_argument(
        "--eval-partition", choices=["all", "train", "validation", "test"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument(
        "--checkpoint", help="Existing model file or completed training run"
    )
    parser.add_argument("--references")
    parser.add_argument("--prepare-render", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoint and "train" in args.stages:
        parser.error("Use --stages evaluate collect with an existing --checkpoint")
    config = resolve_args(args)
    commands = planned_commands(
        config,
        args.data,
        args.out,
        args.device,
        args.threads,
        args.references,
        args.prepare_render,
        args.checkpoint,
        args.eval_data,
        args.eval_partition,
    )
    out = Path(args.out).resolve()
    if not args.dry_run:
        if (out / "task.json").exists() and read(out / "task.json") != config:
            raise ValueError(
                "Output has a different task configuration; choose a new --out"
            )
        write(out / "task.json", config)
    for stage in STAGES:
        if stage not in args.stages:
            continue
        for command in commands[stage]:
            print(f"[{stage}] {shlex.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
