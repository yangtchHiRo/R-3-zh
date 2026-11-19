"""Evaluate directed scene transfers using completed source training runs."""

import argparse
import subprocess
import sys
from pathlib import Path
from rrf.config import CONFIG_DIR
from rrf.io import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument(
        "--models",
        required=True,
        help="JSON mapping source scenes to training run directories",
    )
    parser.add_argument("--pairs", default=str(CONFIG_DIR / "transfer_example.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-render", action="store_true")
    args = parser.parse_args()
    pairs, models = read(args.pairs), read(args.models)
    if not pairs:
        parser.error("Provide at least one source/target pair")
    summaries = []
    for pair in pairs:
        source, target = pair["source"], pair["target"]
        output = Path(args.out) / (source + "__to__" + target)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rrf.evaluate",
                "--dataset",
                "phystwin",
                "--data",
                str(Path(args.prepared) / target),
                "--scenes",
                target,
                "--run",
                models[source],
                "--out",
                str(output),
                "--device",
                args.device,
            ]
            + (["--prepare-render"] if args.prepare_render else []),
            check=True,
        )
        summary = read(output / "summary.json")
        summary.update(source_scene=source, target_scene=target)
        write(output / "summary.json", summary)
        summaries.append(str(output / "summary.json"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rrf.collect",
            "--inputs",
            *summaries,
            "--out",
            str(Path(args.out) / "tables"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
