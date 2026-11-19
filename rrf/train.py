"""One configuration-driven trainer for all RRF tasks."""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .config import add_config_arguments, resolve_args
from .data.training import build_training_data
from .dynamics import masked_rollout_loss
from .io import write, seed_all, atomic_torch, digest, sha
from .model import build_model, model_config
from .schedules import build_optimizer, learning_rate


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def run(args, experiment=None):
    exp = experiment if experiment is not None else resolve_args(args)
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop-after must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(exp["seed"])
    source = build_training_data(args.data, exp, device)
    config = model_config(source.metadata, exp)
    config.update(source.model_extras)
    config.update(
        dataset=exp["dataset"],
        steps=exp["train"]["steps"],
        seed=exp["seed"],
        initialization="scratch",
    )
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, exp)
    identity = digest(dict(experiment=exp, model=config, data=source.provenance))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    resume = out / "resume.pt"
    start = 1
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError(
                "Resume configuration/data mismatch; use a new output directory."
            )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        source.load_state_dict(saved["data_state"])
        # Reconstructing a loader can consume RNG; restore global state afterwards.
        restore_rng(saved["rng"])
        start = saved["step"] + 1
    elif (out / "config.json").exists() or list(out.glob("step_*.pt")):
        raise ValueError("Run exists without a resume checkpoint; use a new output")
    write(out / "resolved_config.json", exp)
    write(
        out / "config.json",
        dict(
            config=config,
            experiment=exp,
            identity=identity,
            data_provenance=source.provenance,
        ),
    )
    t = exp["train"]
    stop = min(t["steps"], args.stop_after) if args.stop_after else t["steps"]
    checkpoint_interval = 1 if args.smoke else t["checkpoint_interval"]
    resume_interval = 1 if args.smoke else t["resume_interval"]
    if args.smoke:
        stop = min(stop, start + 1)
    logfile = out / "train.jsonl"
    if logfile.exists():
        committed = []
        for line in logfile.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["step"] < start:
                committed.append(line)
        logfile.write_text("".join(line + "\n" for line in committed))
    for step in range(start, stop + 1):
        tick = time.perf_counter()
        model.train()
        lr = learning_rate(step, exp)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        kinds = []
        for stream in source.streams:
            raw, _, stats = stream.batch()
            batch = {key: value.to(device) for key, value in raw.items()}
            loss, result = masked_rollout_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            (loss / len(source.streams)).backward()
            stream.advance(result)
            loss_value += float(loss.detach()) / len(source.streams)
            kinds.append(stats["input_kind"])
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), t["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        if step <= 10 or step % t["log_interval"] == 0 or step == stop:
            row = dict(
                step=step,
                loss=loss_value,
                lr=lr,
                grad_norm=float(grad),
                seconds=time.perf_counter() - tick,
                horizon=5,
                input_kinds=kinds,
            )
            with logfile.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        save_resume = step % resume_interval == 0 or step == stop
        save_snapshot = step % checkpoint_interval == 0 or step == stop
        if save_resume or save_snapshot:
            saved = dict(
                step=step,
                config=config,
                model_config=config,
                experiment_config=exp,
                identity=identity,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                data_state=source.state_dict(),
            )
            if save_resume:
                atomic_torch(saved, resume)
            if save_snapshot:
                atomic_torch(saved, out / f"step_{step:07d}.pt")
    if stop == t["steps"]:
        final_state = torch.load(resume, map_location="cpu", weights_only=False)
        if final_state["step"] != t["steps"] or final_state["identity"] != identity:
            raise ValueError("Final training state is incomplete or inconsistent")
        atomic_torch(final_state, out / "final.pt")
        write(
            out / "complete.json",
            dict(
                step=stop,
                identity=identity,
                checkpoint="final.pt",
                checkpoint_sha256=sha(out / "final.pt"),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--data", help="Prepared data directory")
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    exp = resolve_args(args)
    if args.print_config:
        print(json.dumps(exp, indent=2))
        return
    if not args.data or not args.out:
        parser.error("--data and --out are required for training")
    run(args, exp)


if __name__ == "__main__":
    main()
