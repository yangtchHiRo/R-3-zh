"""Use the installed PGND metric and aggregation code without importing render stacks."""

import ast
import copy
from pathlib import Path
import numpy as np
from .external import extract_functions
from ..io import write as _write


def write_json(path, value):
    return _write(path, value)


AGGREGATE_NAMES = {
    "mean_metric",
    "std_metric",
    "n_steps",
    "mean_metric_step",
    "std_metric_step",
}


def aggregate_program(pgnd_root):
    path = Path(pgnd_root) / "experiments/train/eval.py"
    tree = ast.parse(path.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    statements = [
        copy.deepcopy(n)
        for n in fn.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in AGGREGATE_NAMES
    ]
    if {n.targets[0].id for n in statements} != AGGREGATE_NAMES:
        raise ValueError(
            "Unsupported official aggregate code: expected mean_metric/std_metric/n_steps and endpoint selection"
        )
    program = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    return program, path


def official_aggregate(values, pgnd_root):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError(
            "Official aggregate requires [episodes, future_steps, 3] finite values"
        )
    program, path = aggregate_program(pgnd_root)
    ns = {"np": np, "metrics_list": values}
    exec(compile(program, str(path), "exec"), ns)
    index = int(ns["n_steps"])
    if index < 0 or index >= values.shape[1]:
        raise ValueError("Official reporting index is outside the evaluated horizon")
    return (
        np.asarray(ns["mean_metric_step"]),
        np.asarray(ns["std_metric_step"]),
        index,
        ast.unparse(program),
    )


def split_for_task(pgnd_root, cfg, task):
    tree = ast.parse((Path(pgnd_root) / "experiments/train/eval.py").read_text())
    entry = next(
        n
        for n in tree.body
        if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
    )
    values = {}
    for n in ast.walk(entry):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            target = ast.unparse(n.targets[0])
            if target in ("cfg.start_episode", "cfg.end_episode"):
                values[target] = ast.unparse(n.value)
    expected = {
        "cfg.start_episode": "cfg.train.eval_start_episode",
        "cfg.end_episode": "cfg.train.eval_end_episode",
    }
    if values == expected:
        return (
            int(cfg["train"]["eval_start_episode"]),
            int(cfg["train"]["eval_end_episode"]),
            "hydra eval split via official entry",
        )
    if values == {
        "cfg.start_episode": "best_models[args.task][3][0]",
        "cfg.end_episode": "best_models[args.task][3][1]",
    }:
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "best_models" for t in n.targets
            ):
                data = ast.literal_eval(n.value)[task]
                return (
                    int(data[3][0]),
                    int(data[3][1]),
                    "official best_models task split",
                )
    raise ValueError(
        f"Unsupported official split selection: {values}; refusing a guessed episode subset"
    )


def state_only_metric_program(path):
    tree = ast.parse(Path(path).read_text())
    fn = copy.deepcopy(
        next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "do_metric"
        )
    )
    fn.decorator_list = []
    # The original function unconditionally allocates an image LPIPS model even
    # with use_gs=False. This dead image-only initialization has no 3D metric role.
    fn.body = [
        n
        for n in fn.body
        if not (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "loss_fn_vgg" for t in n.targets
            )
        )
    ]
    return ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                fn,
            ],
            type_ignores=[],
        )
    )


def metric_state_layout(path):
    """Recognize the two inspected local PGND scoring timelines explicitly."""
    tree = ast.parse(Path(path).read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "do_metric"
    )
    assignments = {
        ast.unparse(n.targets[0]): ast.unparse(n.value)
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
    }
    loops = {
        ast.unparse(n.iter)
        for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "i"
    }
    gt = assignments.get("gt_traj", "")
    if (
        gt == "xyz_orig[(cfg.sim.n_history + 1) * skip_frame::skip_frame]"
        and "range(len(traj))" in loops
    ):
        return "future_only"
    if (
        gt
        == "xyz_orig[cfg.sim.n_history * skip_frame::cfg.sim.skip_frame * skip_frame]"
        and "range(1, len(traj))" in loops
    ):
        return "initial_plus_future"
    raise ValueError(
        f"Unsupported local official metric timeline: gt_traj={gt}; loops={loops}"
    )


def run_official_metrics(
    pgnd_root, cfg, episode, indices, x0, pred, job, method, max_cost_gib
):
    import torch, scipy, json, os, math
    from omegaconf import OmegaConf

    pgnd = Path(pgnd_root).resolve()
    base = Path(job) / "official_outputs" / method
    cc = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cc.train.name = str(base.resolve())
    state = (
        base
        / f"official_eval/shared/{int(cfg.iteration):06d}/state/episode_{episode:04d}"
    )
    state.mkdir(parents=True, exist_ok=True)
    # Match the ACTUAL local reader. A future-only implementation
    # already starts GT at n_history+1 and scores every file starting at 0000.
    layout = metric_state_layout(pgnd / "experiments/train/metric_eval.py")
    states = pred if layout == "future_only" else torch.cat([x0[None], pred], 0)
    for stale in state.glob("*.pt"):
        stale.unlink()
    for h, xyz in enumerate(states):
        torch.save({"x": xyz.detach().cpu()}, state / f"{h:04d}.pt")
    namespace = {
        "torch": torch,
        "np": np,
        "scipy": scipy,
        "json": json,
        "os": os,
        "math": math,
        "Path": Path,
    }
    extract_functions(
        pgnd / "experiments/train/metric.py",
        ["inverse_preprocess", "mean_dist", "chamfer_dist", "em_distance"],
        namespace,
    )
    native_emd = namespace["em_distance"]

    def guarded_emd(x, y):
        need = 8 * len(x) * len(y) / 1024**3
        if need > max_cost_gib:
            raise MemoryError(
                f"Official EMD cost needs {need:.2f} GiB; raise MAX_COST_GIB if RAM permits. No subsampling."
            )
        return native_emd(x, y)

    namespace["em_distance"] = guarded_emd
    path = pgnd / "experiments/train/metric_eval.py"
    program = state_only_metric_program(path)
    source = ast.unparse(program) + "\n"
    executable = Path(job) / "executed_official_metric.py"
    executable.write_text(source)
    exec(compile(source, str(executable.resolve()), "exec"), namespace)
    values = namespace["do_metric"](
        cc,
        pgnd / "experiments/log",
        int(cfg.iteration),
        [f"episode_{episode:04d}"],
        indices[None].clone(),
        "official_eval",
        dataset_name="shared",
        use_gs=False,
    )
    values = np.asarray(values)
    if values.shape != (1, len(pred), 3) or not np.isfinite(values).all():
        raise ValueError(
            f"Unexpected official metric output for {method}: {values.shape}"
        )
    # Keep native metric.txt and downsample_indices.npy; disk states are derived
    # from the saved prediction and can be discarded after successful scoring.
    import shutil

    # STAGE51_METRIC_CLEANUP_FIX_V1: an absent derived directory is already clean.
    try:
        shutil.rmtree(state)
    except FileNotFoundError:
        pass
    return values[0]
