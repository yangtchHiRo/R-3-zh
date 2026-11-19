from pathlib import Path
import shlex, re, os
from ..io import read, write, sha, digest

SPECS = {
    "rope": (651, 691, 1000, 1, 1.0),
    "bread": (143, 163, 500, 2, 3.0),
    "sloth": (113, 133, 1000, 1, 1.0),
    "box": (303, 323, 1000, 2, 1.0),
    "cloth": (610, 650, 1000, 2, 0.8),
    "paperbag": (200, 220, 1000, 1, 1.0),
}
NEW = ("box", "cloth", "paperbag")
CLOTH_CLEAN_POLICY = "cloth_train_dedup_v1"
CLOTH_DUPLICATE_PAIRS = {i: i + 82 for i in range(528, 558)}


def split_ids(task, policy="official"):
    end, total, *_ = SPECS[task]
    if policy == "official":
        return list(range(end)), list(range(end, total))
    if task == "cloth" and policy == CLOTH_CLEAN_POLICY:
        return [i for i in range(end) if i not in CLOTH_DUPLICATE_PAIRS], list(
            range(end, total)
        )
    raise ValueError("Unsupported named split policy: " + str(policy))


def validate_manifest(m, meta, task):
    end, total, n, g, _ = SPECS[task]
    if m["task"] != task or meta["task"] != task:
        raise ValueError("Wrong material identity")
    policy = m.get("split_policy", "official")
    if policy == "custom":
        from .splits import episode_ids

        train, test = episode_ids(
            [e["episode"] for e in m["train"]], [e["episode"] for e in m["eval"]]
        )
        if task == "cloth" and any(
            (a in train and b in test) or (b in train and a in test)
            for a, b in CLOTH_DUPLICATE_PAIRS.items()
        ):
            raise ValueError("Cloth split crosses known duplicate episode pairs")
    else:
        train, test = split_ids(task, policy)
    if [e["episode"] for e in m["train"]] != train:
        raise ValueError("Wrong TRAIN split: " + task)
    if [e["episode"] for e in m["eval"]] != test:
        raise ValueError("Wrong TEST split: " + task)
    if policy != "official":
        if policy != "custom" and m.get("excluded_train_ids") != sorted(
            CLOTH_DUPLICATE_PAIRS
        ):
            raise ValueError("Wrong dedup exclusions")
        if (
            meta.get("normalization_train_ids") != train
            or meta.get("split_policy") != policy
        ):
            raise ValueError("Clean split normalization mismatch")
    if meta["n_particles"] != n or meta["num_grippers"] != g:
        raise ValueError("Wrong particle/gripper count: " + task)
    if m["load_skip_frame"] != 3 or m["dataset_skip_frame"] != 1:
        raise ValueError("Unsupported frame stride")
    if meta["load_skip_frame"] != 3 or meta["dataset_skip_frame"] != 1:
        raise ValueError("Metadata stride mismatch")


def official_overrides(script):
    tokens = shlex.split(Path(script).read_text().replace("\\\n", " "), comments=True)
    return [
        t for t in tokens if t.startswith(("train.", "sim.", "model.")) and "=" in t
    ]


def make_config(pgnd, task, source, dataset):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    pgnd = Path(pgnd).resolve()
    script = pgnd / "experiments/scripts" / f"train_{task}.sh"
    overrides = official_overrides(script)
    with initialize_config_dir(
        config_dir=str(pgnd / "experiments/cfg"), version_base=None
    ):
        cfg = compose(config_name="default", overrides=overrides)
    end, total, n, g, scale = SPECS[task]
    actual = (
        int(cfg.train.training_start_episode),
        int(cfg.train.training_end_episode),
        int(cfg.train.eval_start_episode),
        int(cfg.train.eval_end_episode),
        int(cfg.sim.n_particles),
        int(cfg.sim.num_grippers),
        float(cfg.sim.preprocess_scale),
    )
    if actual != (0, end, end, total, n, g, scale):
        raise ValueError(
            f"Local official configuration differs from reviewed {task} profile: {actual}"
        )
    if (
        int(cfg.train.dataset_load_skip_frame),
        int(cfg.train.dataset_skip_frame),
        int(cfg.sim.n_history),
        int(cfg.sim.num_steps_train),
        float(cfg.sim.dt),
        int(cfg.sim.interval),
    ) != (3, 1, 2, 5, 0.1, 1):
        raise ValueError(
            "Unsupported local time/history configuration; inspect before conversion"
        )
    if task == "box" and (not cfg.sim.gripper_points or cfg.sim.gripper_forcing):
        raise ValueError("Box official interaction flags differ")
    cfg.train.source_dataset_name = str(Path(source).resolve())
    cfg.train.dataset_name = str(Path(dataset).resolve())
    with open_dict(cfg.train):
        cfg.train.dataset_non_overwrite = True
    return OmegaConf.to_container(cfg, resolve=True)
