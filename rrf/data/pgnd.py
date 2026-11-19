"""Disk-backed PGND training windows with episode-local particle identities."""

from __future__ import annotations
import bisect
from collections import OrderedDict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from ..io import read
from .pgnd_config import validate_manifest
from .pgnd_prepare import frame_files, load_pt


def encode_window(xs, grips, ids, dt=0.1):
    """Eight model frames: three input frames and five targets, fixed IDs."""
    from kornia.geometry.conversions import (
        quaternion_to_rotation_matrix,
        rotation_matrix_to_axis_angle,
    )

    if xs.shape[0] != 8 or grips.shape[0] != 8:
        raise ValueError("Expected full 3+5 frame window")
    x = xs[:, ids].float()
    g = grips.float()
    actions = g[2:7].clone()
    nxt = g[3:8].clone()
    actions[..., 3:6] = (nxt[..., :3] - actions[..., :3]) / dt
    r0 = quaternion_to_rotation_matrix(actions[..., 6:10].reshape(-1, 4))
    r1 = quaternion_to_rotation_matrix(nxt[..., 6:10].reshape(-1, 4))
    actions[..., 10:13] = (
        rotation_matrix_to_axis_angle(r0.bmm(r1.transpose(1, 2))).reshape(
            5, len(g[0]), 3
        )
        / dt
    )
    pos = torch.cat((x[:3], g[:3, :, :3]), dim=1).permute(1, 0, 2).contiguous()
    typ = torch.cat(
        (
            torch.zeros(len(ids), dtype=torch.long),
            torch.ones(g.shape[1], dtype=torch.long),
        )
    )
    return dict(
        position_sequence=pos,
        node_type=typ,
        actions=actions,
        next_grippers=nxt,
        gt_x=x[3:8],
        gt_v=(x[3:8] - x[2:7]) / dt,
    )


class NativeWindowDataset(Dataset):
    def __init__(self, cache, frame_cache_size=24, native_state_root=None):
        alias = Path(cache).absolute()
        self.root = alias.resolve()
        self.meta = read(self.root / "metadata.json")
        self.manifest = read(self.root / "manifest.json")
        task = self.meta["task"]
        validate_manifest(self.manifest, self.meta, task)
        self.entries = []
        self.cumulative = []
        self.frames = []
        total = 0
        for e in self.manifest["train"]:
            source = e.get("source_directory")
            if source is None:
                if native_state_root is None:
                    raise ValueError(
                        "Old manifest needs native_state_root or six_dataset_registry.json"
                    )
                source = Path(native_state_root) / f'episode_{e["episode"]:04d}'
            p = Path(source).resolve()
            if p.name != f'episode_{e["episode"]:04d}':
                raise ValueError("Source changes episode identity")
            files = frame_files(p)[:: self.manifest["load_skip_frame"]]
            if len(files) != e["num_frames"]:
                raise ValueError("Native frame count changed: " + str(p))
            windows = max(0, len(files) - 7)
            if not windows:
                continue
            total += windows
            self.entries.append(e)
            self.frames.append(files)
            self.cumulative.append(total)
        if not total:
            raise ValueError("No complete Full-5 windows")
        self._cache = OrderedDict()
        self.frame_cache_size = frame_cache_size
        self.audit = dict(
            task=task,
            actually_loaded_train_ids=[e["episode"] for e in self.entries],
            loaded_test_ids=[],
            windows=total,
            particle_sampling="randperm without replacement per window",
            action_convention="current pose and next-minus-current interval velocity",
            native_dtype="float32",
        )

    def __len__(self):
        return self.cumulative[-1]

    def _load(self, p):
        if p in self._cache:
            value = self._cache.pop(p)
        else:
            value = load_pt(p)
        self._cache[p] = value
        while len(self._cache) > self.frame_cache_size:
            self._cache.popitem(last=False)
        if (
            value["x"].dtype != torch.float32
            or value["grippers"].dtype != torch.float32
        ):
            raise ValueError("Full-5 requires FP32 native states")
        return value

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        ep = bisect.bisect_right(self.cumulative, i)
        start = i - (self.cumulative[ep - 1] if ep else 0)
        ss = [self._load(p) for p in self.frames[ep][start : start + 8]]
        n = self.meta["n_particles"]
        available = len(ss[0]["x"])
        if available < n:
            raise ValueError("Too few particles; refusing silent smaller batch")
        ids = torch.randperm(available)[:n]
        return encode_window(
            torch.stack([s["x"] for s in ss]),
            torch.stack([s["grippers"] for s in ss]),
            ids,
        )
