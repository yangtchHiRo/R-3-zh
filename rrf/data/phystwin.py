from __future__ import annotations
import numpy as np
from ..io import read, sha


def control_states(ctrl, dt):
    g = np.zeros((*ctrl.shape[:2], 15), dtype=np.float32)
    g[..., :3] = ctrl
    g[..., 6] = 1
    g[1:, :, 3:6] = np.diff(ctrl, axis=0) / dt
    return g


def normalize_motion_mask(raw, T, N):
    a = np.asarray(raw)
    if a.shape == (T - 1, N):
        return a.astype(bool)
    if a.shape == (T, N):
        return a[:-1].astype(bool)
    raise ValueError(f"object_motions_valid must be [T-1,N] or [T,N], got {a.shape}")


def window_info(x, vis, motion, t, boundary, horizon=5):
    """Read only training frames through t+horizon < boundary."""
    if horizon < 1 or t < 2 or t + horizon >= boundary:
        raise ValueError("Window crosses train boundary or invalid horizon")
    hi = np.arange(t - 2, t + 1)
    history_ok = (vis[hi] & np.isfinite(x[hi]).all(-1)).all(0)
    if t > 0:
        history_ok &= motion[max(0, t - 2) : t].all(0)
    labels = []
    chain = np.ones(x.shape[1], bool)
    for k in range(1, horizon + 1):
        chain &= motion[t + k - 1]
        labels.append(chain & vis[t + k] & np.isfinite(x[t + k]).all(-1))
    return hi, history_ok, np.stack(labels)


def choose_ids(pool, labels, n, rng, fixed=False):
    """Original sampling, with explicit label coverage for every horizon."""
    mandatory = []
    for label in labels:
        candidates = pool[label[pool]]
        if not len(candidates):
            raise ValueError("A target horizon has no valid points")
        mandatory.append(int(candidates[0] if fixed else rng.choice(candidates)))
    mandatory = np.unique(mandatory)
    if len(mandatory) > n:
        # Small synthetic/user batches can share one point across several labels.
        # Real runs have n>=32 and H<=10, so retain their original sampler exactly.
        uncovered = np.ones(len(labels), bool)
        selected = []
        while uncovered.any() and len(selected) < n:
            coverage = labels[uncovered][:, pool].sum(0)
            if selected:
                coverage[np.isin(pool, selected)] = 0
            if coverage.max() == 0:
                break
            candidates = pool[coverage == coverage.max()]
            point = int(candidates[0] if fixed else rng.choice(candidates))
            selected.append(point)
            uncovered &= ~labels[:, point]
        if uncovered.any():
            raise ValueError(
                "Too few particle slots to cover all target horizons; increase min_particles"
            )
        mandatory = np.asarray(selected, dtype=np.int64)
    if n > len(pool) or n < len(mandatory):
        raise ValueError("Invalid particle count")
    rest = np.setdiff1d(pool, mandatory)
    ids = np.concatenate(
        [mandatory, rng.choice(rest, n - len(mandatory), replace=False)]
    ).astype(np.int64)
    if not labels[:, ids].any(1).all():
        raise AssertionError("Uncovered target horizon")
    return ids


class Case:
    def __init__(self, plan, case_id, verify=True):
        from pathlib import Path

        self.plan = read(plan) if not isinstance(plan, dict) else plan
        self.entry = next(e for e in self.plan["cases"] if e["id"] == case_id)
        p = Path(self.entry["cache"])
        if verify and sha(p) != self.entry["cache_sha256"]:
            raise ValueError("Prepared cache changed: " + str(p))
        with np.load(p, allow_pickle=False) as z:
            self.x = z["x"]
            self.vis = z["vis"]
            self.motion = z["motion"]
            self.ctrl = z["ctrl"]
            self.gt = z["gt"]
        self.b = self.entry["train_end"]
        self.T = len(self.x)
        self.meta = self.entry["metadata"]
        self.g = control_states(self.ctrl, self.entry["dt"])
        self.eval_ids = np.asarray(self.entry["initial_ids"], dtype=np.int64)
        from ..models.regions import reference_pack
        from ..io import fps

        observed = (self.vis[:3] & np.isfinite(self.x[:3]).all(-1)).all(
            0
        ) & self.motion[:2].all(0)
        pool = np.flatnonzero(observed)
        if not len(pool):
            raise ValueError("No persistent particles in real H3")
        self.eval_ids = pool[fps(self.x[0, pool], self.plan["particles"])]
        good = self.vis[0] & np.isfinite(self.x[0]).all(-1)
        self.patch_reference = reference_pack(self.x[0, good], 64)
        self._windows = {}
        self.windows = self.windows_for(5)

    def windows_for(self, horizon):
        if horizon not in self._windows:
            legal = []
            for t in range(2, max(2, self.b - horizon)):
                _, good, labels = window_info(
                    self.x, self.vis, self.motion, t, self.b, horizon
                )
                if (
                    good.sum() >= self.plan["min_particles"]
                    and labels[:, good].any(1).all()
                ):
                    legal.append(t)
            self._windows[horizon] = legal
        return self._windows[horizon]

    def sample(self, t, ids, horizon):
        hi, good, labels = window_info(
            self.x, self.vis, self.motion, t, self.b, horizon
        )
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or not good[ids].all():
            raise ValueError("Invalid persistent input IDs")
        valid = labels[:, ids]
        if not valid.any(1).all():
            raise ValueError("Every horizon needs a valid label")
        pos = np.concatenate([self.x[hi][:, ids], self.ctrl[hi]], axis=1).transpose(
            1, 0, 2
        )
        target = self.x[t + 1 : t + horizon + 1, ids].copy()
        target[~valid] = 0
        typ = np.r_[
            np.zeros(len(ids), dtype=np.int64),
            np.ones(len(self.ctrl[0]), dtype=np.int64),
        ]
        return dict(
            position_sequence=pos.copy(),
            node_type=typ,
            gt_x=target,
            valid=valid.copy(),
            actions=self.g[t : t + horizon].copy(),
            next_grippers=self.g[t + 1 : t + horizon + 1].copy(),
        )

    @staticmethod
    def stack(rows):
        import torch

        return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}

    def batch_at(self, times, rng, horizon=5, fixed=False):
        info = [
            window_info(self.x, self.vis, self.motion, t, self.b, horizon)
            for t in times
        ]
        n = min(self.plan["particles"], *(int(g.sum()) for _, g, _ in info))
        rows = []
        for t, (_, good, labels) in zip(times, info):
            local_rng = np.random.default_rng(0) if fixed else rng
            ids = choose_ids(np.flatnonzero(good), labels, n, local_rng, fixed)
            rows.append(self.sample(t, ids, horizon))
        return self.stack(rows)

    def batch(self, rng, fixed=False, horizon=5):
        windows = self.windows_for(horizon)
        if not windows:
            raise ValueError(f"No valid Full-{horizon} windows")
        times = (
            [windows[0]] * 2 if fixed else [int(rng.choice(windows)) for _ in range(2)]
        )
        return self.batch_at(times, rng, horizon, fixed)

    def preflight_batch(self, horizon=10):
        windows = self.windows_for(horizon)
        if not windows:
            raise ValueError(f"No valid Full-{horizon} windows")
        t = max(
            windows,
            key=lambda u: min(
                self.plan["particles"],
                int(
                    window_info(self.x, self.vis, self.motion, u, self.b, horizon)[
                        1
                    ].sum()
                ),
            ),
        )
        return self.batch_at([t, t], np.random.default_rng(781923), horizon), t
