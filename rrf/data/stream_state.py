import numpy as np
import torch
from .phystwin import window_info, choose_ids


def tree(x, device=None):
    if torch.is_tensor(x):
        return x.detach().to(device) if device else x.detach()
    if isinstance(x, dict):
        return {k: tree(v, device) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(tree(v, device) for v in x)
    if isinstance(x, list):
        return [tree(v, device) for v in x]
    return x


class Stream:
    def __init__(self, case, seed, state=None):
        self.case = case
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.completed = 0
        if state:
            self.rng.bit_generator.state = state["rng"]
            self.state = state["stream"]
            self.completed = state["completed"]

    def reset(self):
        c = self.case
        _, good, labels = window_info(c.x, c.vis, c.motion, 2, c.b, 5)
        pool = np.flatnonzero(good)
        n = min(c.plan["particles"], len(pool))
        if n < c.plan["min_particles"]:
            raise ValueError("Too few true H3 points for continuous forcing")
        ids = choose_ids(pool, labels, n, self.rng)
        seq = np.concatenate((c.x[:3, ids], c.ctrl[:3]), 1).transpose(1, 0, 2).copy()
        self.state = dict(
            ids=ids,
            t=2,
            sequence=torch.from_numpy(seq)[None],
            memory=None,
            chain=np.ones(n, bool),
            blocks=0,
        )

    def batch(self):
        if self.state is None or self.state["t"] >= self.case.b - 1:
            if self.state is not None:
                self.completed += 1
            self.reset()
        s = self.state
        c = self.case
        ids = s["ids"]
        t = s["t"]
        H = min(5, c.b - 1 - t)
        masks = []
        chain = s["chain"].copy()
        for q in range(1, H + 1):
            chain &= c.motion[t + q - 1, ids]
            masks.append(
                chain & c.vis[t + q, ids] & np.isfinite(c.x[t + q, ids]).all(-1)
            )
        valid = np.stack(masks)
        gt = c.x[t + 1 : t + H + 1, ids].copy()
        gt[~valid] = 0
        ref = c.x[0, ids].copy()
        rv = c.vis[0, ids] & np.isfinite(ref).all(-1)
        ref[~rv] = 0
        row = dict(
            particle_ids=ids,
            gt_x=gt,
            valid=valid,
            actions=c.g[t : t + H].copy(),
            next_grippers=c.g[t + 1 : t + H + 1].copy(),
            initial_ref=ref,
            reference_valid=rv,
            initial_ctrl_ref=c.ctrl[0].copy(),
            history_seed_observed=np.asarray(s["blocks"] == 0),
            **{k: v.copy() for k, v in c.patch_reference.items()}
        )
        b = {k: torch.from_numpy(np.asarray(v).copy())[None] for k, v in row.items()}
        b["position_sequence"] = s["sequence"]
        self.pending = (H, chain)
        return (
            b,
            s["memory"],
            dict(
                stream_t=t,
                stream_age=t - 2,
                stream_blocks=s["blocks"],
                completed_streams=self.completed,
                stream_horizon=H,
                stream_valid_per_frame=valid.sum(-1).tolist(),
            ),
        )

    def advance(self, result):
        H, chain = self.pending
        s = self.state
        c = self.case
        t = s["t"]
        seq = s["sequence"].cpu()
        pred = result["pred"].detach().cpu()
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite continuous forcing state")
        for q in range(H):
            ctrl = torch.from_numpy(c.ctrl[t + q + 1].copy())[None]
            nxt = torch.cat((pred[:, q], ctrl), 1)
            seq = torch.cat((seq[:, :, 1:], nxt[:, :, None]), 2)
        s.update(
            sequence=seq,
            memory=tree(result.get("memory"), "cpu"),
            chain=chain,
            t=t + H,
            blocks=s["blocks"] + 1,
        )

    def snapshot(self):
        return dict(
            rng=self.rng.bit_generator.state,
            stream=tree(self.state, "cpu"),
            completed=self.completed,
        )


def masked_loss(pred, target, valid):
    # Exactly original position loss when every frame has labels. Empty frames
    # contribute zero and remain logged; never invent correspondence/targets.
    err = (pred - target).square().mean(-1)
    w = valid.to(err.dtype)
    return ((err * w).sum(-1) / w.sum(-1).clamp_min(1)).mean(0).sum()
