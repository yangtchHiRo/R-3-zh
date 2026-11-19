"""Episode-fixed geometry, differentiable membership and relation priors.

No particle/patch ID embeddings. FPS is only a geometric candidate construction;
all learned parameters receive trajectory-prediction gradients.
"""

import numpy as np
import torch
from torch import nn
from .encoders import history_velocities


def reference_pack(points, count=64):
    x = np.asarray(points, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or (not len(x)) or (not np.isfinite(x).all()):
        raise ValueError("Reference support must contain finite frame-0 observations")
    x = x[np.lexsort((x[:, 2], x[:, 1], x[:, 0]))]
    center = x.mean(0)
    m = min(count, len(x))
    d = np.full(len(x), np.inf)
    chosen = []
    nxt = int(np.argmax(((x - center) ** 2).sum(-1)))
    for _ in range(m):
        chosen.append(nxt)
        d = np.minimum(d, ((x - x[nxt]) ** 2).sum(-1))
        d[chosen] = -1
        nxt = int(d.argmax())
    anchors = x[chosen].copy()
    if m > 1:
        dist = np.linalg.norm(anchors[:, None] - anchors[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        spacing = np.maximum(dist.min(-1), 0.0001)
    else:
        spacing = np.array(
            [max(float(np.linalg.norm(x - center, axis=-1).max()), 0.0001)], np.float32
        )
    return dict(
        patch_anchors=anchors,
        reference_center=center.astype("f"),
        patch_spacing=spacing.astype("f"),
    )


def descriptor(x, center, radius):
    r = (x - center[:, None]) / radius
    f = r[..., None] * r.new_tensor([1.0, 2.0, 4.0, 8.0])
    return torch.cat(
        (
            r,
            r.square(),
            r.norm(dim=-1, keepdim=True),
            f.sin().flatten(-2),
            f.cos().flatten(-2),
            torch.ones_like(r[..., :1]),
        ),
        -1,
    )


def gather_nodes(x, idx):
    return x[torch.arange(x.shape[0], device=x.device)[:, None, None], idx]


def pool_sparse(values, idx, weight, m):
    """[B,N,K,D] -> [B,M,D], normalized weighted sum; no N*M*D tensor."""
    b, n, k, d = values.shape
    dest = values.new_zeros(b, m, d).scatter_add(
        1,
        idx.reshape(b, -1, 1).expand(-1, -1, d),
        (values * weight[..., None]).reshape(b, -1, d),
    )
    mass = weight.new_zeros(b, m).scatter_add(
        1, idx.reshape(b, -1), weight.reshape(b, -1)
    )
    return (dest / mass.clamp_min(1e-08)[..., None], mass)


class Structure(nn.Module):

    def __init__(self, meta, cfg):
        super().__init__()
        self.k = cfg.get("candidates", 8)
        self.radius = float(meta["connectivity_radius"])
        self.geometry = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32))
        self.assignment = nn.Sequential(nn.Linear(68, 64), nn.SiLU(), nn.Linear(64, 1))
        self.relation = nn.Sequential(nn.Linear(68, 64), nn.SiLU(), nn.Linear(64, 4))
        nn.init.zeros_(self.assignment[-1].weight)
        nn.init.zeros_(self.assignment[-1].bias)
        nn.init.zeros_(self.relation[-1].weight)
        nn.init.zeros_(self.relation[-1].bias)

    def forward(self, b):
        anchors = b["patch_anchors"]
        ref = b["initial_ref"]
        rv = b["reference_valid"].bool()
        center = b["reference_center"]
        m = anchors.shape[1]
        ga = descriptor(anchors, center, self.radius)
        gi = descriptor(ref, center, self.radius)
        c = ga + self.geometry(ga)
        ci = gi + self.geometry(gi)
        dist = torch.cdist(ref, anchors)
        idx = dist.topk(min(self.k, m), largest=False).indices
        ap = gather_nodes(anchors, idx)
        cp = gather_nodes(c, idx)
        rel = (ref[:, :, None] - ap) / self.radius
        spacing = (
            b["patch_spacing"]
            .gather(1, idx.flatten(1))
            .reshape_as(idx)
            .clamp_min(0.0001)
        )
        logits = -((ref[:, :, None] - ap).norm(dim=-1) / spacing).square()
        pair = torch.cat(
            (ci[:, :, None].expand_as(cp), cp, rel, rel.norm(dim=-1, keepdim=True)), -1
        )
        logits = logits + 2 * self.assignment(pair).squeeze(-1).tanh()
        weights = logits.softmax(-1) * rv[..., None]
        delta = (anchors[:, None] - anchors[:, :, None]) / self.radius
        distance = delta.norm(dim=-1, keepdim=True)
        a0 = 0.05 + 0.95 * torch.exp(-distance)
        pair = torch.cat(
            (
                c[:, :, None].expand(-1, -1, m, -1),
                c[:, None].expand(-1, m, -1, -1),
                delta,
                distance,
            ),
            -1,
        )
        logit = torch.logit(a0.clamp(0.0001, 1 - 0.0001))
        a = (logit + self.relation(pair)).sigmoid().clamp_min(1e-05)
        mass = weights.new_zeros(weights.shape[0], m).scatter_add(
            1, idx.flatten(1), weights.flatten(1)
        )
        return dict(
            idx=idx,
            weight=weights,
            C=c,
            A=a,
            anchors=anchors,
            mass=mass,
            active=mass > 1e-07,
            reference_valid=rv,
        )


def pooled(x, z):
    k = z["idx"].shape[-1]
    return pool_sparse(
        x[:, :, None].expand(-1, -1, k, -1),
        z["idx"],
        z["weight"],
        z["anchors"].shape[1],
    )[0]


def normalize_pool(values, z, weight=None):
    w = z["weight"] if weight is None else weight
    return pool_sparse(values, z["idx"], w, z["anchors"].shape[1])


def geometric_state(current, z):
    ph = z["anchors"][:, :, None] + pooled(
        (current["hist"] - current["initial_ref"][:, :, None]).flatten(-2), z
    ).reshape(*z["anchors"].shape[:2], 3, 3)
    vv = history_velocities(current["hist"], current["hist_times"])
    pv = pooled(vv.flatten(-2), z).reshape(*z["anchors"].shape[:2], 2, 3)
    return ph, pv
