"""Shared SI-unit encoders.  No forward method reads future object labels."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def statistic(meta, key, default=1.0):
    norm = meta.get("normalization", {})
    value = norm.get(key, meta.get(key, default))
    return torch.as_tensor(value, dtype=torch.float32)


def physical_history(hist, times):
    """hist [B,N,3,3]; times [B,3] seconds -> x, m/s, m/s²."""
    dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-7)
    velocities = (hist[:, :, 1:] - hist[:, :, :-1]) / dt[:, None, :, None]
    acceleration = (velocities[:, :, 1] - velocities[:, :, 0]) / (
        (dt[:, 0] + dt[:, 1]) * 0.5
    )[:, None, None]
    return hist[:, :, -1], velocities[:, :, -1], acceleration


def history_velocities(hist, times):
    dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-7)
    return (hist[:, :, 1:] - hist[:, :, :-1]) / dt[:, None, :, None]


def controller_at(batch, tau):
    """Last available controller sample at/before each query, no suffix reads."""
    path, times = batch["ctrl_path"], batch["path_times"]
    valid = batch.get("path_mask", torch.ones_like(times, dtype=torch.bool))
    allowed = valid & (times <= tau[:, None] + 1e-7)
    index = (
        torch.where(
            allowed, torch.arange(times.shape[1], device=times.device)[None], -1
        )
        .amax(1)
        .clamp_min(0)
    )
    return path[torch.arange(len(path), device=path.device), index]


def available_control(batch, meta, value):
    available = batch.get(
        "ctrl_available",
        meta.get("control_availability_mask", meta.get("ctrl_available")),
    )
    if available is None:
        return value
    available = torch.as_tensor(available, dtype=value.dtype, device=value.device)
    if available.ndim == 1:
        available = available[None]
    while available.ndim < value.ndim:
        available = available.unsqueeze(1)
    return value * available


class Block(nn.Module):
    """The original PF pre-LN 4-head/FFN block, with optional pair bias."""

    def __init__(self, d=128, heads=4, ffn=512):
        super().__init__()
        if d % heads:
            raise ValueError("width must be divisible by attention heads")
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, bias=None, modulation=None):
        b, n, d = x.shape
        normalized = self.norm1(x)
        if modulation is not None:
            shift1, scale1, shift2, scale2 = modulation.chunk(4, -1)
            normalized = normalized * (1 + scale1[:, None]) + shift1[:, None]
        q, k, v = (
            self.qkv(normalized)
            .reshape(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, d))
        normalized = self.norm2(x)
        if modulation is not None:
            normalized = normalized * (1 + scale2[:, None]) + shift2[:, None]
        return x + self.ffn(normalized)


class TypedEncoder(nn.Module):
    """Rich observed-state encoder with shared versus separate role networks.

    Returns z_object [B,N,D], z_controller [B,C,D], auxiliary observed SI state.
    The optional tau changes prescribed control displacement, never object labels.
    """

    def __init__(self, meta, variant, width=128):
        super().__init__()
        self.meta, self.variant = meta, variant
        self.scale = float(meta.get("scale", 1.0))
        self.radius = float(meta.get("connectivity_radius", 0.03))

        self.register_buffer(
            "velocity_scale", statistic(meta, "velocity_std", 0.1).clamp_min(1e-4)
        )
        self.register_buffer(
            "acceleration_scale",
            statistic(meta, "acceleration_std", 1.0).clamp_min(1e-3),
        )
        self.role = nn.Embedding(2, 16)
        dim = 3 + 6 + 3 + 3 + 3 + 3 + 9 + 3 + 16
        network = lambda: nn.Sequential(
            nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, width)
        )
        self.object_encoder = network()
        self.controller_encoder = (
            network()
            if variant.get("encoder", "shared") in ("typed", "separate")
            else None
        )

    def forward(self, batch, tau=None):
        hist, times = batch["hist"], batch["hist_times"]
        ch = batch["ctrl_hist"]
        obj, velocity, acc = physical_history(hist, times)
        control, cv, ca = physical_history(ch[..., :3], times)
        all_x = torch.cat((obj, control), 1)
        hv = torch.cat(
            (history_velocities(hist, times), history_velocities(ch[..., :3], times)), 1
        )
        all_acc = torch.cat((acc, ca), 1)
        initial = batch.get("initial_ref", hist[:, :, 0])
        initial_ctrl = batch.get("initial_ctrl_ref", ch[:, :, 0, :3])
        all_ref = torch.cat((initial, initial_ctrl), 1)
        b, n, _ = obj.shape
        c = control.shape[1]
        rv = batch.get(
            "reference_valid", torch.ones(b, n, device=obj.device, dtype=torch.bool)
        )
        center = (initial * rv[..., None]).sum(1, keepdim=True) / rv.sum(
            1, keepdim=True
        ).clamp_min(1)[..., None]
        position = all_x / self.scale
        if self.variant.get("centered", False):
            position = (all_x - center) / self.scale
        relative = (all_ref - center) / self.scale
        if "reference_valid" in batch:
            relative = (
                relative
                * torch.cat(
                    (
                        batch["reference_valid"],
                        torch.ones(b, c, device=obj.device, dtype=torch.bool),
                    ),
                    1,
                )[..., None]
            )
        distances = torch.cdist(obj, control)
        nearest = distances.argmin(-1)
        ctrl_near = control.gather(1, nearest[..., None].expand(-1, -1, 3))
        cv_near = cv.gather(1, nearest[..., None].expand(-1, -1, 3))
        rel = torch.cat((ctrl_near - obj, torch.zeros_like(control)), 1) / self.radius
        relv = (
            torch.cat((cv_near - velocity, torch.zeros_like(control)), 1)
            / self.velocity_scale
        )
        known = available_control(batch, self.meta, ch[:, :, -1])
        aux = torch.cat(
            (
                torch.zeros(b, n, 9, device=obj.device, dtype=obj.dtype),
                known[..., 6:15],
            ),
            1,
        )
        u = torch.zeros_like(all_x)
        if tau is not None:
            endpoint = controller_at(batch, tau)
            u[:, n:] = (endpoint[..., :3] - control) / self.radius
        ids = torch.cat(
            (
                torch.zeros(b, n, device=obj.device, dtype=torch.long),
                torch.ones(b, c, device=obj.device, dtype=torch.long),
            ),
            1,
        )
        features = torch.cat(
            (
                position,
                (hv / self.velocity_scale).flatten(2),
                all_acc / self.acceleration_scale,
                relative,
                rel,
                relv,
                aux,
                u,
                self.role(ids),
            ),
            -1,
        )
        zo = self.object_encoder(features[:, :n])
        zc = (self.controller_encoder or self.object_encoder)(features[:, n:])
        return (
            zo,
            zc,
            {
                "x": obj,
                "velocity": velocity,
                "acceleration": acc,
                "ctrl_x": control,
                "ctrl_velocity": cv,
                "features": features,
            },
        )


class ActionEncoder(nn.Module):
    """Every temporal layer is causal; query pooling also masks its suffix.

    Controller tokens are pooled *within* a timestamp, reducing temporal attention
    to A frames rather than A*C points. output memory [B,Q,A,D], padding [B,Q,A]
    (True means blocked), summary [B,Q,D]. Times are seconds relative to start.
    """

    def __init__(self, width=128, input_dim=15, layers=2, heads=4):
        super().__init__()
        self.width = width
        self.point = nn.Sequential(
            nn.Linear(input_dim + 9, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.pool = nn.Linear(2 * width, width)
        self.time = nn.Linear(16, width)
        self.layers = nn.ModuleList(
            [Block(width, heads, 4 * width) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(width)
        self.register_buffer("frequencies", 2.0 ** torch.arange(8, dtype=torch.float32))

    def encode(self, path, path_times, path_mask=None):
        b, a, c, _ = path.shape
        reference = path[:, :1, :, :3].expand(-1, a, -1, -1)
        displacement = path[..., :3] - reference
        lever = reference - reference.mean(2, keepdim=True)
        motion_moment = torch.cross(lever, path[..., 3:6], dim=-1)
        p = self.point(torch.cat((path, reference, displacement, motion_moment), -1))
        h = self.pool(torch.cat((p.mean(2), p.amax(2)), -1))
        t = path_times[..., None] * self.frequencies
        h = h + self.time(torch.cat((t.sin(), t.cos()), -1))
        valid = (
            torch.ones(b, a, device=path.device, dtype=torch.bool)
            if path_mask is None
            else path_mask.bool()
        )
        causal = (
            torch.arange(a, device=path.device)[None, :]
            <= torch.arange(a, device=path.device)[:, None]
        )
        allowed = causal[None] & valid[:, None, :]
        # At least the initial known action exists. Fail clearly on invalid input.
        if not bool(valid[:, 0].all()):
            raise ValueError("action path must contain a valid initial timestamp")
        bias = torch.zeros(
            b, 1, a, a, device=path.device, dtype=path.dtype
        ).masked_fill(~allowed[:, None], float("-inf"))
        for layer in self.layers:
            h = layer(h, bias)
        return self.norm(h), valid

    def forward(self, path, path_times, path_mask, query_times):
        h, valid = self.encode(path, path_times, path_mask)
        q = query_times.shape[1]
        allowed = valid[:, None, :] & (
            path_times[:, None, :] <= query_times[:, :, None] + 1e-7
        )
        if not bool(allowed.any(-1).all()):
            raise ValueError("query before initial action timestamp")
        summary = (h[:, None] * allowed[..., None]).sum(2) / allowed.sum(2).clamp_min(
            1
        )[..., None]
        return h[:, None].expand(-1, q, -1, -1), ~allowed, summary


class CrossBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, z, memory, padding_mask):
        v, _ = self.attention(
            self.norm(z),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        z = z + v
        return z + self.ffn(self.ffn_norm(z))
