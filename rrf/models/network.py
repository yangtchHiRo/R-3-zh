"""RRF: reference-supported regional dynamics with same-step response recycling."""

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .encoders import (
    Block,
    CrossBlock,
    TypedEncoder,
    ActionEncoder,
    available_control,
    controller_at,
    statistic,
)
from .reference import ReferenceBias
from .regions import (
    Structure,
    gather_nodes,
    pool_sparse,
    pooled,
    normalize_pool,
    geometric_state,
)
from .regional import PatchBranch
from .feedback import ResponseFeedback, mlp


def append_history(batch, pred, next_control, tau):
    new = dict(batch)
    new["hist"] = torch.cat((batch["hist"][:, :, 1:], pred[:, :, None]), 2)
    new["ctrl_hist"] = torch.cat(
        (batch["ctrl_hist"][:, :, 1:], next_control[:, :, None]), 2
    )
    # Time is relative to the original window origin in path_times/query_times.
    new["hist_times"] = torch.cat((batch["hist_times"][:, 1:], tau[:, None]), 1)
    return new


class RRF(ResponseFeedback, nn.Module):
    def __init__(self, meta, cfg=None):
        super().__init__()
        self.meta = meta
        d = 128
        self.encoder = TypedEncoder(meta, dict(encoder="shared"), d)
        self.action = ActionEncoder(d, layers=2, heads=4)
        self.blocks = nn.ModuleList([Block(d, 4, 512) for _ in range(3)])
        self.cross = nn.ModuleList([CrossBlock(d, 4) for _ in range(3)])
        self.bias = ReferenceBias(meta)
        self.norm = nn.LayerNorm(d)
        self.time = nn.Sequential(nn.Linear(9, d), nn.SiLU(), nn.Linear(d, d))
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 3))
        self.register_buffer(
            "acceleration_scale",
            statistic(meta, "acceleration_std", 1.0).clamp_min(1e-3),
        )
        cfg = {} if cfg is None else cfg
        self.register_buffer("geom_freq", torch.tensor([1.0, 2.0, 4.0, 8.0]) * math.pi)
        self.geometry = nn.Sequential(
            nn.Linear(81, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.geometry_gain = nn.Parameter(torch.tensor(0.1))
        self.coarse_weight = float(cfg.get("coarse_loss_weight", 0.1))
        self.structure = Structure(meta, cfg)
        self.patch_branch = PatchBranch(meta)
        self.deformation = mlp(134)
        self.response = mlp(262)
        self.response_value = nn.Linear(128, 128)
        self.response_message = mlp(6)
        self.response_out = mlp(128, zero=True)
        self.response_region = mlp(128, zero=True)

    def ck(self, fn, *args):
        return (
            checkpoint(fn, *args, use_reentrant=False)
            if self.training and torch.is_grad_enabled()
            else fn(*args)
        )

    def fourier(self, b):
        x = b["hist"][:, :, -1]
        c = b["ctrl_hist"][:, :, -1, :3]
        ref = b["initial_ref"]
        rv = b["reference_valid"][..., None]
        center = (ref * rv).sum(1, keepdim=True) / rv.sum(1, keepdim=True).clamp_min(1)
        near = torch.cdist(x, c).argmin(-1)
        ctrl = c.gather(1, near[..., None].expand(-1, -1, 3))
        f = torch.cat(
            (
                (x - center) / self.encoder.scale,
                (x - ref) * rv / self.encoder.radius,
                (ctrl - x) / self.encoder.radius,
            ),
            -1,
        )
        phase = f[..., None] * self.geom_freq
        return self.geometry(
            torch.cat((f, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
        )

    def solve(
        self,
        current,
        tau,
        action,
        pad,
        zbase,
        zwrite,
        ph,
        pv,
        encoded,
        context,
        recycled=None,
        local=None,
        memory=None,
    ):
        zo, zc, obs = encoded
        N = zo.shape[1]
        t = tau[:, None]
        tf = torch.cat(
            (
                t,
                t.sin(),
                t.cos(),
                (2 * t).sin(),
                (2 * t).cos(),
                (4 * t).sin(),
                (4 * t).cos(),
                t.square(),
                t.sqrt(),
            ),
            -1,
        )
        z = torch.cat((zo, zc), 1) + self.time(tf)[:, None]
        particles = z[:, :N] + self.geometry_gain * self.fourier(current)
        region_injection = None
        if recycled is not None:
            pi, region_injection = self.region_feedback(
                current, zwrite, ph, recycled, zo
            )
            particles = particles + pi
        z = torch.cat((particles, z[:, N:]), 1)
        bias = self.bias(current)
        for block, cross in zip(self.blocks, self.cross):
            z = self.ck(block, z, bias)
            z = self.ck(cross, z, action, pad)
        h = z[:, :N]
        zc = z[:, N:]
        br = self.patch_branch
        idx = zwrite["idx"]
        K = idx.shape[-1]
        cp = ph[:, :, -1]
        rel = (obs["x"][:, :, None] - gather_nodes(cp, idx)) / br.radius
        relv = (obs["velocity"][:, :, None] - gather_nodes(pv[:, :, -1], idx)) / br.vs
        ref = zbase["anchors"]
        rel0 = (
            (current["initial_ref"][:, :, None] - gather_nodes(ref, idx))
            / br.radius
            * current["reference_valid"][:, :, None, None]
        )
        down = br.down(
            torch.cat((h[:, :, None].expand(-1, -1, K, -1), rel, relv, rel0), -1)
        )
        hp = normalize_pool(down, zwrite)[0] + br.property(zbase["C"])
        if region_injection is not None:
            hp = hp + region_injection
        coarse = dict(
            current, hist=ph, initial_ref=ref, reference_valid=zwrite["active"]
        )
        pbias = self.bias(coarse)
        relation = torch.nn.functional.pad(
            zbase["A"].permute(0, 3, 1, 2).log(), (0, zc.shape[1], 0, zc.shape[1])
        )
        keys = torch.cat(
            (
                zwrite["active"],
                torch.ones(zc.shape[:2], device=zc.device, dtype=torch.bool),
            ),
            1,
        )
        pbias = (pbias + relation).masked_fill(~keys[:, None, None], float("-inf"))
        hp = self.ck(br.block, torch.cat((hp, zc), 1), pbias)
        hp = self.ck(br.cross, hp, action, pad)[:, : cp.shape[1]]
        di, dw = self.read_support(current, zbase, cp)
        dr = (obs["x"][:, :, None] - gather_nodes(cp, di)) / br.radius
        dv = (obs["velocity"][:, :, None] - gather_nodes(pv[:, :, -1], di)) / br.vs
        d0 = (
            (current["initial_ref"][:, :, None] - gather_nodes(ref, di))
            / br.radius
            * current["reference_valid"][:, :, None, None]
        )
        msg = br.up(
            torch.cat(
                (
                    gather_nodes(hp, di),
                    h[:, :, None].expand(-1, -1, K, -1),
                    gather_nodes(zbase["C"], di),
                    dr,
                    dv,
                    d0,
                ),
                -1,
            )
        )
        h = h + br.gain * (msg * dw[..., None]).sum(2)
        ac = br.motion_head(hp) * self.acceleration_scale
        ai = (gather_nodes(ac, di) * dw[..., None]).sum(2)
        residual = self.decoder(self.norm(h)) * self.acceleration_scale
        dt = (tau - current["hist_times"][:, -1])[:, None, None]
        pred = obs["x"] + dt * (obs["velocity"] + dt * (ai + residual))
        c = dict(
            x=obs["x"],
            v=obs["velocity"],
            dt=dt,
            a=ac,
            shared_particle=ai,
            particle_residual=residual,
        )
        rec = dict(
            x=pred.detach(),
            h=h.detach(),
            hp=hp.detach(),
            ac=ac.detach(),
            region_dy=pooled(pred - obs["x"], zwrite).detach(),
        )
        return (pred, c, rec)

    def forward(self, b, diagnostics=False):
        if b["hist"].shape[2] != 3:
            raise ValueError("Expected three historical states")
        action, pad, _ = self.action(
            available_control(b, self.meta, b["ctrl_path"]),
            b["path_times"],
            b["path_mask"],
            b["query_times"],
        )
        reference = self.structure(b)
        current = dict(b)
        pred = []
        coarse = []
        drafts = []
        for k, tau in enumerate(b["query_times"].unbind(1)):
            ph, pv = geometric_state(current, reference)
            encoded = self.encoder(current, tau)
            args = (
                current,
                tau,
                action[:, k],
                pad[:, k],
                reference,
                reference,
                ph,
                pv,
                encoded,
                None,
            )
            with torch.no_grad():
                draft, _, response = self.solve(*args)
            final, c, _ = self.solve(*args, recycled=response)
            pred.append(final)
            coarse.append(c)
            if diagnostics:
                drafts.append(draft.detach())
            current = append_history(current, final, controller_at(b, tau), tau)
        result = dict(pred=torch.stack(pred, 1), coarse=coarse, structure=reference)
        if diagnostics:
            result["draft"] = torch.stack(drafts, 1)
        return result

    def auxiliary_loss(self, result, target, valid, reference):
        # Same loss in BOTH arms. Compare actual interpolated field, not ac alone.
        z = result["structure"]
        idx = z["idx"]
        w = z["weight"].detach()
        m = z["anchors"].shape[1]
        k = idx.shape[-1]
        terms = []
        for q, c in enumerate(result["coarse"]):
            ww = w * valid[:, q, :, None]

            def pool(x):
                return pool_sparse(x[:, :, None].expand(-1, -1, k, -1), idx, ww, m)

            virtual = c["x"] + c["dt"] * (c["v"] + c["dt"] * c["shared_particle"])
            pred, mass = pool(virtual - reference)
            gt, _ = pool(target[:, q] - reference)
            active = (mass > 1e-7).to(pred.dtype)
            terms.append(
                (
                    ((pred - gt).square().mean(-1) * active).sum(-1)
                    / active.sum(-1).clamp_min(1)
                ).mean()
            )
        return 100 * self.coarse_weight * torch.stack(terms).sum()
