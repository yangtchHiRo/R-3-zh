"""Detached draft response, nonlinear regional aggregation, fixed-support readback."""

import torch
from torch import nn
from .regions import gather_nodes, pooled, normalize_pool


def mlp(n, out=128, zero=False):
    m = nn.Sequential(
        nn.LayerNorm(n), nn.Linear(n, 128), nn.SiLU(), nn.Linear(128, out)
    )
    if zero:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m


class ResponseFeedback:
    def read_support(self, current, z, cp):
        idx, w = z["idx"], z["weight"]
        K = idx.shape[-1]
        distance = torch.cdist(current["hist"][:, :, -1], cp).masked_fill(
            ~z["active"][:, None], 1e8
        )
        ri = distance.topk(K, largest=False).indices
        rw = (-distance.gather(2, ri) / self.encoder.radius).softmax(-1)
        rw = rw * gather_nodes(z["active"][..., None].float(), ri).squeeze(-1)
        rw = rw / rw.sum(-1, keepdim=True).clamp_min(1e-8)
        rv = current["reference_valid"][..., None]
        return torch.where(rv, idx, ri), torch.where(rv, w, rw)

    def region_feedback(self, current, z, ph, rec, context_tokens):
        # Candidate x/h/hp/ac are detached. Feedback encoders below are live.
        # Freeze the aggregation basis in the candidate summary too; the live
        # basis separately receives gradients through the final prediction pass.
        zd = {k: v.detach() if torch.is_tensor(v) else v for k, v in z.items()}
        x = current["hist"][:, :, -1].detach()
        cp = ph[:, :, -1].detach()
        dy = rec["x"] - x
        mean_d = pooled(dy, zd)
        idx = zd["idx"]
        K = idx.shape[-1]
        rel = (x[:, :, None] - gather_nodes(cp, idx)) / self.encoder.radius
        dd = (dy[:, :, None] - gather_nodes(mean_d, idx)) / self.encoder.radius
        deform = self.deformation(
            torch.cat((rel, dd, rec["h"][:, :, None].expand(-1, -1, K, -1)), -1)
        )
        deform = normalize_pool(deform, zd)[0]
        response = self.response(
            torch.cat(
                (
                    rec["hp"],
                    mean_d / self.encoder.radius,
                    rec["ac"] / self.acceleration_scale,
                    deform,
                ),
                -1,
            )
        )
        ri, rw = self.read_support(current, zd, cp)
        er = (x[:, :, None] - gather_nodes(cp, ri)) / self.encoder.radius
        ed = (dy[:, :, None] - gather_nodes(mean_d, ri)) / self.encoder.radius
        pair = torch.cat((er, ed), -1)
        alpha = rw[..., None].expand(-1, -1, -1, 4)
        values = (
            gather_nodes(self.response_value(response), ri)
            + self.response_message(pair)
        ).reshape(*ri.shape, 4, 32)
        msg = (alpha[..., None] * values).sum(2).flatten(-2)
        return self.response_out(msg), self.response_region(response)
