import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ReferenceBias(nn.Module):
    """Dense attention, no distance cutoff. Initial terms only for valid object pairs."""

    def __init__(self, meta, heads=4):
        super().__init__()
        self.radius = float(meta["connectivity_radius"])
        self.net = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, heads))

    def forward(self, b):
        x = torch.cat((b["hist"][:, :, -1], b["ctrl_hist"][:, :, -1, :3]), 1)
        B, N, _ = x.shape
        n = b["hist"].shape[1]
        ref = torch.cat((b["initial_ref"], torch.zeros_like(x[:, n:])), 1)
        valid = torch.cat(
            (
                b.get(
                    "reference_valid",
                    torch.ones(B, n, dtype=torch.bool, device=x.device),
                ),
                torch.zeros(B, N - n, dtype=torch.bool, device=x.device),
            ),
            1,
        )
        obj = torch.arange(N, device=x.device) < n
        out = []
        r = self.radius
        for s in range(0, N, 128):
            d = (x[:, None] - x[:, s : s + 128, None]).norm(dim=-1)
            d0 = (ref[:, None] - ref[:, s : s + 128, None]).norm(dim=-1)
            material = (valid[:, None] & valid[:, s : s + 128, None]).to(x.dtype)
            control = (
                (~(obj[None, :] & obj[s : s + 128, None]))[None]
                .expand(B, -1, -1)
                .to(x.dtype)
            )
            strain = ((d - d0) / d0.clamp_min(0.25 * r)).clamp(-10, 10)
            f = torch.stack(
                (
                    d / r,
                    material * d0 / r,
                    material * strain,
                    material * torch.exp(-d0 / r),
                    torch.exp(-d / r),
                    material,
                    control,
                    control * torch.exp(-d / r),
                ),
                -1,
            )
            # Checkpoint the pair MLP too: do not retain N^2 x 32 hidden features at H5.
            y = (
                checkpoint(self.net, f, use_reentrant=False)
                if self.training and torch.is_grad_enabled()
                else self.net(f)
            )
            out.append(y.permute(0, 3, 1, 2))
        return torch.cat(out, 2)
