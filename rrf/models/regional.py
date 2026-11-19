import torch
from torch import nn
from .encoders import statistic, Block, CrossBlock


class PatchBranch(nn.Module):
    def __init__(self, meta):
        super().__init__()
        self.radius = float(meta["connectivity_radius"])
        self.register_buffer("vs", statistic(meta, "velocity_std", 0.1).clamp_min(1e-4))
        self.down = nn.Sequential(
            nn.LayerNorm(137), nn.Linear(137, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.up = nn.Sequential(
            nn.LayerNorm(297), nn.Linear(297, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.property = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 128))
        self.block = Block(128, 4, 512)
        self.cross = CrossBlock(128, 4)
        self.gain = nn.Parameter(torch.tensor(0.1))
        motion = True
        if motion:
            self.motion_head = nn.Sequential(
                nn.LayerNorm(128), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3)
            )
            nn.init.zeros_(self.motion_head[-1].weight)
            nn.init.zeros_(self.motion_head[-1].bias)
