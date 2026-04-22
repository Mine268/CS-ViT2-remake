from __future__ import annotations

import einops as eps
import torch
import torch.nn as nn

from .common import TransformerDecoder


class PerspInfoEmbedderCrossAttn(nn.Module):
    def __init__(self, hidden_size: int, num_sample: int, num_token: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sample = num_sample

        self.net = TransformerDecoder(
            num_tokens=num_token,
            token_dim=self.hidden_size,
            dim=self.hidden_size,
            depth=1,
            heads=8,
            mlp_dim=4 * self.hidden_size,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            emb_dropout_type="drop",
            norm="layer",
            norm_cond_dim=-1,
            context_dim=4,
            skip_token_embedding=False,
        )

        self.zero_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.zeros_(self.zero_linear.weight)

        grid_edge = torch.linspace(
            1 / self.num_sample * 0.5,
            1 - 1 / self.num_sample * 0.5,
            self.num_sample,
        )
        grid_uv = torch.stack(
            [
                grid_edge[:, None].expand(-1, grid_edge.shape[0]),
                grid_edge[None, :].expand(grid_edge.shape[0], -1),
            ],
            dim=-1,
        )
        self.register_buffer("grid_edge", grid_edge)
        self.register_buffer("grid_uv", grid_uv)

    def forward(
        self,
        feats: torch.Tensor,
        bbox: torch.Tensor,
        focal: torch.Tensor,
        princpt: torch.Tensor,
    ) -> torch.Tensor:
        x_grid = bbox[:, 0:1] + (bbox[:, 2:3] - bbox[:, 0:1]) * self.grid_edge[None, :]
        y_grid = bbox[:, 1:2] + (bbox[:, 3:4] - bbox[:, 1:2]) * self.grid_edge[None, :]
        grid_xy = torch.stack(
            [
                x_grid[:, :, None].expand(-1, -1, self.grid_edge.shape[0]),
                y_grid[:, None, :].expand(-1, self.grid_edge.shape[0], -1),
            ],
            dim=-1,
        )

        directions = (grid_xy - princpt[:, None, None, :]) / focal[:, None, None, :]
        directions = torch.cat([directions, torch.ones_like(directions[..., :1])], dim=-1)
        directions = directions / torch.norm(directions, p="fro", dim=-1, keepdim=True)
        directions = directions[..., :2]
        directions = torch.cat(
            [
                directions,
                self.grid_uv[None, ...].expand(directions.shape[0], -1, -1, -1),
            ],
            dim=-1,
        )
        directions = eps.rearrange(directions, "b p q d -> b (p q) d")

        out = self.net(feats, context=directions)
        out = self.zero_linear(out)
        return out + feats
