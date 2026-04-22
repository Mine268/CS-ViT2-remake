from __future__ import annotations

import einops as eps
import torch
import torch.nn as nn

from .common import FeedForward, PreNorm, default


class TRotionalPositionEmbedding(nn.Module):
    def __init__(self, dim: int, multi_head: bool = False):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.multi_head = multi_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        freqs = t.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)
        if self.multi_head:
            cos_vals = cos_vals.unsqueeze(1)
            sin_vals = sin_vals.unsqueeze(1)
        return self._apply_rope(x, cos_vals, sin_vals)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshaped.unbind(dim=-1)
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        return torch.stack([x1_rot, x2_rot], dim=-1).flatten(start_dim=-2)


class CausalTRoPESelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.pe = TRotionalPositionEmbedding(dim_head, multi_head=bool(heads > 1))
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda tensor: eps.rearrange(tensor, "b n (h d) -> b h n d", h=self.heads),
            [q, k, v],
        )
        q = self.pe(q, t)
        k = self.pe(k, t)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        n = x.shape[1]
        causal_mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(diagonal=1)
        dots.masked_fill_(causal_mask[None, None], float("-inf"))
        attn = torch.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = eps.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TRoPECrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.pe = TRotionalPositionEmbedding(dim_head, multi_head=bool(heads > 1))
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, tq, tk, context=None):
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(
            lambda tensor: eps.rearrange(tensor, "b n (h d) -> b h n d", h=self.heads),
            [q, k, v],
        )
        q = self.pe(q, tq)
        k = self.pe(k, tk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = eps.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TRoPETransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = CausalTRoPESelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = TRoPECrossAttention(
                dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, tq: torch.Tensor, tk: torch.Tensor, context=None):
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, t=tq) + x
            x = cross_attn(x, tq=tq, tk=tk, context=context) + x
            x = ff(x) + x
        return x


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_head: int,
        num_layer: int,
        dropout: float,
        trope_scalar: float = 20.0,
        zero_linear: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.trope_scalar = trope_scalar
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            sa = CausalTRoPESelfAttention(dim, heads=num_head, dim_head=dim // num_head, dropout=dropout)
            ff = FeedForward(dim, dim * 2, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm="layer", norm_cond_dim=-1),
                        PreNorm(dim, ff, norm="layer", norm_cond_dim=-1),
                    ]
                )
            )

        self.zero_linear = nn.Linear(self.dim, self.dim, bias=True)
        if zero_linear:
            nn.init.zeros_(self.zero_linear.weight)
            nn.init.zeros_(self.zero_linear.bias)

    def forward(self, token: torch.Tensor, timestamp: torch.Tensor) -> torch.Tensor:
        timestamp = timestamp / self.trope_scalar
        x = token
        for sa, ff in self.layers:
            x = sa(x, t=timestamp) + x
            x = ff(x) + x
        delta = self.zero_linear(x)
        return token + delta
