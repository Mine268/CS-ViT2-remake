"""Token-space transformation helpers for TI regularization."""

from __future__ import annotations

from typing import Tuple

import kornia.geometry.conversions as KC
import torch
import torch.nn as nn

from ..utils.rot import rotation6d_to_rotation_matrix, rotation_matrix_to_rotation6d
from .common import FrequencyEmbedder, Transformer


def build_z_rotation_matrix(angle_rad: torch.Tensor) -> torch.Tensor:
    """Build per-sample camera-z rotation matrices for `angle_rad` shaped `[N]`."""
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    row0 = torch.stack([cos, -sin, zeros], dim=-1)
    row1 = torch.stack([sin, cos, zeros], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotate_vectors_z(vectors: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Rotate `[N, 3]` vectors around camera z by the provided angles."""
    rot = build_z_rotation_matrix(angle_rad)
    return torch.matmul(rot, vectors.unsqueeze(-1)).squeeze(-1)


def invert_ti_pose_root(
    pose: torch.Tensor,
    angle_rad: torch.Tensor,
    joint_rep_type: str,
) -> torch.Tensor:
    """Apply the inverse TI z-rotation to the MANO global-orient component only."""
    root_rot = build_z_rotation_matrix(-angle_rad)
    if joint_rep_type == "3":
        root = pose[:, :3]
        hand_pose = pose[:, 3:]
        root_mat = KC.axis_angle_to_rotation_matrix(root)
        root_inv = KC.rotation_matrix_to_axis_angle(torch.matmul(root_rot, root_mat))
        return torch.cat([root_inv, hand_pose], dim=-1)
    if joint_rep_type == "6d":
        root = pose[:, :6]
        hand_pose = pose[:, 6:]
        root_mat = rotation6d_to_rotation_matrix(root)
        root_inv = rotation_matrix_to_rotation6d(torch.matmul(root_rot, root_mat))
        return torch.cat([root_inv, hand_pose], dim=-1)
    if joint_rep_type == "quat":
        root = pose[:, :4]
        hand_pose = pose[:, 4:]
        root_mat = KC.quaternion_to_rotation_matrix(root)
        root_inv = KC.rotation_matrix_to_quaternion(torch.matmul(root_rot, root_mat))
        return torch.cat([root_inv, hand_pose], dim=-1)
    raise NotImplementedError(f"Unsupported joint_rep_type={joint_rep_type}")


def invert_ti_camera(
    pred_ray_unit: torch.Tensor,
    pred_rho: torch.Tensor,
    scale: torch.Tensor,
    angle_rad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undo TI scale/rotation using the ray-distance parameterization."""
    ray_inv = rotate_vectors_z(pred_ray_unit, -angle_rad)
    rho_inv = pred_rho / scale[:, None]
    trans_inv = ray_inv * rho_inv
    return trans_inv, ray_inv, rho_inv


class TIParamEmbedder(nn.Module):
    """Embed continuous TI parameters `(angle, log-scale)` into FiLM coefficients."""

    def __init__(self, hidden_size: int, num_frequencies: int = 6):
        super().__init__()
        param_dim = 2 * (2 * num_frequencies + 1)
        self.embedder = FrequencyEmbedder(num_frequencies, num_frequencies - 1)
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, scale: torch.Tensor, angle_rad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = torch.stack([angle_rad, torch.log(scale)], dim=-1)
        embeds = self.embedder(params)
        gamma, beta = self.net(embeds).chunk(2, dim=-1)
        return gamma, beta


class TITokenTransform(nn.Module):
    """FiLM-modulated token transformer with zero-init residual output."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        num_frequencies: int = 6,
    ):
        super().__init__()
        if heads <= 0 or dim % heads != 0:
            raise ValueError(
                f"TITokenTransform requires dim % heads == 0, got dim={dim}, heads={heads}"
            )
        self.param_embedder = TIParamEmbedder(hidden_size=dim, num_frequencies=num_frequencies)
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim // heads,
            mlp_dim=int(dim * mlp_ratio),
            dropout=dropout,
            norm="layer",
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, tokens: torch.Tensor, scale: torch.Tensor, angle_rad: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.param_embedder(scale=scale, angle_rad=angle_rad)
        conditioned = (1.0 + gamma[:, None, :]) * tokens + beta[:, None, :]
        delta = self.out_proj(self.out_norm(self.transformer(conditioned)))
        return tokens + delta
