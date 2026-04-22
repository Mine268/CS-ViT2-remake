from __future__ import annotations

from typing import Dict, List, Optional

import einops as eps
from accelerate.logging import get_logger
import torch
import torch.nn as nn
import transformers


logger = get_logger(__name__)


class ViTBackbone(nn.Module):
    def __init__(
        self,
        backbone_str: str,
        img_size: Optional[int],
        infusion_feats_lyr: Optional[List[int]],
        backbone_kwargs: Dict,
    ):
        super().__init__()

        self.backbone_str = backbone_str
        self.img_size = img_size
        self.infusion_feats_lyr = infusion_feats_lyr

        backbone_cfg = transformers.AutoConfig.from_pretrained(self.backbone_str)
        self.model_type = backbone_cfg.model_type
        self.has_cls_token = self.model_type not in {"swin", "swinv2"}
        self.patch_size = backbone_cfg.patch_size
        self.hidden_size = backbone_cfg.hidden_size
        self.feature_stride = getattr(backbone_cfg, "encoder_stride", None) or self.patch_size

        if self.img_size is None:
            self.img_size = backbone_cfg.image_size
            logger.info("No img_size provided for backbone. Use pretrained config img_size=%s", self.img_size)

        if self.img_size % self.patch_size != 0:
            raise ValueError(f"img_size={self.img_size} and patch_size={self.patch_size} mismatch")
        if self.img_size % self.feature_stride != 0:
            raise ValueError(
                f"img_size={self.img_size} and feature_stride={self.feature_stride} "
                f"mismatch for backbone={self.backbone_str}"
            )
        self.num_patch = self.img_size // self.feature_stride

        self.backbone = transformers.AutoModel.from_pretrained(
            self.backbone_str,
            output_hidden_states=True,
            **backbone_kwargs,
        )

        if self.infusion_feats_lyr is not None:
            self.proj_size = 64
            self.projection_cls = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_size, self.proj_size, bias=True),
                        nn.BatchNorm1d(self.proj_size),
                        nn.ReLU(),
                    )
                    for _ in range(len(self.infusion_feats_lyr))
                ]
            )
            self.projections_map = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.hidden_size, self.proj_size, kernel_size=1),
                        nn.BatchNorm2d(self.proj_size),
                        nn.ReLU(),
                    )
                    for _ in range(len(self.infusion_feats_lyr))
                ]
            )
            self.fusion_cls = nn.Sequential(
                nn.Linear(self.proj_size * len(self.infusion_feats_lyr), self.hidden_size, bias=True),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(
                    self.proj_size * len(self.infusion_feats_lyr),
                    self.hidden_size,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.hidden_size),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("Input tensor must be square")
        if x.shape[-1] != self.img_size:
            raise ValueError(f"Input tensor shape {x.shape} does not match img_size={self.img_size}")

        backbone_output = self.backbone(x)
        if self.infusion_feats_lyr is None:
            return backbone_output.last_hidden_state

        hidden_states = [backbone_output.hidden_states[i] for i in self.infusion_feats_lyr]
        token_clss, token_patches = [], []
        expected_patch_token_count = self.num_patch ** 2
        for idx, hidden_state in enumerate(hidden_states):
            if self.has_cls_token:
                expected_token_count = expected_patch_token_count + 1
                if hidden_state.shape[1] != expected_token_count:
                    raise ValueError(
                        f"Hidden state index={self.infusion_feats_lyr[idx]} token_count="
                        f"{hidden_state.shape[1]}, expected={expected_token_count}"
                    )
                token_cls, token_patch = hidden_state[:, 0], hidden_state[:, 1:]
            else:
                if hidden_state.shape[1] != expected_patch_token_count:
                    raise ValueError(
                        f"Hidden state index={self.infusion_feats_lyr[idx]} token_count="
                        f"{hidden_state.shape[1]}, expected={expected_patch_token_count}"
                    )
                token_cls = torch.mean(hidden_state, dim=1)
                token_patch = hidden_state
            token_patch = eps.rearrange(token_patch, "b (h w) c -> b c h w", h=self.num_patch)
            token_cls = self.projection_cls[idx](token_cls)
            token_patch = self.projections_map[idx](token_patch)
            token_clss.append(token_cls)
            token_patches.append(token_patch)

        token_clss = torch.cat(token_clss, dim=-1)
        token_patches = torch.cat(token_patches, dim=1)
        token_clss = self.fusion_cls(token_clss)
        token_patches = self.fusion_conv(token_patches)
        token_patches = eps.rearrange(token_patches, "b c h w -> b (h w) c")
        if self.has_cls_token:
            return torch.cat([token_clss[:, None], token_patches], dim=1)
        return token_patches

    def get_patch_size(self):
        return self.patch_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_img_size(self):
        return self.img_size

    def get_num_patch(self):
        return self.num_patch

    def get_has_cls_token(self):
        return self.has_cls_token
