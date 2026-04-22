"""Thin wrapper around MANO forward kinematics used by losses and predictions."""

import einops as eps
import smplx
import numpy as np
import torch
import torch.nn as nn

from ..constant import *


class RMANOLayer(nn.Module):
    """Run MANO and return root-relative joints/vertices in millimeters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "J_regressor_mano",
            torch.from_numpy(np.load(MANO_J_REGRESSOR_PATH)).type(torch.float32)
        )

        self.rmano_layer = smplx.create(MANO_ROOT, "mano", is_rhand=True, use_pca=False)
        self.rmano_layer.requires_grad_(False)
        self.rmano_layer.eval()

    def forward(self, pose, shape):
        """
        Args:
            pose: [B, T, 48] axis-angle MANO pose parameters.
            shape: [B, T, 10] MANO shape coefficients.

        Returns:
            joint_rel: [B, T, J, 3] joints relative to the wrist/root.
            verts_rel: [B, T, V, 3] vertices relative to the same root.
        """
        batch_size, _, _ = pose.shape
        njoint_hand = self.J_regressor_mano.shape[0]

        shape = eps.rearrange(shape, "b t d -> (b t) d")
        pose = eps.rearrange(pose, "b t d -> (b t) d")

        mano_output = self.rmano_layer(
            betas=shape,
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:],
            transl=torch.zeros(size=(pose.shape[0], 3), device=pose.device)
        )

        joints = torch.einsum(
            "nvd,jv->njd",
            mano_output.vertices, self.J_regressor_mano
        )
        joint_root_detach = joints[:, :1].detach()

        # Convert to millimeters and detach the root so only relative pose/shape matter here.
        verts_rel = eps.rearrange(
            (mano_output.vertices - joint_root_detach) * 1e3, # to mm
            "(b t) v d -> b t v d", b=batch_size
        )
        joint_rel = eps.rearrange(
            (joints - joint_root_detach) * 1e3,
            "(b t) j d -> b t j d", b=batch_size, j=njoint_hand
        )

        return joint_rel, verts_rel
