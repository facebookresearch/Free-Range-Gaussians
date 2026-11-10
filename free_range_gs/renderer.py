# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from gsplat.rendering import rasterization

from .gaussian import GaussianParams


def render_gaussians(
    gaussians: GaussianParams,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    width: int,
    height: int,
    background: str = "black",
) -> tuple[torch.Tensor, torch.Tensor]:
    if background not in {"black", "white"}:
        raise ValueError("background must be 'black' or 'white'")
    batch, views = world_to_camera.shape[:2]
    value = 0.0 if background == "black" else 1.0
    backgrounds = torch.full(
        (batch, views, 3), value, device=gaussians.positions.device, dtype=torch.float32
    )
    colors, alphas, _ = rasterization(
        means=gaussians.positions.float(),
        colors=gaussians.colors.float(),
        opacities=gaussians.opacities.squeeze(-1).float(),
        scales=gaussians.scales.float(),
        quats=gaussians.rotations.float(),
        viewmats=world_to_camera.float(),
        Ks=intrinsics.float(),
        width=width,
        height=height,
        packed=False,
        backgrounds=backgrounds,
    )
    return colors.clamp(0.0, 1.0), alphas.clamp(0.0, 1.0)
