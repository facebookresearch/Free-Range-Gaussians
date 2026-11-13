# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import nn

from .gaussian import GaussianParamNormalizer
from .image_encoder import MultiviewImageEncoder
from .point_generator import PointGenerator
from .renderer import render_gaussians


class FreeRangeGS(nn.Module):
    def __init__(self, config, stats, dino_model=None):
        super().__init__()
        image_config = config.image_encoder.to_dict()
        image_config["dino_model"] = dino_model
        self.config = config
        self.image_encoder = MultiviewImageEncoder(**image_config)
        self.point_generator = PointGenerator(**config.point_generator.to_dict())
        self.N_points = config.point_generator.num_points
        self.gs_normalizer = GaussianParamNormalizer(stats)

    def get_gaussian_gradients(self, images, gauss_t, intrinsics, world_to_camera, alphas=None):
        with torch.enable_grad():
            values = gauss_t.detach().clone().requires_grad_(True)
            gaussians = self.gs_normalizer.decode(values)
            rendered, rendered_alpha = render_gaussians(
                gaussians,
                intrinsics,
                world_to_camera,
                self.config.image_encoder.input_size,
                self.config.image_encoder.input_size,
                background="black",
            )
            rendered = rendered.permute(0, 1, 4, 2, 3)
            energy = nn.functional.l1_loss(rendered, images)
            if alphas is not None:
                rendered_alpha = rendered_alpha.permute(0, 1, 4, 2, 3)
                energy = energy + nn.functional.l1_loss(rendered_alpha, alphas)
            energy.backward()
            return values.grad.detach()

    def reconstruct(
        self,
        images,
        intrinsics,
        world_to_camera,
        euler_steps=50,
        cfg_scale=2.0,
        render_grads_scale=50.0,
        alphas=None,
    ):
        image_features = self.image_encoder(images, Ks=intrinsics, view_matrices=world_to_camera)
        batch = images.shape[0]
        null_features = torch.zeros_like(image_features) if cfg_scale != 1.0 else None
        with torch.no_grad():
            values = torch.randn(batch, self.N_points, 14, device=images.device)
            dt = 1.0 / float(euler_steps)
            for step in range(euler_steps):
                timestep = torch.full((batch,), step * dt, device=images.device)
                if self.config.point_generator.use_render_grads:
                    gradients = self.get_gaussian_gradients(
                        images, values, intrinsics, world_to_camera, alphas
                    )
                    model_input = torch.cat([values, gradients], dim=-1)
                else:
                    model_input = values
                conditional = self.point_generator.flow_forward(image_features, model_input, timestep)
                if cfg_scale != 1.0:
                    unconditional = self.point_generator.flow_forward(null_features, model_input, timestep)
                    prediction = unconditional + cfg_scale * (conditional - unconditional)
                else:
                    prediction = conditional
                if getattr(self.config, "x_0_prediction", False):
                    velocity = (prediction - values) / (1 - timestep).clamp(min=0.05)
                else:
                    velocity = prediction
                render_gradient = 0
                if render_grads_scale > 0.0:
                    render_gradient = self.get_gaussian_gradients(
                        images,
                        values + velocity * (1 - step * dt),
                        intrinsics,
                        world_to_camera,
                        alphas,
                    )
                values = values + dt * velocity - render_grads_scale * render_gradient
        return self.gs_normalizer.decode(values)
