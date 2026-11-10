# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .transformer1d import Transformer1D


def timestep_position_encoding(t, dim, max_period=10000):
    if t.ndim != 1:
        raise ValueError("timesteps must have shape [B]")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * frequencies[None, :] * 2 * math.pi
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return F.pad(embedding, (0, 1)) if dim % 2 else embedding


class PointGenerator(nn.Module):
    def __init__(
        self,
        num_points=2048,
        feature_dim=768,
        hidden_dim=512,
        num_layers=10,
        num_heads=8,
        attention_head_dim=32,
        dropout=0.0,
        activation_fn="geglu",
        attention_bias=False,
        use_mixed_precision=False,
        use_gaussian_param_pe=False,
        pe_num_freqs_abs=6,
        pe_num_freqs_rel=4,
        pe_include_x_abs=True,
        pe_use_relative=False,
        patch_size=1,
        permute=False,
        use_render_grads=False,
        backbone_type="transformer1d",
        use_learnable_queries=False,
    ):
        super().__init__()
        if backbone_type != "transformer1d":
            raise ValueError("this inference release only supports backbone_type=transformer1d")
        if use_gaussian_param_pe or use_learnable_queries:
            raise ValueError("this inference release does not support optional training-only point-generator modes")
        self.gaussian_dim = 14
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.permute = permute
        self.input_dim = self.gaussian_dim * 2 if use_render_grads else self.gaussian_dim
        self.image_proj = nn.Linear(feature_dim, hidden_dim)
        self.time_embedding = nn.Linear(hidden_dim, hidden_dim)
        self.point_embeddings = nn.Parameter(torch.randn(1, 32, hidden_dim) * 0.02)
        self.backbone_type = backbone_type
        self.use_learnable_queries = use_learnable_queries
        self.use_gaussian_param_pe = use_gaussian_param_pe
        self.input_mapper = nn.Sequential(
            nn.Linear(self.input_dim * patch_size, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.output_mapper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.gaussian_dim * patch_size)
        )
        self.backbone = Transformer1D(
            num_attention_heads=num_heads,
            attention_head_dim=attention_head_dim,
            in_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cross_attention_dim=None,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            use_mixed_precision=use_mixed_precision,
        )

    def flow_forward(self, image_features, gauss_t, t):
        batch_size = image_features.shape[0]
        point_count = gauss_t.shape[1]
        if self.patch_size > 1:
            if point_count % self.patch_size:
                raise ValueError("number of Gaussians must be divisible by patch_size")
            if self.permute:
                permutation = torch.randperm(point_count, device=gauss_t.device)
                inverse = torch.argsort(permutation)
                gauss_t = gauss_t[:, permutation, :]
            gauss_t = rearrange(gauss_t, "b (n p) d -> b n (p d)", p=self.patch_size)
            point_count //= self.patch_size
        point_tokens = self.input_mapper(gauss_t)
        image_features = self.image_proj(image_features)
        _ = self.time_embedding(timestep_position_encoding(t, self.hidden_dim)).unsqueeze(1)
        point_embeds = self.point_embeddings.expand(batch_size, -1, -1)
        tokens = torch.cat((image_features, point_embeds, point_tokens), dim=1)
        tokens = self.backbone(hidden_states=tokens, encoder_hidden_states=None, timestep=t)
        velocity = self.output_mapper(tokens[:, -point_count:, :])
        if self.patch_size > 1:
            velocity = rearrange(velocity, "b n (p d) -> b (n p) d", p=self.patch_size)
            if self.permute:
                velocity = velocity[:, inverse, :]
        return velocity
