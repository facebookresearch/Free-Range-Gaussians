from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from pathlib import Path

from transformers import Dinov2Config, Dinov2Model


def get_2d_sincos_pos_embed(embed_dim, grid_size, dtype=torch.float32, device=None):
    if embed_dim % 4:
        raise ValueError("embedding dimension must be divisible by four")
    grid_h = torch.arange(grid_size, dtype=dtype, device=device)
    grid_w = torch.arange(grid_size, dtype=dtype, device=device)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=-1).reshape(-1, 2)
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim // 2, dtype=dtype, device=device) / (half_dim // 2)
    omega = 1.0 / (10000**omega)
    pos_h = grid[:, 0:1] * omega
    pos_w = grid[:, 1:2] * omega
    return torch.cat(
        [torch.sin(pos_h), torch.cos(pos_h), torch.sin(pos_w), torch.cos(pos_w)], dim=-1
    )


def plucker_raymap(intrinsics, world_to_camera, image_size, raymap_size):
    # Preserve the released checkpoint's evaluation camera encoding.
    # Changes to these ray features require validation with compatible weights.
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[:, None]
    if world_to_camera.ndim == 3:
        world_to_camera = world_to_camera[:, None]
    height = width = int(raymap_size)
    if raymap_size != image_size:
        scale = float(raymap_size) / float(image_size)
        intrinsics = intrinsics.clone()
        intrinsics[..., 0, :] *= scale
        intrinsics[..., 1, :] *= scale
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    rotation_t = rotation.transpose(-1, -2)
    xs, ys = torch.meshgrid(
        torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype),
        torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype),
        indexing="xy",
    )
    pixels = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)
    pixels = pixels.view(1, 1, height, width, 3).expand(*intrinsics.shape[:2], height, width, 3)
    camera_points = torch.einsum("bvij,bvhwj->bvhwi", torch.inverse(intrinsics), pixels)
    world_points = torch.einsum(
        "bvij,bvhwj->bvhwi", rotation_t, camera_points - translation[..., None, None, :]
    )
    directions = world_points - translation[..., None, None, :]
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    centers = translation[..., None, None, :]
    moments = torch.linalg.cross(centers.expand_as(directions), directions, dim=-1)
    return torch.cat([directions, moments], dim=-1)


class MultiviewImageEncoder(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov2-base",
        input_size=224,
        freeze_backbone=True,
        feature_dim=2048,
        patch_size=14,
        debug=False,
        use_camera_conditioning=True,
        use_downsampled_colors=True,
        use_img_pe=False,
        dino_model=None,
    ):
        super().__init__()
        self.model_name = str(dino_model or model_name)
        config_source = (
            str(Path(__file__).with_name("dinov2_config.json"))
            if self.model_name == "facebook/dinov2-base"
            else self.model_name
        )
        # All encoder parameters are loaded from the Free-Range Gaussians checkpoint.
        # Loading a second pretrained weight file would duplicate that download.
        self.model = Dinov2Model(Dinov2Config.from_pretrained(config_source))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)
        self.input_size = input_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.use_img_pe = use_img_pe
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()
        self.use_camera_conditioning = use_camera_conditioning
        self.use_downsampled_colors = use_downsampled_colors
        self.num_patches_per_dim = input_size // patch_size
        self.total_patches = self.num_patches_per_dim**2
        self.img_pe_dim = 32

    def forward(self, images, **kwargs):
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(f"expected {self.input_size}x{self.input_size} images")
        if images.min() < 0.0 or images.max() > 1.0:
            raise ValueError("images must be in [0, 1]")
        batch, views, channels, height, width = images.shape
        originals = images
        flat = images.reshape(batch * views, channels, height, width)
        flat = (flat - self.image_mean) / self.image_std
        with torch.set_grad_enabled(not self.freeze_backbone):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                features = self.model(flat).last_hidden_state.float()
        special_tokens = features.shape[1] - self.total_patches
        if special_tokens < 0:
            raise RuntimeError("DINOv2 returned fewer tokens than the configured patch grid")
        features = features.reshape(batch, views, features.shape[1], features.shape[2])
        if self.use_img_pe:
            position = get_2d_sincos_pos_embed(
                self.img_pe_dim, self.num_patches_per_dim, features.dtype, features.device
            )
            position = position[None, None].expand(batch, views, -1, -1)
            prefix = torch.zeros(batch, views, special_tokens, self.img_pe_dim, device=features.device, dtype=features.dtype)
            features = torch.cat([features, torch.cat([prefix, position], dim=2)], dim=-1)
        if self.use_downsampled_colors:
            colors = nn.functional.adaptive_avg_pool2d(
                originals.reshape(batch * views, channels, height, width),
                (self.num_patches_per_dim, self.num_patches_per_dim),
            )
            colors = colors.reshape(batch, views, channels, -1).permute(0, 1, 3, 2).to(features.dtype)
            prefix = torch.zeros(batch, views, special_tokens, channels, device=features.device, dtype=features.dtype)
            features = torch.cat([features, torch.cat([prefix, colors], dim=2)], dim=-1)
        if self.use_camera_conditioning:
            rays = plucker_raymap(kwargs["Ks"], kwargs["view_matrices"], self.input_size, self.num_patches_per_dim)
            rays = rearrange(rays, "b v h w c -> b v (h w) c").to(features.dtype)
            prefix = torch.zeros(batch, views, special_tokens, 6, device=features.device, dtype=features.dtype)
            features = torch.cat([features, torch.cat([prefix, rays], dim=2)], dim=-1)
        return rearrange(features, "b v n c -> b (v n) c")
