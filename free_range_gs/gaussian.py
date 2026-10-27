from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn


SH_C0 = 0.28209479177387814


class GaussianParams:
    def __init__(self, positions, scales, rotations, opacities, colors):
        if positions.ndim != 3:
            raise ValueError("Gaussian positions must have shape [B, N, 3]")
        self.positions = positions
        self.scales = scales
        self.rotations = rotations
        self.opacities = opacities
        self.colors = colors

    def to(self, device: torch.device | str) -> "GaussianParams":
        self.positions = self.positions.to(device)
        self.scales = self.scales.to(device)
        self.rotations = self.rotations.to(device)
        self.opacities = self.opacities.to(device)
        self.colors = self.colors.to(device)
        return self

    def save(self, path: str | Path, index: int = 0) -> None:
        save_ply(
            path,
            self.positions[index],
            self.colors[index],
            self.opacities[index],
            self.scales[index],
            self.rotations[index],
        )


class GaussianParamNormalizer(nn.Module):
    def __init__(self, stats: dict[str, Any], eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        for name in (
            "pos_mean",
            "pos_std",
            "log_scale_mean",
            "log_scale_std",
            "color_mean",
            "color_std",
            "logit_alpha_mean",
            "logit_alpha_std",
            "rot_mean",
            "rot_std",
        ):
            self.register_buffer(
                name,
                torch.tensor(stats[name], dtype=torch.float32),
                persistent=False,
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "GaussianParamNormalizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def decode(self, values: torch.Tensor) -> GaussianParams:
        if values.shape[-1] != 14:
            raise ValueError(f"expected 14 Gaussian channels, got {values.shape[-1]}")
        pos_n, log_scale_n, quat_n, alpha_n, color_n = torch.split(
            values, [3, 3, 4, 1, 3], dim=-1
        )
        positions = pos_n * (self.pos_std + self.eps) + self.pos_mean
        log_scales = log_scale_n * (self.log_scale_std + self.eps) + self.log_scale_mean
        scales = torch.exp(log_scales.clamp(min=-25, max=10))
        colors = (color_n * (self.color_std + self.eps) + self.color_mean).clamp(0.0, 1.0)
        logits = alpha_n * (self.logit_alpha_std + self.eps) + self.logit_alpha_mean
        opacities = torch.sigmoid(logits)
        rotations = quat_n * (self.rot_std + self.eps) + self.rot_mean
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + self.eps)
        return GaussianParams(positions, scales, rotations, opacities, colors)


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_ply(path, xyz, colors, opacity, scales, rotations) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = _numpy(xyz).astype(np.float32)
    colors = _numpy(colors).astype(np.float32)
    opacity = _numpy(opacity).reshape(-1, 1).astype(np.float32)
    scales = _numpy(scales).astype(np.float32)
    rotations = _numpy(rotations).astype(np.float32)
    n = xyz.shape[0]
    normals = np.zeros_like(xyz)
    opacity = np.log(opacity / (1.0 - opacity + 1e-9) + 1e-9).astype(np.float32)
    scales = np.log(scales + 1e-9).astype(np.float32)
    sh = np.zeros((n, 16, 3), dtype=np.float32)
    sh[:, 0, :] = (colors - 0.5) / SH_C0
    dc = sh[:, 0, :]
    rest = np.concatenate([sh[:, 1:, 0], sh[:, 1:, 1], sh[:, 1:, 2]], axis=1)
    names = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        *[f"f_rest_{i}" for i in range(45)],
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    array = np.empty(n, dtype=[(name, "f4") for name in names])
    stacked = np.concatenate([xyz, normals, dc, rest, opacity, scales, rotations], axis=1)
    array[:] = list(map(tuple, stacked))
    PlyData([PlyElement.describe(array, "vertex")], text=False).write(str(path))
