# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import Config


PACKAGE_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = PACKAGE_ROOT.parent
CONFIG_PATH = PACKAGE_ROOT / "config.yaml"
CHECKPOINT_PATH = RELEASE_ROOT / "checkpoint" / "frg_ckpt.pt"
CHECKPOINT_REPO_ID = "facebook/free-range-gaussians"
CHECKPOINT_DOWNLOAD_URL = f"https://huggingface.co/{CHECKPOINT_REPO_ID}"


def load_config() -> Config:
    return Config.from_yaml(CONFIG_PATH)


def checkpoint_path(path: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser().resolve() if path is not None else CHECKPOINT_PATH
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Free-Range Gaussians checkpoint not found: {candidate}\n"
        f"Request access through the gated Hugging Face model repository: {CHECKPOINT_DOWNLOAD_URL}\n"
        "After approval, run these commands from the code checkout:\n"
        "  hf auth login\n"
        f"  hf download {CHECKPOINT_REPO_ID} frg_ckpt.pt --local-dir checkpoint"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda":
        raise RuntimeError("Free-Range Gaussians inference requires a CUDA GPU; CPU inference is unsupported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return requested


def prepare_output(path: str | Path, overwrite: bool) -> Path:
    output = Path(path).resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_model(checkpoint, dino_model, device, num_gaussians=8192, image_size=512):
    from .model import FreeRangeGS

    config = load_config()
    if num_gaussians <= 0 or num_gaussians % int(config.model.point_generator.patch_size):
        raise ValueError("num_gaussians must be positive and divisible by the configured patch size")
    config.model.point_generator.set("num_points", int(num_gaussians))
    config.model.image_encoder.set("input_size", int(image_size))
    model = FreeRangeGS(config.model, config.normalization.to_dict(), dino_model=dino_model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("checkpoint must contain model_state_dict")
    incompatibility = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"non-exact checkpoint load: {incompatibility}")
    model.to(device).eval()
    return model, config
