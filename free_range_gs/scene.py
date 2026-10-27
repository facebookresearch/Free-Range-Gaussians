from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .cameras import parse_camera


@dataclass
class LoadedScene:
    name: str
    images: torch.Tensor
    input_intrinsics: torch.Tensor
    input_world_to_camera: torch.Tensor
    render_intrinsics: torch.Tensor
    render_world_to_camera: torch.Tensor
    render_ids: list[str]
    render_references: list[np.ndarray | None]
    scene_path: Path


def _asset(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    if Path(relative).is_absolute():
        raise ValueError(f"{label} must be relative to the scene directory")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label} escapes the scene directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def _load_rgb(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    if image.size != (image_size, image_size):
        raise ValueError(
            f"{path} must be {image_size}x{image_size}, got {image.width}x{image.height}. "
            "Resize the image and adjust its camera intrinsics together before inference."
        )
    values = np.asarray(image, dtype=np.float32) / 255.0
    alpha = values[..., 3:4]
    return (values[..., :3] * alpha).astype(np.float32)


def load_scene(path: str | Path, image_size: int, expected_views: int = 4) -> LoadedScene:
    scene_path = Path(path).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene file does not exist: {scene_path}")
    try:
        document = json.loads(scene_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid scene JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("scene JSON must contain an object")
    if document.get("schema_version") != 1:
        raise ValueError("scene.schema_version must be 1")
    inputs = document.get("inputs")
    renders = document.get("renders")
    if not isinstance(inputs, list) or len(inputs) != expected_views:
        raise ValueError(f"scene.inputs must contain exactly {expected_views} views")
    if not isinstance(renders, list) or not renders:
        raise ValueError("scene.renders must contain at least one camera")
    root = scene_path.parent.resolve()
    image_values, input_intrinsics, input_extrinsics = [], [], []
    for index, record in enumerate(inputs):
        if not isinstance(record, dict):
            raise ValueError(f"inputs[{index}] must be an object")
        image_path = _asset(root, record.get("image"), f"inputs[{index}].image")
        rgb = _load_rgb(image_path, image_size)
        intrinsics, extrinsics = parse_camera(record, f"inputs[{index}]")
        image_values.append(torch.from_numpy(rgb).permute(2, 0, 1))
        input_intrinsics.append(intrinsics)
        input_extrinsics.append(extrinsics)
    render_intrinsics, render_extrinsics, render_ids, render_references = [], [], [], []
    seen_ids: set[str] = set()
    for index, record in enumerate(renders):
        if not isinstance(record, dict):
            raise ValueError(f"renders[{index}] must be an object")
        render_id = str(record.get("id", f"view_{index:02d}"))
        if not render_id or render_id in seen_ids or "/" in render_id or "\\" in render_id:
            raise ValueError(f"invalid or duplicate renders[{index}].id")
        seen_ids.add(render_id)
        intrinsics, extrinsics = parse_camera(record, f"renders[{index}]")
        reference = None
        if record.get("reference") is not None:
            reference_path = _asset(root, record["reference"], f"renders[{index}].reference")
            reference = _load_rgb(reference_path, image_size)
        render_intrinsics.append(intrinsics)
        render_extrinsics.append(extrinsics)
        render_ids.append(render_id)
        render_references.append(reference)
    return LoadedScene(
        name=str(document.get("name") or scene_path.parent.name),
        images=torch.stack(image_values).unsqueeze(0),
        input_intrinsics=torch.from_numpy(np.stack(input_intrinsics)).unsqueeze(0),
        input_world_to_camera=torch.from_numpy(np.stack(input_extrinsics)).unsqueeze(0),
        render_intrinsics=torch.from_numpy(np.stack(render_intrinsics)).unsqueeze(0),
        render_world_to_camera=torch.from_numpy(np.stack(render_extrinsics)).unsqueeze(0),
        render_ids=render_ids,
        render_references=render_references,
        scene_path=scene_path,
    )
