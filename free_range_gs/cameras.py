from __future__ import annotations

from typing import Any

import numpy as np


def _matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def parse_camera(record: dict[str, Any], label: str) -> tuple[np.ndarray, np.ndarray]:
    if "intrinsics" not in record:
        raise ValueError(f"{label} is missing intrinsics")
    intrinsics = _matrix(record["intrinsics"], (3, 3), f"{label}.intrinsics")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError(f"{label} focal lengths must be positive")
    if abs(float(np.linalg.det(intrinsics))) < 1e-8:
        raise ValueError(f"{label}.intrinsics is singular")
    has_w2c = "world_to_camera" in record
    has_c2w = "camera_to_world" in record
    if has_w2c == has_c2w:
        raise ValueError(f"{label} must define exactly one of world_to_camera or camera_to_world")
    if has_w2c:
        world_to_camera = _matrix(record["world_to_camera"], (4, 4), f"{label}.world_to_camera")
    else:
        camera_to_world = _matrix(record["camera_to_world"], (4, 4), f"{label}.camera_to_world")
        try:
            world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{label}.camera_to_world is singular") from error
    if not np.allclose(world_to_camera[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{label} camera matrix must end with [0, 0, 0, 1]")
    rotation = world_to_camera[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=5e-3, rtol=0
    ):
        raise ValueError(f"{label} rotation must be orthonormal with determinant +1")
    return intrinsics, world_to_camera
