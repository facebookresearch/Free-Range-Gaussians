# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class Config:
    """Small recursive mapping with attribute access."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = dict(values)
        for key, value in self._values.items():
            setattr(self, key, self._wrap(value))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError(f"configuration must be a mapping: {path}")
        return cls(values)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(item) for item in value]
        return value

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value.to_dict()
        if isinstance(value, list):
            return [Config._unwrap(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        return {key: self._unwrap(getattr(self, key)) for key in self._values}

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value
        setattr(self, key, self._wrap(value))
