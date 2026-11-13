#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OBJECT=${1:-TOWER_TUMBLING}
CASE=${2:-nc4}
OUTPUT=${3:-"$ROOT/outputs/${OBJECT}_$CASE"}

if [[ "$CASE" != "nc1" && "$CASE" != "nc4" ]]; then
  echo "Usage: ./run_example.sh [object] [nc1|nc4] [output-dir]" >&2
  exit 2
fi
SCENE="$ROOT/examples/$OBJECT/scene_$CASE.json"
if [[ ! -f "$SCENE" ]]; then
  echo "Unknown example object: $OBJECT" >&2
  exit 2
fi

cd "$ROOT"
env -u PYTHONPATH python -m free_range_gs.infer \
  --scene "$SCENE" \
  --dino-model "${FREE_RANGE_GS_DINO_MODEL:-facebook/dinov2-base}" \
  --output-dir "$OUTPUT" \
  --showcase
