# Inference notes

## Environment and checkpoint

Use the virtual environment and installation commands in the README. A CUDA GPU
is required for reconstruction. If gsplat builds its extension on first use, a
compatible CUDA toolkit (`nvcc`), C++ compiler, and Ninja must be available;
PyTorch's CUDA runtime alone does not provide a compiler. See
[gsplat installation](https://docs.gsplat.studio/main/INSTALL.html) for matching
prebuilt packages and build instructions.

Complete the first CUDA extension build in a single process before parallel
inference. Concurrent first-use builds into the same cache can race. Use separate
`TORCH_EXTENSIONS_DIR` directories for different PyTorch/CUDA environments.

The released checkpoint includes DINOv2 encoder weights. Its architecture config
is bundled, so the default backbone needs no separate weight download.

For a wheel install or a checkpoint outside the code checkout:

```bash
free-range-gs-infer --scene path/to/scene.json \
  --checkpoint /path/to/frg_ckpt.pt --output-dir outputs/my_scene
```

## Scene format

Start with `examples/TOWER_TUMBLING/scene_nc4.json`. A scene contains:

- `schema_version: 1`;
- exactly four `inputs`, each with an image path and camera;
- one or more `renders`, each with a unique ID and camera;
- optionally, a `reference` image for each render camera.

Image paths are relative to the scene JSON and stay inside that directory.
All images must already be **512×512**. RGBA is alpha-composited on black; RGB
images retain their existing background. Foreground segmentation and camera-pose
estimation are not part of this release.

Each camera needs a 3×3 pixel-coordinate intrinsic matrix and exactly one 4×4
`world_to_camera` or `camera_to_world` matrix. Poses use the OpenCV convention:
`p_camera = R p_world + t`, with +Z forward and +Y down. Rotations must be proper
orthonormal matrices. Keep all cameras and the object in a consistent normalized
coordinate frame; use the bundled object-centric scenes as a scale reference.

For a resize from `(W, H)` to `(W', H')`, use
`K' = diag(W'/W, H'/H, 1) K`. For a crop starting at `(x0, y0)`, first subtract
`x0` and `y0` from the principal point, then apply the resize. The loader rejects
size mismatches to avoid silently changing images without updating calibration.

Validate without a GPU or weights:

```bash
python -m free_range_gs.infer --scene path/to/scene.json --validate-scene
```

## Sampler and guidance

| Setting | Released default |
| --- | --- |
| Input views / resolution | 4 / 512×512 |
| Gaussian count / patch size | 8,192 / 2 |
| Euler steps | 50 |
| Seed | 42 |
| Classifier-free guidance scale | 2.0 |
| Additional photometric guidance scale | 50.0 |
| Render background | Black |

Rendering-gradient **conditioning** is active in the checkpoint's model. The
additional gradient correction to the Euler update is controlled separately by
`--render-grads-scale` (alias `--photometric-guidance-scale`). The default
λPG=50 matches the paper and the v52 evaluation script:

```bash
python -m free_range_gs.infer \
  --scene examples/TOWER_TUMBLING/scene_nc4.json \
  --output-dir outputs/tower_pg50 --render-grads-scale 50 --showcase
```

The CLI uses λPG=50 by default. Guidance changes affect results and runtime.
The released sampler also retains its evaluation snapshot's
camera-feature encoding and near-end velocity denominator floor of 0.05; this
command alone does not establish reproduction of the paper's reported metrics.

## Outputs

- `reconstruction.ply`: 3DGS-compatible parameters, including log-scales,
  logit-opacity, quaternions, and DC spherical-harmonic color coefficients.
- `renders/*.png`, or `showcase.png` with `--showcase`.
- `run.json`: checkpoint SHA-256, seed, sampler/guidance options, software versions,
  GPU, reconstruction time, peak allocated GPU memory, and artifact paths.

The showcase layout expects `input_00`…`input_03`, followed by
`validation_00`…`validation_03`, with references for all validation cameras.
For custom cameras, omit `--showcase` to save individual renders. Set external
PLY viewers to a black background for comparable visualizations.
