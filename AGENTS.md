# Working with Free-Range Gaussians

These instructions apply to agents working in this repository. Follow any
additional user or workspace policies.

## Read first

- Read `README.md` and `docs/USAGE.md` before preparing a scene or changing inference.
- This release provides inference for the supplied 8,192-Gaussian checkpoint.
  Training and full benchmark reproduction are separate tasks.
- Use the approved Hugging Face checkpoint. Keep weights, tokens, environments,
  and generated runs out of commits.
- Read `.github/CONTRIBUTING.md` for the submission workflow. The source of truth
  is the configured internal Meta repository; maintainers land changes internally
  and publish them through ShipIt. Community contributions use GitHub pull requests.

## Prepare and run

- Use a virtual environment and the documented PyTorch/CUDA setup.
- Complete one CUDA run before launching parallel jobs if gsplat needs to build
  its extension. Keep build caches separate between incompatible environments.
- Inputs must be four 512×512 RGB/RGBA views with calibrated OpenCV camera poses.
  Images must share the scene's coordinate frame and scale. Do not invent poses.
- If resizing or cropping inputs, adjust intrinsics too; see `docs/USAGE.md`.
- Validate first: `python -m free_range_gs.infer --scene path/to/scene.json --validate-scene`.
- Use a new output directory for each experiment. Use `--overwrite` only when
  replacing that directory is explicitly intended.
- Preserve seed, Euler steps, guidance settings, and checkpoint identity in reports.
  The CLI writes these to `run.json`.

## Rendering and comparisons

- Use a black background for renders, image grids, videos, and PLY viewer
  previews unless the user explicitly requests another background.
- RGBA inputs are composited on black. Opaque RGB inputs must already have the
  intended foreground/background; this code does not segment objects.
- Distinguish input views, held-out reference images, and model predictions.
  `--showcase` requires the bundled eight-camera layout with reference images.
- Use `run.json` to identify changes in method or settings. Five example objects
  do not establish reproduction of the paper's benchmark tables.

## Preserve checkpoint compatibility

- Keep the released Gaussian ordering/pairing, normalization, camera/ray encoding,
  and model architecture unchanged unless a method change is explicitly requested.
- Do not wrap reconstruction in `torch.inference_mode()`: rendering-gradient
  conditioning needs autograd even during inference.
- Keep the photometric-guidance default at 50, matching the paper and v52
  evaluation script. Record the setting in experiment reports.
- Do not silently change precision or disable gradient conditioning to speed up a run.
- After model/renderer changes, validate a bundled scene, run a CUDA example, and inspect
  finite outputs, Gaussian count, black backgrounds, and the recorded settings.
