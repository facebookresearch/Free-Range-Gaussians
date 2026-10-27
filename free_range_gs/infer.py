from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .renderer import render_gaussians
from .runtime import (
    checkpoint_path,
    load_config,
    load_model,
    prepare_output,
    require_cuda,
    seed_everything,
)
from .scene import load_scene


DEFAULTS = load_config().inference.to_dict()


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    image = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image, "RGB").save(path)


def validate_showcase(scene) -> None:
    if len(scene.render_ids) != 8:
        raise ValueError("showcase scenes must render four input and four validation cameras")
    expected_inputs = [f"input_{index:02d}" for index in range(4)]
    expected_validation = [f"validation_{index:02d}" for index in range(4)]
    if scene.render_ids[:4] != expected_inputs or scene.render_ids[4:] != expected_validation:
        raise ValueError("showcase render IDs must be input_00..03 followed by validation_00..03")
    if any(reference is None for reference in scene.render_references[4:]):
        raise ValueError("showcase validation cameras must provide reference images")


def save_showcase(path: Path, scene, predictions: np.ndarray) -> None:
    validate_showcase(scene)
    inputs = scene.images[0].permute(0, 2, 3, 1).cpu().numpy()
    references = [*inputs, *scene.render_references[4:]]
    tile_size = int(predictions.shape[1])
    border, gap, margin = 7, 8, 14
    title_height, group_height, row_gap = 42, 32, 10
    framed = tile_size + 2 * border
    width = 2 * margin + 8 * framed + 7 * gap
    height = 2 * margin + title_height + group_height + 2 * framed + row_gap
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.load_default(size=28)
        group_font = ImageFont.load_default(size=22)
    except TypeError:
        title_font = group_font = ImageFont.load_default()

    def centered_text(bounds, label, font):
        x0, y0, x1, y1 = bounds
        box = draw.textbbox((0, 0), label, font=font)
        x = x0 + (x1 - x0 - (box[2] - box[0])) / 2
        y = y0 + (y1 - y0 - (box[3] - box[1])) / 2
        draw.text((x, y), label, fill=(245, 245, 245), font=font)

    centered_text(
        (margin, margin, width - margin, margin + title_height),
        "TOP: REFERENCE VIEWS    |    BOTTOM: MODEL PREDICTIONS",
        title_font,
    )
    grid_y = margin + title_height + group_height
    input_color = (22, 92, 54)
    validation_color = (120, 38, 47)
    group_width = 4 * framed + 3 * gap
    input_bounds = (margin, margin + title_height, margin + group_width, grid_y)
    validation_x = margin + 4 * (framed + gap)
    validation_bounds = (validation_x, margin + title_height, validation_x + group_width, grid_y)
    draw.rectangle(input_bounds, fill=input_color)
    draw.rectangle(validation_bounds, fill=validation_color)
    centered_text(input_bounds, "INPUT VIEWS", group_font)
    centered_text(validation_bounds, "HELD-OUT VALIDATION VIEWS", group_font)

    for row, images in enumerate((references, predictions)):
        y = grid_y + row * (framed + row_gap)
        for column, values in enumerate(images):
            x = margin + column * (framed + gap)
            color = input_color if column < 4 else validation_color
            draw.rectangle((x, y, x + framed - 1, y + framed - 1), fill=color)
            tile = Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8), "RGB")
            canvas.paste(tile, (x + border, y + border))

    canvas.save(path)


def execute(args: argparse.Namespace) -> dict:
    if args.euler_steps <= 0:
        raise ValueError("euler-steps must be positive")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 0:
        raise ValueError("cfg-scale must be finite and non-negative")
    if not math.isfinite(args.render_grads_scale) or args.render_grads_scale < 0:
        raise ValueError("render-grads-scale must be finite and non-negative")
    scene = load_scene(args.scene, DEFAULTS["image_size"], expected_views=DEFAULTS["num_views"])
    if args.showcase:
        validate_showcase(scene)
    checkpoint = checkpoint_path(args.checkpoint)
    requested_output = Path(args.output_dir).expanduser().resolve()
    protected = (scene.scene_path, checkpoint.resolve(), Path(__file__).resolve().parent)
    if any(requested_output == path or requested_output in path.parents for path in protected):
        raise ValueError("output-dir must not contain the scene, checkpoint, or inference package")
    if requested_output.exists():
        if not requested_output.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {requested_output}")
        if any(requested_output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {requested_output}; pass --overwrite to replace it")
    device = require_cuda(args.device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(args.seed)

    print(f"Using checkpoint: {checkpoint}", flush=True)
    print(f"Loading DINOv2 and Free-Range Gaussians on {device}", flush=True)
    model = load_model(
        checkpoint,
        args.dino_model,
        device,
        DEFAULTS["num_gaussians"],
        DEFAULTS["image_size"],
    )[0]

    print(f"Loading scene: {args.scene}", flush=True)
    images = scene.images.to(device)
    input_intrinsics = scene.input_intrinsics.to(device)
    input_extrinsics = scene.input_world_to_camera.to(device)
    render_intrinsics = scene.render_intrinsics.to(device)
    render_extrinsics = scene.render_world_to_camera.to(device)
    print(
        f"Reconstructing {DEFAULTS['num_gaussians']} Gaussians with {args.euler_steps} Euler steps",
        flush=True,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    gaussians = model.reconstruct(
        images,
        input_intrinsics,
        input_extrinsics,
        euler_steps=args.euler_steps,
        cfg_scale=args.cfg_scale,
        render_grads_scale=args.render_grads_scale,
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    for name in ("positions", "scales", "rotations", "opacities", "colors"):
        if not torch.isfinite(getattr(gaussians, name)).all():
            raise RuntimeError(f"reconstruction produced non-finite {name}")
    print(f"Rendering {len(scene.render_ids)} views", flush=True)
    with torch.no_grad():
        colors, _ = render_gaussians(
            gaussians,
            render_intrinsics,
            render_extrinsics,
            DEFAULTS["image_size"],
            DEFAULTS["image_size"],
            "black",
        )

    output = prepare_output(requested_output, args.overwrite)
    gaussian_path = output / "reconstruction.ply"
    gaussians.save(gaussian_path)
    predictions = colors[0].float().cpu().numpy()
    if args.showcase:
        showcase_path = output / "showcase.png"
        save_showcase(showcase_path, scene, predictions)
        render_paths = [str(showcase_path)]
    else:
        render_dir = output / "renders"
        render_dir.mkdir()
        render_paths = []
        for index, render_id in enumerate(scene.render_ids):
            render_path = render_dir / f"{render_id}.png"
            save_rgb(render_path, predictions[index])
            render_paths.append(str(render_path))
    checksum = hashlib.sha256()
    with checkpoint.open("rb") as reader:
        while block := reader.read(8 * 1024 * 1024):
            checksum.update(block)
    record = {
        "scene": str(scene.scene_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checksum.hexdigest(),
        "seed": args.seed,
        "euler_steps": args.euler_steps,
        "cfg_scale": args.cfg_scale,
        "render_grads_scale": args.render_grads_scale,
        "background": "black",
        "image_size": DEFAULTS["image_size"],
        "num_gaussians": DEFAULTS["num_gaussians"],
        "device": torch.cuda.get_device_name(device),
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "gsplat", "numpy", "Pillow")
        },
        "ply": str(gaussian_path),
        "renders": render_paths,
    }
    run_path = output / "run.json"
    run_path.write_text(json.dumps(record, indent=2) + "\n")
    return {"ply": str(gaussian_path), "renders": render_paths, "run": str(run_path)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Standalone Free-Range Gaussians v52 inference")
    result.add_argument("--scene", required=True, help="Path to scene.json")
    result.add_argument("--output-dir", help="Directory for reconstruction and render outputs")
    result.add_argument("--checkpoint", help="Override the checkpoint file path")
    result.add_argument(
        "--validate-scene", action="store_true",
        help="Validate scene inputs on CPU without loading model weights",
    )
    result.add_argument("--dino-model", default="facebook/dinov2-base", help="HF model ID or local DINOv2 directory")
    result.add_argument("--euler-steps", type=int, default=DEFAULTS["euler_steps"])
    result.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    result.add_argument("--cfg-scale", type=float, default=DEFAULTS["cfg_scale"])
    result.add_argument(
        "--render-grads-scale", "--photometric-guidance-scale", type=float,
        default=DEFAULTS["render_grads_scale"],
        help="Additional photometric guidance scale; default 50 matches the paper",
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--showcase", action="store_true", help="save one labeled reference/prediction grid")
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.validate_scene:
        scene = load_scene(args.scene, DEFAULTS["image_size"], expected_views=DEFAULTS["num_views"])
        if args.showcase:
            validate_showcase(scene)
        print(
            f"Valid scene: {scene.name}; {scene.images.shape[1]} input views, "
            f"{len(scene.render_ids)} render cameras; black background"
        )
        return
    if not args.output_dir:
        argument_parser.error("--output-dir is required for inference")
    artifacts = execute(args)
    print(f"Saved {artifacts['ply']}")
    for path in artifacts["renders"]:
        print(f"Saved {path}")
    print(f"Saved {artifacts['run']}")


if __name__ == "__main__":
    main()
