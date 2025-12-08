"""
Command-line interface for TensorRT Upscaler.
"""

import argparse
import os
import sys
from pathlib import Path

from .upscaler import ImageUpscaler
from .animated import is_animated, AnimatedUpscaler

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def print_progress(current: int, total: int):
    """Print progress bar to console."""
    pct = int((current / total) * 100) if total > 0 else 0
    bar_len = 40
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r[{bar}] {pct}% ({current}/{total})", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT Image Upscaler - Minimal dependency super-resolution"
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="Input image file(s) or directory"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: same as input)"
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to ONNX model file"
    )

    parser.add_argument(
        "--tile-width",
        type=int,
        default=512,
        help="Tile width (default: 512)"
    )

    parser.add_argument(
        "--tile-height",
        type=int,
        default=512,
        help="Tile height (default: 512)"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=16,
        help="Tile overlap in pixels (default: 16)"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision"
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 precision (default)"
    )

    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 precision"
    )

    parser.add_argument(
        "--suffix",
        default="_upscaled",
        help="Output filename suffix (default: _upscaled)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="Output quality for lossy formats (default: 90)"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Collect input files
    input_files = []
    for path in args.input:
        if os.path.isfile(path):
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                input_files.append(path)
        elif os.path.isdir(path):
            if args.recursive:
                for root, _, files in os.walk(path):
                    for f in files:
                        if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                            input_files.append(os.path.join(root, f))
            else:
                for f in os.listdir(path):
                    if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                        input_files.append(os.path.join(path, f))

    if not input_files:
        print("No valid image files found.")
        sys.exit(1)

    # Validate model
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    # Create upscaler
    bf16 = args.bf16 and not args.no_bf16
    print(f"Loading model: {args.model}")
    print(f"Precision: {'FP16' if args.fp16 else ('BF16' if bf16 else 'FP32')}")
    print(f"Tile size: {args.tile_width}x{args.tile_height}, overlap: {args.overlap}")

    upscaler = ImageUpscaler(
        onnx_path=args.model,
        tile_size=(args.tile_width, args.tile_height),
        overlap=args.overlap,
        fp16=args.fp16,
        bf16=bf16,
    )

    animated_upscaler = AnimatedUpscaler(upscaler)

    # Process files
    total_files = len(input_files)
    print(f"\nProcessing {total_files} file(s)...")

    for i, input_path in enumerate(input_files):
        filename = os.path.basename(input_path)
        print(f"\n[{i+1}/{total_files}] {filename}")

        # Determine output path
        if args.output:
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.dirname(input_path)

        stem = Path(input_path).stem
        ext = Path(input_path).suffix

        # Check if animated
        if is_animated(input_path):
            output_path = os.path.join(output_dir, f"{stem}{args.suffix}{ext}")
            print(f"  Animated: {output_path}")
            animated_upscaler.upscale_animated(
                input_path,
                output_path,
                progress_callback=None if args.quiet else print_progress
            )
        else:
            output_path = os.path.join(output_dir, f"{stem}{args.suffix}.png")
            print(f"  Output: {output_path}")
            from .fast_io import load_image_fast, save_image_fast
            img, has_alpha = load_image_fast(input_path)
            result = upscaler.upscale_array(
                img,
                has_alpha=has_alpha,
                progress_callback=None if args.quiet else print_progress
            )
            save_image_fast(result, output_path, has_alpha)

        if not args.quiet:
            print()  # Newline after progress bar

    print(f"\nDone! Processed {total_files} file(s).")


if __name__ == "__main__":
    main()
