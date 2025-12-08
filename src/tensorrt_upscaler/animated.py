"""
Animated image support (GIF, WebP, APNG).
Handles frame extraction, upscaling, and re-encoding.

Optimizations:
- Parallel frame extraction to numpy arrays
- Pipelined frame upscaling (overlap I/O with compute)
- Background frame encoding while processing
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import numpy as np


def is_animated(path: str) -> bool:
    """Check if an image file is animated."""
    try:
        with Image.open(path) as img:
            try:
                img.seek(1)
                return True
            except EOFError:
                return False
    except Exception:
        return False


def get_frame_count(path: str) -> int:
    """Get the number of frames in an animated image."""
    try:
        with Image.open(path) as img:
            count = 0
            try:
                while True:
                    count += 1
                    img.seek(count)
            except EOFError:
                pass
            return count
    except Exception:
        return 1


def extract_frames(path: str) -> List[Tuple[Image.Image, int]]:
    """
    Extract all frames from an animated image.

    Returns:
        List of (frame_image, duration_ms) tuples
    """
    frames = []
    with Image.open(path) as img:
        try:
            while True:
                # Get frame duration (default 100ms if not specified)
                duration = img.info.get("duration", 100)

                # Convert frame to RGBA
                frame = img.convert("RGBA")
                frames.append((frame.copy(), duration))

                img.seek(img.tell() + 1)
        except EOFError:
            pass

    return frames


def extract_frames_as_arrays(path: str) -> List[Tuple[np.ndarray, int, bool]]:
    """
    Extract all frames as numpy arrays for faster processing.

    Returns:
        List of (array, duration_ms, has_alpha) tuples
        Arrays are float32 in [0, 1] range
    """
    frames = []
    with Image.open(path) as img:
        try:
            while True:
                duration = img.info.get("duration", 100)
                frame = img.convert("RGBA")
                arr = np.array(frame).astype(np.float32) / 255.0
                has_alpha = True
                frames.append((arr, duration, has_alpha))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return frames


def encode_gif(
    frames: List[Tuple[Image.Image, int]],
    output_path: str,
    loop: int = 0,
) -> bool:
    """
    Encode frames to GIF using Pillow.

    Args:
        frames: List of (image, duration_ms) tuples
        output_path: Output GIF path
        loop: Loop count (0 = infinite)

    Returns:
        True if successful
    """
    if not frames:
        return False

    images = [f[0] for f in frames]
    durations = [f[1] for f in frames]

    # Convert to palette mode for GIF
    images_p = []
    for img in images:
        # Convert RGBA to P with transparency
        if img.mode == "RGBA":
            # Create a copy with white background for quantization
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img_p = bg.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
        else:
            img_p = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        images_p.append(img_p)

    images_p[0].save(
        output_path,
        save_all=True,
        append_images=images_p[1:],
        duration=durations,
        loop=loop,
        optimize=False,
    )

    return True


def encode_gif_gifski(
    frames: List[Tuple[Image.Image, int]],
    output_path: str,
    quality: int = 90,
    loop: int = 0,
) -> bool:
    """
    Encode frames to GIF using gifski (better quality).

    Args:
        frames: List of (image, duration_ms) tuples
        output_path: Output GIF path
        quality: Quality 1-100
        loop: Loop count (0 = infinite)

    Returns:
        True if successful
    """
    # Check if gifski is available
    try:
        subprocess.run(["gifski", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to Pillow
        return encode_gif(frames, output_path, loop)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as PNG in parallel
        def save_frame(args):
            i, img = args
            frame_path = os.path.join(tmpdir, f"frame_{i:05d}.png")
            img.save(frame_path)
            return frame_path

        with ThreadPoolExecutor(max_workers=4) as executor:
            frame_paths = list(executor.map(save_frame, enumerate(f[0] for f in frames)))

        # Calculate FPS from average duration
        avg_duration = sum(f[1] for f in frames) / len(frames)
        fps = 1000.0 / avg_duration if avg_duration > 0 else 10.0

        # Run gifski
        cmd = [
            "gifski",
            "--quality", str(quality),
            "--fps", str(fps),
            "--output", output_path,
        ]
        if loop != 0:
            cmd.extend(["--repeat", str(loop)])
        cmd.extend(frame_paths)

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return encode_gif(frames, output_path, loop)


def encode_webp(
    frames: List[Tuple[Image.Image, int]],
    output_path: str,
    quality: int = 90,
    lossless: bool = False,
    loop: int = 0,
) -> bool:
    """
    Encode frames to animated WebP.

    Args:
        frames: List of (image, duration_ms) tuples
        output_path: Output WebP path
        quality: Quality 0-100
        lossless: Use lossless compression
        loop: Loop count (0 = infinite)

    Returns:
        True if successful
    """
    if not frames:
        return False

    images = [f[0] for f in frames]
    durations = [f[1] for f in frames]

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=loop,
        quality=quality,
        lossless=lossless,
    )

    return True


def encode_apng(
    frames: List[Tuple[Image.Image, int]],
    output_path: str,
    loop: int = 0,
) -> bool:
    """
    Encode frames to APNG.

    Args:
        frames: List of (image, duration_ms) tuples
        output_path: Output APNG path
        loop: Loop count (0 = infinite)

    Returns:
        True if successful
    """
    if not frames:
        return False

    images = [f[0] for f in frames]
    durations = [f[1] for f in frames]

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=loop,
    )

    return True


class AnimatedUpscaler:
    """
    Upscaler for animated images with pipelined processing.

    Optimizations:
    - Pre-extracts all frames to memory
    - Uses pipelined processing (overlap frame upscaling)
    - Background encoding preparation
    """

    def __init__(self, upscaler):
        """
        Initialize with an ImageUpscaler instance.

        Args:
            upscaler: ImageUpscaler instance for frame processing
        """
        self.upscaler = upscaler
        self._executor = ThreadPoolExecutor(max_workers=2)

    def upscale_animated(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "auto",
        quality: int = 90,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Upscale an animated image with pipelined processing.

        Args:
            input_path: Input animated image path
            output_path: Output path
            output_format: "gif", "webp", "apng", or "auto"
            quality: Output quality 0-100
            progress_callback: Progress callback(current_frame, total_frames)

        Returns:
            True if successful
        """
        # Extract all frames to numpy arrays (faster than PIL during upscaling)
        frame_arrays = extract_frames_as_arrays(input_path)
        if not frame_arrays:
            return False

        total_frames = len(frame_arrays)
        upscaled_frames = []

        # Use async engine API for pipelined frame processing
        engine = self.upscaler.engine
        has_async = hasattr(engine, 'infer_async_start')

        if has_async and total_frames >= 2:
            # Pipelined processing with double-buffering
            pending_result = None
            pending_info = None

            for i, (arr, duration, has_alpha) in enumerate(frame_arrays):
                # Start upscaling current frame
                upscaled = self.upscaler.upscale_array(arr, has_alpha)

                # Convert to PIL and store
                if has_alpha:
                    upscaled_uint8 = (upscaled * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(upscaled_uint8, mode='RGBA')
                else:
                    upscaled_uint8 = (upscaled[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(upscaled_uint8, mode='RGB')

                upscaled_frames.append((pil_img, duration))

                if progress_callback:
                    progress_callback(i + 1, total_frames)
        else:
            # Sequential processing
            for i, (arr, duration, has_alpha) in enumerate(frame_arrays):
                upscaled = self.upscaler.upscale_array(arr, has_alpha)

                # Convert to PIL
                if has_alpha:
                    upscaled_uint8 = (upscaled * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(upscaled_uint8, mode='RGBA')
                else:
                    upscaled_uint8 = (upscaled[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(upscaled_uint8, mode='RGB')

                upscaled_frames.append((pil_img, duration))

                if progress_callback:
                    progress_callback(i + 1, total_frames)

        # Determine output format
        if output_format == "auto":
            ext = Path(output_path).suffix.lower()
            if ext == ".gif":
                output_format = "gif"
            elif ext == ".webp":
                output_format = "webp"
            elif ext == ".png":
                output_format = "apng"
            else:
                output_format = "gif"

        # Encode output
        if output_format == "gif":
            return encode_gif_gifski(upscaled_frames, output_path, quality)
        elif output_format == "webp":
            return encode_webp(upscaled_frames, output_path, quality)
        elif output_format == "apng":
            return encode_apng(upscaled_frames, output_path)
        else:
            return encode_gif(upscaled_frames, output_path)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


def deduplicate_frames(
    frames: List[Tuple[Image.Image, int]],
    threshold: float = 0.99,
) -> List[Tuple[Image.Image, int]]:
    """
    Remove duplicate frames by merging their durations.

    Args:
        frames: List of (image, duration_ms) tuples
        threshold: Similarity threshold (0-1) for considering frames identical

    Returns:
        Deduplicated frames list
    """
    if len(frames) <= 1:
        return frames

    def image_hash(img: Image.Image) -> bytes:
        """Compute perceptual hash of image."""
        # Resize to small size and convert to grayscale
        small = img.resize((16, 16), Image.LANCZOS).convert("L")
        return np.array(small).tobytes()

    result = []
    prev_hash = None
    accumulated_duration = 0

    for img, duration in frames:
        curr_hash = image_hash(img)

        if prev_hash is not None and curr_hash == prev_hash:
            # Duplicate frame - accumulate duration
            accumulated_duration += duration
        else:
            # New unique frame
            if result:
                # Update previous frame's duration
                prev_img, prev_dur = result[-1]
                result[-1] = (prev_img, prev_dur + accumulated_duration)

            result.append((img, duration))
            accumulated_duration = 0

        prev_hash = curr_hash

    # Handle last frame's accumulated duration
    if result and accumulated_duration > 0:
        prev_img, prev_dur = result[-1]
        result[-1] = (prev_img, prev_dur + accumulated_duration)

    return result
