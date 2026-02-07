"""
Animated image encoding functions.

Supports encoding to GIF (Pillow/gifski), WebP, APNG, and AVIF formats.
Extracted from animated.py to keep AnimatedUpscaler focused on the upscaling pipeline.
"""

import os
import subprocess
import tempfile
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import numpy as np


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


def _rgb_to_yuv444(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB to YUV444 (BT.601 full range).

    Args:
        rgb: RGB array [H, W, 3] uint8

    Returns:
        (Y, U, V) planes as uint8 arrays
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # BT.601 full range
    y = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
    u = (-0.169 * r - 0.331 * g + 0.500 * b + 128).clip(0, 255).astype(np.uint8)
    v = (0.500 * r - 0.419 * g - 0.081 * b + 128).clip(0, 255).astype(np.uint8)

    return y, u, v


def _build_y4m_stream(
    frames: List[Tuple[Image.Image, int]],
    has_alpha: bool,
) -> bytes:
    """
    Build Y4M byte stream from PIL frames.

    Args:
        frames: List of (image, duration_ms) tuples
        has_alpha: Whether to include alpha plane (C444alpha vs C444)

    Returns:
        Y4M byte stream
    """
    if not frames:
        return b''

    first_img = frames[0][0]
    width, height = first_img.size

    # Calculate FPS from average duration
    avg_duration = sum(f[1] for f in frames) / len(frames)
    # Use timescale of 1000 for millisecond precision
    timescale = 1000
    fps_num = timescale
    fps_den = int(avg_duration) if avg_duration > 0 else 100

    # Y4M header - C444alpha supports alpha plane
    colorspace = "C444alpha" if has_alpha else "C444"
    header = f"YUV4MPEG2 W{width} H{height} F{fps_num}:{fps_den} Ip A1:1 {colorspace}\n"

    chunks = [header.encode()]

    for img, duration in frames:
        # Convert to RGB/RGBA
        if has_alpha:
            rgba = np.array(img.convert("RGBA"))
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3]
        else:
            rgb = np.array(img.convert("RGB"))
            alpha = None

        # Convert RGB to YUV
        y, u, v = _rgb_to_yuv444(rgb)

        # Frame header
        chunks.append(b"FRAME\n")

        # Y, U, V planes (row-major order)
        chunks.append(y.tobytes())
        chunks.append(u.tobytes())
        chunks.append(v.tobytes())

        # Alpha plane if present
        if has_alpha and alpha is not None:
            chunks.append(alpha.tobytes())

    return b''.join(chunks)


def encode_avif(
    frames: List[Tuple[Image.Image, int]],
    output_path: str,
    lossless: bool = False,
    color_quality: int = 80,
    alpha_quality: int = 90,
    speed: int = 6,
    loop: int = 0,
) -> bool:
    """
    Encode frames to animated AVIF using avifenc with Y4M stdin pipe.

    Uses Y4M C444alpha format piped to avifenc stdin - no temp files needed.

    Args:
        frames: List of (image, duration_ms) tuples
        output_path: Output AVIF path
        lossless: Use lossless compression
        color_quality: Color quality 0-100 (higher = better, ignored if lossless)
        alpha_quality: Alpha quality 0-100 (higher = better, ignored if lossless)
        speed: Encoding speed 0-10 (0=slowest/best, 10=fastest)
        loop: Loop count (0 = infinite)

    Returns:
        True if successful
    """
    if not frames:
        return False

    # Check if avifenc is available
    try:
        subprocess.run(["avifenc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("avifenc not found, falling back to WebP")
        webp_path = output_path.rsplit('.', 1)[0] + '.webp'
        return encode_webp(frames, webp_path, quality=color_quality)

    # Check if any frame has alpha
    has_alpha = any(f[0].mode == "RGBA" for f in frames)

    # Build Y4M stream
    y4m_data = _build_y4m_stream(frames, has_alpha)

    # Build avifenc command
    cmd = ["avifenc", "--stdin"]

    if lossless:
        cmd.append("--lossless")
    else:
        # Use new -q/--qcolor syntax (0-100 where 100 is lossless)
        cmd.extend(["-q", str(color_quality)])
        if has_alpha:
            cmd.extend(["--qalpha", str(alpha_quality)])

    cmd.extend(["--speed", str(speed)])

    if loop != 0:
        cmd.extend(["--repetition-count", str(loop)])

    # Output path
    cmd.append(output_path)

    try:
        subprocess.run(cmd, input=y4m_data, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"avifenc failed: {e.stderr.decode() if e.stderr else str(e)}")
        return False


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
