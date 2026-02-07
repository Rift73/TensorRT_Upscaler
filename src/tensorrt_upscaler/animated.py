"""
Animated image support (GIF, WebP, APNG).
Handles frame extraction, upscaling, and re-encoding.

Optimizations:
- Parallel frame extraction to numpy arrays
- Pipelined frame upscaling (overlap I/O with compute)
- Background frame encoding while processing
"""

from pathlib import Path
from typing import Optional, Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import numpy as np

from .utils import is_animated, get_frame_count  # noqa: F401 - re-exported for backwards compat
from .animated_encoders import (  # noqa: F401 - re-exported for backwards compat
    encode_gif,
    encode_gif_gifski,
    encode_webp,
    encode_apng,
    encode_avif,
    deduplicate_frames,
)


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

    def _process_frame(
        self,
        arr: np.ndarray,
        has_alpha: bool,
        resize_array,
        compute_scaled_size,
        cas_sharpen_array,
        adaptive_sharpen_array,
        prescale_enabled: bool,
        prescale_mode: str,
        prescale_width: int,
        prescale_height: int,
        prescale_kernel: str,
        prescale_scale_factor: float,
        custom_res_enabled: bool,
        custom_res_mode: str,
        custom_res_width: int,
        custom_res_height: int,
        custom_res_keep_aspect: bool,
        custom_res_kernel: str,
        custom_res_scale_factor: float,
        sharpen_enabled: bool,
        sharpen_value: float,
        sharpen_method: str,
        sharpen_anime_mode: bool,
    ) -> Image.Image:
        """Process a single frame: pre-scale, upscale, custom res, sharpen, convert to PIL."""
        height, width = arr.shape[:2]

        # Pre-scale (before upscaling)
        if prescale_enabled:
            new_size = compute_scaled_size(
                width, height,
                prescale_mode, prescale_width, prescale_height,
                scale_factor=prescale_scale_factor,
            )
            arr = resize_array(arr, new_size, prescale_kernel, has_alpha)

        # Upscale
        upscaled = self.upscaler.upscale_array(arr, has_alpha)
        height, width = upscaled.shape[:2]

        # Custom resolution (after upscaling)
        if custom_res_enabled:
            new_size = compute_scaled_size(
                width, height,
                custom_res_mode, custom_res_width, custom_res_height,
                keep_aspect=custom_res_keep_aspect,
                scale_factor=custom_res_scale_factor,
            )
            upscaled = resize_array(upscaled, new_size, custom_res_kernel, has_alpha)

        # Sharpening
        if sharpen_enabled and sharpen_value > 0:
            if sharpen_method == 'adaptive':
                upscaled = adaptive_sharpen_array(
                    upscaled, sharpen_value, has_alpha,
                    overshoot_ctrl=False, anime_mode=sharpen_anime_mode
                )
            else:  # cas
                upscaled = cas_sharpen_array(upscaled, sharpen_value, has_alpha)

        # Convert to PIL
        if has_alpha:
            upscaled_uint8 = (upscaled * 255.0).clip(0, 255).astype(np.uint8)
            return Image.fromarray(upscaled_uint8, mode='RGBA')
        else:
            upscaled_uint8 = (upscaled[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
            return Image.fromarray(upscaled_uint8, mode='RGB')

    @staticmethod
    def _encode_frames(
        frames: List[Tuple[Image.Image, int]],
        output_path: str,
        output_format: str,
        quality: int,
    ) -> bool:
        """Determine output format and encode frames."""
        if output_format == "auto":
            ext = Path(output_path).suffix.lower()
            format_map = {".gif": "gif", ".webp": "webp", ".png": "apng", ".avif": "avif"}
            output_format = format_map.get(ext, "gif")

        if output_format == "gif":
            return encode_gif_gifski(frames, output_path, quality)
        elif output_format == "webp":
            return encode_webp(frames, output_path, quality)
        elif output_format == "apng":
            return encode_apng(frames, output_path)
        elif output_format == "avif":
            return encode_avif(frames, output_path, color_quality=quality)
        else:
            return encode_gif(frames, output_path)

    def upscale_animated(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "auto",
        quality: int = 90,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        # Resolution options
        prescale_enabled: bool = False,
        prescale_mode: str = "scale",
        prescale_width: int = 0,
        prescale_height: int = 0,
        prescale_kernel: str = "lanczos",
        prescale_scale_factor: float = 2.0,
        custom_res_enabled: bool = False,
        custom_res_mode: str = "scale",
        custom_res_width: int = 0,
        custom_res_height: int = 0,
        custom_res_keep_aspect: bool = True,
        custom_res_kernel: str = "lanczos",
        custom_res_scale_factor: float = 2.0,
        # Sharpening options
        sharpen_enabled: bool = False,
        sharpen_value: float = 0.0,
        sharpen_method: str = "cas",
        sharpen_anime_mode: bool = False,
    ) -> bool:
        """
        Upscale an animated image with pipelined processing.

        Args:
            input_path: Input animated image path
            output_path: Output path
            output_format: "gif", "webp", "apng", "avif", or "auto"
            quality: Output quality 0-100
            progress_callback: Progress callback(current_frame, total_frames)
            prescale_enabled: Enable pre-scaling before upscale
            prescale_mode: Pre-scale mode (scale/width/height/fit/fill)
            prescale_width: Pre-scale target width
            prescale_height: Pre-scale target height
            prescale_kernel: Pre-scale interpolation kernel
            custom_res_enabled: Enable custom resolution after upscale
            custom_res_mode: Custom res mode (scale/width/height/fit/fill)
            custom_res_width: Custom res target width
            custom_res_height: Custom res target height
            custom_res_keep_aspect: Keep aspect ratio for custom res
            custom_res_kernel: Custom res interpolation kernel
            sharpen_enabled: Enable sharpening
            sharpen_value: Sharpening strength
            sharpen_method: Sharpening method (cas/adaptive)
            sharpen_anime_mode: Use anime mode for adaptive sharpening

        Returns:
            True if successful
        """
        # Import resize and sharpen functions
        from .resize import resize_array, compute_scaled_size
        from .sharpening import cas_sharpen_array, adaptive_sharpen_array

        # Extract all frames to numpy arrays (faster than PIL during upscaling)
        frame_arrays = extract_frames_as_arrays(input_path)
        if not frame_arrays:
            return False

        total_frames = len(frame_arrays)
        upscaled_frames = []

        # Frame deduplication hash function
        def frame_hash(arr: np.ndarray) -> bytes:
            """Compute hash of frame for deduplication."""
            arr_uint8 = (arr[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr_uint8, mode='RGB')
            small = pil.resize((16, 16), Image.LANCZOS).convert('L')
            return np.array(small).tobytes()

        prev_hash = None
        skipped_duplicates = 0

        # Process each frame
        for i, (arr, duration, has_alpha) in enumerate(frame_arrays):
            # Check for duplicate frame
            curr_hash = frame_hash(arr)
            if prev_hash is not None and curr_hash == prev_hash:
                # Duplicate frame — merge duration into previous frame
                if upscaled_frames:
                    prev_img, prev_dur = upscaled_frames[-1]
                    upscaled_frames[-1] = (prev_img, prev_dur + duration)
                skipped_duplicates += 1
                if progress_callback:
                    progress_callback(i + 1, total_frames)
                prev_hash = curr_hash
                continue

            pil_img = self._process_frame(
                arr, has_alpha,
                resize_array, compute_scaled_size,
                cas_sharpen_array, adaptive_sharpen_array,
                prescale_enabled, prescale_mode, prescale_width, prescale_height,
                prescale_kernel, prescale_scale_factor,
                custom_res_enabled, custom_res_mode, custom_res_width, custom_res_height,
                custom_res_keep_aspect, custom_res_kernel, custom_res_scale_factor,
                sharpen_enabled, sharpen_value, sharpen_method, sharpen_anime_mode,
            )

            upscaled_frames.append((pil_img, duration))
            prev_hash = curr_hash

            if progress_callback:
                progress_callback(i + 1, total_frames)

        if skipped_duplicates > 0:
            print(f"Deduplicated {skipped_duplicates} frames ({total_frames} -> {len(upscaled_frames)} frames)")

        return self._encode_frames(upscaled_frames, output_path, output_format, quality)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
