"""
Image upscaler with tiling support and alpha channel handling.
Uses direct TensorRT inference - no VapourSynth dependency.

Optimizations:
- Pre-allocated blend weight cache
- In-place array operations
- Optimized tile blending with cached weights
- Reduced memory allocations during processing
- Double-buffered async tile processing (overlapped compute/transfer)
- Parallel tile extraction using thread pool
- Numba-accelerated blend weight generation
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor

from .engine import TensorRTEngine
from .dml_engine import DirectMLEngine, is_directml_available
from .pytorch_engine import PyTorchEngine, is_pytorch_available
from .fast_io import load_image_fast, save_image_fast, CV2_AVAILABLE

# Try to import Numba for JIT acceleration
NUMBA_AVAILABLE = False
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    pass


# Numba-accelerated blend weight application
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _apply_weighted_blend_numba(
        output: np.ndarray,
        weights: np.ndarray,
        result: np.ndarray,
        weight: np.ndarray,
        out_y: int,
        out_x: int,
        out_th: int,
        out_tw: int,
    ):
        """Numba-accelerated weighted blend of tile into output."""
        for y in prange(out_th):
            for x in range(out_tw):
                w = weight[y, x, 0]
                for c in range(3):
                    output[out_y + y, out_x + x, c] += result[y, x, c] * w
                weights[out_y + y, out_x + x, 0] += w

    @njit(parallel=True, fastmath=True, cache=True)
    def _normalize_output_numba(output: np.ndarray, weights: np.ndarray):
        """Numba-accelerated output normalization."""
        h, w, c = output.shape
        for y in prange(h):
            for x in range(w):
                wt = weights[y, x, 0]
                if wt > 1e-8:
                    inv_wt = 1.0 / wt
                    for ch in range(c):
                        output[y, x, ch] *= inv_wt

    @njit(fastmath=True, cache=True)
    def _create_blend_weight_numba(h: int, w: int, overlap_scaled: int) -> np.ndarray:
        """Numba-accelerated blend weight creation."""
        weight = np.ones((h, w, 1), dtype=np.float32)

        if overlap_scaled > 0:
            # Create linear ramps
            for i in range(overlap_scaled):
                ramp_val = i / overlap_scaled

                # Horizontal feather - left edge
                for y in range(h):
                    weight[y, i, 0] *= ramp_val

                # Horizontal feather - right edge
                for y in range(h):
                    weight[y, w - 1 - i, 0] *= ramp_val

                # Vertical feather - top edge
                for x in range(w):
                    weight[i, x, 0] *= ramp_val

                # Vertical feather - bottom edge
                for x in range(w):
                    weight[h - 1 - i, x, 0] *= ramp_val

        return weight


class ImageUpscaler:
    """
    High-level image upscaler with tiling and alpha support.

    Features:
    - Automatic tiling for large images (VRAM efficiency)
    - Overlap blending for seamless tile boundaries
    - Alpha channel preservation
    - Progress callbacks
    - Double-buffered async tile processing
    """

    TILE_ALIGNMENT = 64  # Tiles must be multiples of 64

    def __init__(
        self,
        onnx_path: str = "",
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 16,
        fp16: bool = False,
        bf16: bool = True,
        tf32: bool = False,
        backend: str = "tensorrt",
        disable_tile_limit: bool = False,
        # PyTorch-specific options
        pytorch_model_path: str = "",
        pytorch_device: str = "cuda",
        pytorch_half: bool = False,
        pytorch_bf16: bool = True,
        pytorch_vram_mode: str = "normal",
        # PyTorch optimization options
        pytorch_enable_tf32: bool = True,
        pytorch_channels_last: bool = True,
    ):
        """
        Initialize upscaler with TensorRT, DirectML, or PyTorch engine.

        Args:
            onnx_path: Path to ONNX super-resolution model (for TensorRT/DirectML)
            tile_size: Tile size (width, height) for processing - also used as max engine shape
            overlap: Overlap between tiles in pixels
            fp16: Enable FP16 precision (TensorRT/DirectML)
            bf16: Enable BF16 precision (default, TensorRT only)
            tf32: Enable TF32 precision (TensorRT only)
            backend: "tensorrt", "directml", or "pytorch"
            disable_tile_limit: When True, skip tile alignment to 64 and padding
            pytorch_model_path: Path to .pth/.safetensors model (for PyTorch)
            pytorch_device: Device for PyTorch ("cuda", "cpu")
            pytorch_half: Use FP16 for PyTorch
            pytorch_bf16: Use BF16 for PyTorch (Ampere+ GPUs)
            pytorch_vram_mode: VRAM mode for PyTorch ("normal", "auto", "low_vram")
            pytorch_enable_tf32: Enable TF32 for matmuls/convolutions (Ampere+)
            pytorch_channels_last: Use NHWC memory format (faster for CNNs)
        """
        self.tile_w, self.tile_h = tile_size
        self.overlap = overlap
        self.backend = backend
        self.disable_tile_limit = disable_tile_limit

        # Align tile size to 64 (unless disabled or using PyTorch)
        # PyTorch models are more flexible with input sizes
        if not disable_tile_limit and backend != "pytorch":
            self.tile_w = (self.tile_w // self.TILE_ALIGNMENT) * self.TILE_ALIGNMENT
            self.tile_h = (self.tile_h // self.TILE_ALIGNMENT) * self.TILE_ALIGNMENT

        if backend == "pytorch":
            if not is_pytorch_available():
                raise RuntimeError(
                    "PyTorch backend requested but not available. "
                    "Install with: pip install torch spandrel spandrel_extra_arches"
                )
            self.engine = PyTorchEngine(
                model_path=pytorch_model_path,
                device=pytorch_device,
                half=pytorch_half,
                bf16=pytorch_bf16,
                vram_mode=pytorch_vram_mode,
                enable_tf32=pytorch_enable_tf32,
                channels_last=pytorch_channels_last,
            )
        elif backend == "directml":
            if not is_directml_available():
                raise RuntimeError(
                    "DirectML backend requested but not available. "
                    "Install with: pip install onnxruntime-directml"
                )
            self.engine = DirectMLEngine(
                onnx_path=onnx_path,
                fp16=fp16,
            )
        else:
            # Default to TensorRT
            # Use tile size as max shape (separate width and height)
            max_shape = (1, 3, self.tile_h, self.tile_w)
            opt_shape = (1, 3, self.tile_h, self.tile_w)
            min_shape = (1, 3, 64, 64)

            self.engine = TensorRTEngine(
                onnx_path=onnx_path,
                fp16=fp16,
                bf16=bf16,
                tf32=tf32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            )

        self.scale = self.engine.model_scale

        # Cache for blend weights (key: (w, h))
        self._blend_weight_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Thread pool for parallel tile extraction
        self._tile_executor = ThreadPoolExecutor(max_workers=4)

        # Pre-warm blend weight cache for common tile size
        self._create_blend_weight(self.tile_w * self.scale, self.tile_h * self.scale)

    def upscale_array(
        self,
        img: np.ndarray,
        has_alpha: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Upscale a numpy array image.

        Args:
            img: Input image as numpy array (H, W, C) in [0, 1] range
            has_alpha: Whether image has alpha channel (4 channels)
            progress_callback: Optional callback(current_tile, total_tiles)

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        if has_alpha and img.shape[2] == 4:
            # Split RGB and alpha
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4]

            # Upscale RGB
            rgb_up = self._upscale_array_rgb(rgb, progress_callback)

            # Upscale alpha (as single channel - more efficient)
            # Replicate to 3 channels for model compatibility
            alpha_rgb = np.concatenate([alpha, alpha, alpha], axis=2)
            alpha_up = self._upscale_array_rgb(alpha_rgb, None)
            alpha_up = alpha_up[:, :, 0:1]  # Take first channel

            # Merge back
            return np.concatenate([rgb_up, alpha_up], axis=2)
        else:
            return self._upscale_array_rgb(img[:, :, :3], progress_callback)

    def _upscale_array_rgb(
        self,
        img: np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Upscale RGB numpy array with tiling and double-buffered async."""
        height, width = img.shape[:2]

        # Check if tiling is needed
        if width <= self.tile_w and height <= self.tile_h:
            arr = self._prepare_tile(img)
            result = self.engine.infer(arr)
            result = self._unpad_result(result, width * self.scale, height * self.scale)
            if progress_callback:
                progress_callback(1, 1)
            return result

        # Calculate tiles
        tiles = self._calculate_tiles(width, height)
        total_tiles = len(tiles)

        # Prepare output array
        out_h = height * self.scale
        out_w = width * self.scale
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Process tiles with double-buffered async execution
        if total_tiles >= 2 and hasattr(self.engine, 'infer_async_start'):
            # Extract and start first tile
            x0, y0, tw0, th0 = tiles[0]
            tile0 = self._extract_tile(img, x0, y0, tw0, th0)
            tile0 = self._prepare_tile(tile0)
            buf_idx0, out_shape0 = self.engine.infer_async_start(tile0, buf_idx=0)

            for i in range(1, total_tiles):
                # Extract and start next tile on alternate buffer
                xi, yi, twi, thi = tiles[i]
                tile_i = self._extract_tile(img, xi, yi, twi, thi)
                tile_i = self._prepare_tile(tile_i)
                buf_idx_i, out_shape_i = self.engine.infer_async_start(tile_i, buf_idx=i % 2)

                # Wait for previous tile and blend
                if i == 1:
                    result0 = self.engine.infer_async_wait(buf_idx0, out_shape0)
                    result0 = self._unpad_result(result0, tw0 * self.scale, th0 * self.scale)
                    self._blend_tile(output, weights, result0, x0, y0, tw0, th0)
                    if progress_callback:
                        progress_callback(1, total_tiles)
                else:
                    prev_x, prev_y, prev_tw, prev_th = tiles[i-1]
                    prev_buf = (i-1) % 2
                    prev_result = self.engine.infer_async_wait(prev_buf, out_shape_prev)
                    prev_result = self._unpad_result(prev_result, prev_tw * self.scale, prev_th * self.scale)
                    self._blend_tile(output, weights, prev_result, prev_x, prev_y, prev_tw, prev_th)
                    if progress_callback:
                        progress_callback(i, total_tiles)

                out_shape_prev = out_shape_i

            # Wait for last tile
            last_i = total_tiles - 1
            last_x, last_y, last_tw, last_th = tiles[last_i]
            last_buf = last_i % 2
            last_result = self.engine.infer_async_wait(last_buf, out_shape_prev if total_tiles > 1 else out_shape0)
            last_result = self._unpad_result(last_result, last_tw * self.scale, last_th * self.scale)
            self._blend_tile(output, weights, last_result, last_x, last_y, last_tw, last_th)
            if progress_callback:
                progress_callback(total_tiles, total_tiles)

        else:
            # Fallback: sequential processing
            for i, (x, y, tw, th) in enumerate(tiles):
                tile = self._extract_tile(img, x, y, tw, th)
                tile = self._prepare_tile(tile)
                result = self.engine.infer(tile)
                result = self._unpad_result(result, tw * self.scale, th * self.scale)
                self._blend_tile(output, weights, result, x, y, tw, th)
                if progress_callback:
                    progress_callback(i + 1, total_tiles)

        # Normalize by weights
        if NUMBA_AVAILABLE:
            _normalize_output_numba(output, weights)
        else:
            np.maximum(weights, 1e-8, out=weights)
            np.divide(output, weights, out=output)

        return output

    def _blend_tile(
        self,
        output: np.ndarray,
        weights: np.ndarray,
        result: np.ndarray,
        x: int,
        y: int,
        tw: int,
        th: int,
    ):
        """Blend a tile into output with feathered weights."""
        out_x = x * self.scale
        out_y = y * self.scale
        out_tw = tw * self.scale
        out_th = th * self.scale

        weight = self._create_blend_weight(out_tw, out_th)

        if NUMBA_AVAILABLE:
            _apply_weighted_blend_numba(output, weights, result, weight, out_y, out_x, out_th, out_tw)
        else:
            out_region = output[out_y:out_y + out_th, out_x:out_x + out_tw]
            weight_region = weights[out_y:out_y + out_th, out_x:out_x + out_tw]

            np.multiply(result, weight, out=result)
            np.add(out_region, result, out=out_region)
            np.add(weight_region, weight, out=weight_region)

    def upscale_image(
        self,
        image: Image.Image,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        """
        Upscale a PIL Image.

        Args:
            image: Input PIL Image (RGB or RGBA)
            progress_callback: Optional callback(current_tile, total_tiles)

        Returns:
            Upscaled PIL Image
        """
        has_alpha = image.mode == "RGBA"

        if has_alpha:
            # Split alpha channel
            rgb = image.convert("RGB")
            alpha = image.getchannel("A")

            # Upscale RGB
            rgb_up = self._upscale_rgb(rgb, progress_callback)

            # Upscale alpha (as grayscale)
            alpha_rgb = Image.merge("RGB", (alpha, alpha, alpha))
            alpha_up = self._upscale_rgb(alpha_rgb, None)
            alpha_up = alpha_up.convert("L")

            # Merge back
            rgb_up.putalpha(alpha_up)
            return rgb_up
        else:
            rgb = image.convert("RGB")
            return self._upscale_rgb(rgb, progress_callback)

    def _upscale_rgb(
        self,
        image: Image.Image,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        """Upscale RGB image with tiling."""
        width, height = image.size

        # Check if tiling is needed
        if width <= self.tile_w and height <= self.tile_h:
            # Process entire image at once
            arr = np.array(image).astype(np.float32) / 255.0
            arr = self._prepare_tile(arr)
            result = self.engine.infer(arr)
            result = self._unpad_result(result, width * self.scale, height * self.scale)
            result = (result * 255.0).clip(0, 255).astype(np.uint8)
            if progress_callback:
                progress_callback(1, 1)
            return Image.fromarray(result)

        # Convert to array and use array path
        img_array = np.array(image).astype(np.float32) / 255.0
        result = self._upscale_array_rgb(img_array, progress_callback)

        # Convert to uint8
        np.multiply(result, 255.0, out=result)
        np.clip(result, 0, 255, out=result)
        result = result.astype(np.uint8)

        return Image.fromarray(result)

    def _calculate_tiles(self, width: int, height: int) -> list:
        """Calculate tile positions with overlap."""
        tiles = []
        step_w = self.tile_w - self.overlap * 2
        step_h = self.tile_h - self.overlap * 2

        y = 0
        while y < height:
            th = min(self.tile_h, height - y)
            x = 0
            while x < width:
                tw = min(self.tile_w, width - x)
                tiles.append((x, y, tw, th))
                x += step_w
                if x + self.overlap >= width:
                    break
            y += step_h
            if y + self.overlap >= height:
                break

        return tiles

    def _extract_tile(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Extract tile with mirror padding at edges."""
        img_h, img_w = img.shape[:2]

        # Calculate padding needed
        pad_left = max(0, -x)
        pad_top = max(0, -y)
        pad_right = max(0, (x + w) - img_w)
        pad_bottom = max(0, (y + h) - img_h)

        # Adjust extraction coordinates
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        tile = img[y1:y2, x1:x2].copy()

        # Apply mirror padding if needed
        if pad_left or pad_top or pad_right or pad_bottom:
            tile = np.pad(
                tile,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="reflect"
            )

        return tile

    def _pad_to_alignment(self, arr: np.ndarray) -> np.ndarray:
        """Pad array to be multiple of TILE_ALIGNMENT (skip if disable_tile_limit)."""
        if self.disable_tile_limit:
            return arr

        h, w = arr.shape[:2]
        new_h = ((h + self.TILE_ALIGNMENT - 1) // self.TILE_ALIGNMENT) * self.TILE_ALIGNMENT
        new_w = ((w + self.TILE_ALIGNMENT - 1) // self.TILE_ALIGNMENT) * self.TILE_ALIGNMENT

        if new_h == h and new_w == w:
            return arr

        pad_h = new_h - h
        pad_w = new_w - w

        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    def _pad_to_tile_size(self, arr: np.ndarray) -> np.ndarray:
        """Pad array to full tile size for consistent cuDNN benchmark caching.

        cuDNN benchmark mode caches the fastest algorithm per input shape.
        By ensuring all tiles are the same size, we only benchmark once.
        """
        h, w = arr.shape[:2]

        # Already full size
        if h == self.tile_h and w == self.tile_w:
            return arr

        pad_h = self.tile_h - h
        pad_w = self.tile_w - w

        if pad_h <= 0 and pad_w <= 0:
            return arr

        return np.pad(arr, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)), mode="reflect")

    def _prepare_tile(self, arr: np.ndarray) -> np.ndarray:
        """Prepare tile for inference with appropriate padding.

        For PyTorch backend: no alignment needed (flexible input sizes).
        For other backends: pad to alignment (64 pixels).
        """
        if self.backend == "pytorch":
            # PyTorch doesn't need alignment padding
            return arr
        else:
            # TensorRT/DirectML: align to 64
            return self._pad_to_alignment(arr)

    def _unpad_result(self, arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Remove padding from result to match target size."""
        return arr[:target_h, :target_w]

    def _create_blend_weight(self, w: int, h: int) -> np.ndarray:
        """Create feathered blending weight for tile overlap (cached)."""
        key = (w, h)
        if key in self._blend_weight_cache:
            return self._blend_weight_cache[key]

        overlap_scaled = self.overlap * self.scale

        if NUMBA_AVAILABLE:
            weight = _create_blend_weight_numba(h, w, overlap_scaled)
        else:
            # Create linear ramps for feathering
            weight = np.ones((h, w, 1), dtype=np.float32)

            if overlap_scaled > 0:
                # Horizontal feather
                ramp = np.linspace(0, 1, overlap_scaled, dtype=np.float32)
                weight[:, :overlap_scaled, 0] *= ramp
                weight[:, -overlap_scaled:, 0] *= ramp[::-1]

                # Vertical feather
                ramp = np.linspace(0, 1, overlap_scaled, dtype=np.float32)
                weight[:overlap_scaled, :, 0] *= ramp[:, np.newaxis]
                weight[-overlap_scaled:, :, 0] *= ramp[::-1, np.newaxis]

        # Cache for reuse
        self._blend_weight_cache[key] = weight
        return weight

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_tile_executor'):
            self._tile_executor.shutdown(wait=False)


def upscale_file(
    input_path: str,
    output_path: str,
    onnx_path: str = "",
    tile_size: Tuple[int, int] = (512, 512),
    overlap: int = 16,
    fp16: bool = False,
    bf16: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    backend: str = "tensorrt",
    disable_tile_limit: bool = False,
    # PyTorch-specific options
    pytorch_model_path: str = "",
    pytorch_device: str = "cuda",
    pytorch_half: bool = False,
    pytorch_bf16: bool = True,
    pytorch_vram_mode: str = "normal",
    # PyTorch optimization options
    pytorch_enable_tf32: bool = True,
    pytorch_channels_last: bool = True,
) -> bool:
    """
    Convenience function to upscale an image file.
    Uses fast fpng for PNG saving, OpenCV for loading.

    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image
        onnx_path: Path to ONNX model (for TensorRT/DirectML)
        tile_size: Tile size for processing
        overlap: Overlap between tiles
        fp16: Use FP16 precision
        bf16: Use BF16 precision
        progress_callback: Progress callback
        backend: "tensorrt", "directml", or "pytorch"
        disable_tile_limit: Skip tile alignment and padding
        pytorch_model_path: Path to .pth/.safetensors model (for PyTorch)
        pytorch_device: Device for PyTorch ("cuda", "cpu")
        pytorch_half: Use FP16 for PyTorch
        pytorch_bf16: Use BF16 for PyTorch (Ampere+ GPUs)
        pytorch_vram_mode: VRAM mode for PyTorch ("normal", "auto", "low_vram")
        pytorch_enable_tf32: Enable TF32 for matmuls/convolutions (Ampere+)
        pytorch_channels_last: Use NHWC memory format (faster for CNNs)

    Returns:
        True if successful
    """
    try:
        upscaler = ImageUpscaler(
            onnx_path=onnx_path,
            tile_size=tile_size,
            overlap=overlap,
            fp16=fp16,
            bf16=bf16,
            backend=backend,
            disable_tile_limit=disable_tile_limit,
            pytorch_model_path=pytorch_model_path,
            pytorch_device=pytorch_device,
            pytorch_half=pytorch_half,
            pytorch_bf16=pytorch_bf16,
            pytorch_vram_mode=pytorch_vram_mode,
            pytorch_enable_tf32=pytorch_enable_tf32,
            pytorch_channels_last=pytorch_channels_last,
        )

        # Use fast I/O
        img, has_alpha = load_image_fast(input_path)
        result = upscaler.upscale_array(img, has_alpha, progress_callback)

        # Determine output format
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".png":
            save_image_fast(result, output_path, has_alpha)
        else:
            # Fall back to PIL for other formats
            from .fast_io import array_to_pil
            pil_result = array_to_pil(result, has_alpha)
            if ext in (".jpg", ".jpeg"):
                pil_result = pil_result.convert("RGB")
                pil_result.save(output_path, quality=95)
            elif ext == ".webp":
                pil_result.save(output_path, quality=95, lossless=False)
            else:
                pil_result.save(output_path)

        return True
    except Exception as e:
        print(f"Error upscaling {input_path}: {e}")
        return False
