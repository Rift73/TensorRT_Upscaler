"""
Custom resize kernels - Optimized NumPy/Pillow implementation.
Includes Hermite (Bicubic B=0, C=0) and Lanczos.

Optimizations:
- Optional Numba JIT for kernel computation
- Pre-computed weight matrices
- OpenCV preferred when available (10x faster)
- In-place operations where possible
"""

import numpy as np
from PIL import Image
from typing import Tuple, Literal

# Try to import Numba for JIT acceleration
NUMBA_AVAILABLE = False
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    pass

# Check for OpenCV (much faster than pure NumPy)
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass


# Numba JIT kernels (defined conditionally)
if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def _hermite_kernel_numba(x):
        """Numba JIT Hermite kernel."""
        ax = abs(x)
        if ax <= 1.0:
            return (2.0 * ax - 3.0) * (ax * ax) + 1.0
        return 0.0

    @njit(fastmath=True, cache=True)
    def _lanczos_kernel_numba(x, a):
        """Numba JIT Lanczos kernel."""
        ax = abs(x)
        if ax < 1e-8:
            return 1.0
        if ax >= a:
            return 0.0
        pi_x = 3.141592653589793 * ax
        return (a * np.sin(pi_x) * np.sin(pi_x / a)) / (pi_x * pi_x)

    @njit(parallel=True, fastmath=True, cache=True)
    def _resize_1d_hermite_numba(data, new_size, axis):
        """Numba JIT 1D Hermite resize."""
        if axis == 0:
            old_size = data.shape[0]
            out_shape = (new_size, data.shape[1], data.shape[2])
        else:
            old_size = data.shape[1]
            out_shape = (data.shape[0], new_size, data.shape[2])

        output = np.zeros(out_shape, dtype=np.float32)
        scale = new_size / old_size
        kernel_radius = 1.0

        if scale < 1.0:
            # Downscaling: widen window but don't scale kernel distances
            # This averages more source pixels with proper Hermite weighting
            effective_radius = kernel_radius / scale
        else:
            effective_radius = kernel_radius

        if axis == 0:
            for i in prange(new_size):
                out_coord = (i + 0.5) / scale - 0.5
                left = int(np.floor(out_coord - effective_radius))
                right = int(np.ceil(out_coord + effective_radius))

                for j in range(data.shape[1]):
                    for c in range(data.shape[2]):
                        val = 0.0
                        weight_sum = 0.0
                        for k in range(left, right + 1):
                            # Don't scale distances - use fractional distance to source pixel
                            dist = (out_coord - k) / effective_radius
                            w = _hermite_kernel_numba(dist)
                            idx = max(0, min(old_size - 1, k))
                            val += w * data[idx, j, c]
                            weight_sum += w
                        if weight_sum > 0:
                            output[i, j, c] = val / weight_sum
        else:
            for j in prange(new_size):
                out_coord = (j + 0.5) / scale - 0.5
                left = int(np.floor(out_coord - effective_radius))
                right = int(np.ceil(out_coord + effective_radius))

                for i in range(data.shape[0]):
                    for c in range(data.shape[2]):
                        val = 0.0
                        weight_sum = 0.0
                        for k in range(left, right + 1):
                            # Don't scale distances - use fractional distance to source pixel
                            dist = (out_coord - k) / effective_radius
                            w = _hermite_kernel_numba(dist)
                            idx = max(0, min(old_size - 1, k))
                            val += w * data[i, idx, c]
                            weight_sum += w
                        if weight_sum > 0:
                            output[i, j, c] = val / weight_sum

        return output


def _hermite_kernel(x: np.ndarray) -> np.ndarray:
    """
    Hermite (Cubic B=0, C=0) interpolation kernel.

    For |x| <= 1: (2|x| - 3)|x|^2 + 1
    For |x| > 1: 0

    This is smoother than Catmull-Rom, with no overshoot/ringing.
    """
    ax = np.abs(x)
    result = np.zeros_like(ax)

    # |x| <= 1
    mask = ax <= 1.0
    ax_m = ax[mask]
    result[mask] = (2.0 * ax_m - 3.0) * (ax_m ** 2) + 1.0

    return result


def _lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """
    Lanczos interpolation kernel with parameter a (typically 3).

    lanczos(x) = sinc(x) * sinc(x/a)  for |x| < a
               = 0                     for |x| >= a
    """
    ax = np.abs(x)
    result = np.zeros_like(ax)

    # Avoid division by zero at x=0
    mask_zero = ax < 1e-8
    mask_valid = (ax >= 1e-8) & (ax < a)

    result[mask_zero] = 1.0

    if np.any(mask_valid):
        x_valid = ax[mask_valid]
        pi_x = np.pi * x_valid
        result[mask_valid] = (a * np.sin(pi_x) * np.sin(pi_x / a)) / (pi_x * pi_x)

    return result


def _catmull_rom_kernel(x: np.ndarray) -> np.ndarray:
    """
    Catmull-Rom (Cubic B=0, C=0.5) interpolation kernel.
    This is what Pillow's BICUBIC uses.
    """
    ax = np.abs(x)
    result = np.zeros_like(ax)

    # |x| <= 1
    mask1 = ax <= 1.0
    ax1 = ax[mask1]
    result[mask1] = 1.5 * (ax1 ** 3) - 2.5 * (ax1 ** 2) + 1.0

    # 1 < |x| <= 2
    mask2 = (ax > 1.0) & (ax <= 2.0)
    ax2 = ax[mask2]
    result[mask2] = -0.5 * (ax2 ** 3) + 2.5 * (ax2 ** 2) - 4.0 * ax2 + 2.0

    return result


def _resize_1d(
    data: np.ndarray,
    new_size: int,
    kernel_func,
    kernel_radius: float,
    axis: int = 0,
) -> np.ndarray:
    """
    Resize along one axis using the specified kernel.
    """
    old_size = data.shape[axis]

    if old_size == new_size:
        return data

    # Scale factor
    scale = new_size / old_size

    # For downscaling, we need to widen the kernel window
    if scale < 1.0:
        effective_radius = kernel_radius / scale
    else:
        effective_radius = kernel_radius

    # Output coordinates in input space
    out_coords = (np.arange(new_size) + 0.5) / scale - 0.5

    # For each output pixel, find contributing input pixels
    left = np.floor(out_coords - effective_radius).astype(int)
    right = np.ceil(out_coords + effective_radius).astype(int)

    # Maximum window size
    max_window = int(np.ceil(effective_radius * 2)) + 2

    # Build weight matrix
    weights = np.zeros((new_size, max_window), dtype=np.float32)
    indices = np.zeros((new_size, max_window), dtype=int)

    for i in range(new_size):
        l, r = left[i], right[i]
        window_indices = np.arange(l, r + 1)

        # Fractional distance from output coord to each input coord
        # Normalized by effective_radius so kernel sees values in [-1, 1]
        distances = (out_coords[i] - window_indices) / effective_radius

        # Compute weights
        w = kernel_func(distances)

        # Clamp indices to valid range
        clamped_indices = np.clip(window_indices, 0, old_size - 1)

        # Store
        n = len(w)
        weights[i, :n] = w
        indices[i, :n] = clamped_indices

    # Normalize weights
    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0
    weights /= weight_sums

    # Apply weights
    # Move target axis to position 0
    data_moved = np.moveaxis(data, axis, 0)
    original_shape = data_moved.shape

    # Flatten other dimensions
    data_flat = data_moved.reshape(old_size, -1)

    # Output
    out_flat = np.zeros((new_size, data_flat.shape[1]), dtype=data_flat.dtype)

    for i in range(new_size):
        for j in range(max_window):
            if weights[i, j] > 0:
                out_flat[i] += weights[i, j] * data_flat[indices[i, j]]

    # Reshape back
    new_shape = (new_size,) + original_shape[1:]
    out_moved = out_flat.reshape(new_shape)

    return np.moveaxis(out_moved, 0, axis)


def resize_hermite(
    image: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize image using Hermite (Cubic B=0, C=0) interpolation.

    Args:
        image: Input image (H, W, C) or (H, W), float32 in [0, 1]
        size: Target size as (width, height)

    Returns:
        Resized image
    """
    target_w, target_h = size

    # Ensure 3D array
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    # Ensure float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Use Numba JIT version if available (5-10x faster)
    if NUMBA_AVAILABLE:
        result = _resize_1d_hermite_numba(image, target_h, 0)
        result = _resize_1d_hermite_numba(result, target_w, 1)
        return np.clip(result, 0.0, 1.0)

    # Fall back to NumPy implementation
    result = _resize_1d(image, target_h, _hermite_kernel, 1.0, axis=0)
    result = _resize_1d(result, target_w, _hermite_kernel, 1.0, axis=1)

    return np.clip(result, 0.0, 1.0)


def resize_lanczos(
    image: np.ndarray,
    size: Tuple[int, int],
    a: int = 3,
) -> np.ndarray:
    """
    Resize image using Lanczos interpolation.

    Args:
        image: Input image (H, W, C) or (H, W), float32 in [0, 1]
        size: Target size as (width, height)
        a: Lanczos parameter (default 3)

    Returns:
        Resized image
    """
    target_w, target_h = size

    kernel = lambda x: _lanczos_kernel(x, a)

    result = _resize_1d(image, target_h, kernel, float(a), axis=0)
    result = _resize_1d(result, target_w, kernel, float(a), axis=1)

    return np.clip(result, 0.0, 1.0)


def resize_pil(
    image: Image.Image,
    size: Tuple[int, int],
    kernel: Literal["lanczos", "hermite", "bicubic"] = "lanczos",
) -> Image.Image:
    """
    Resize a PIL Image using the specified kernel.

    Args:
        image: Input PIL Image
        size: Target size as (width, height)
        kernel: "lanczos", "hermite", or "bicubic"

    Returns:
        Resized PIL Image
    """
    if kernel == "lanczos":
        # Use Pillow's built-in Lanczos (faster)
        return image.resize(size, Image.LANCZOS)

    elif kernel == "bicubic":
        # Use Pillow's built-in Bicubic (Catmull-Rom)
        return image.resize(size, Image.BICUBIC)

    elif kernel == "hermite":
        # Use our custom Hermite implementation
        has_alpha = image.mode == 'RGBA'

        if has_alpha:
            rgb = image.convert('RGB')
            alpha = image.getchannel('A')
        else:
            rgb = image.convert('RGB')
            alpha = None

        # Convert to numpy [0, 1]
        arr = np.array(rgb).astype(np.float32) / 255.0

        # Resize with Hermite
        resized = resize_hermite(arr, size)

        # Convert back
        resized = (resized * 255.0).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(resized, mode='RGB')

        if has_alpha:
            # Resize alpha with Hermite too
            alpha_arr = np.array(alpha).astype(np.float32) / 255.0
            alpha_arr = alpha_arr[:, :, np.newaxis]  # Add channel dim
            alpha_resized = resize_hermite(alpha_arr, size)
            alpha_resized = (alpha_resized[:, :, 0] * 255.0).clip(0, 255).astype(np.uint8)
            alpha_pil = Image.fromarray(alpha_resized, mode='L')
            result.putalpha(alpha_pil)

        return result

    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def compute_scaled_size(
    orig_w: int,
    orig_h: int,
    mode: str,
    target_w: int = 1920,
    target_h: int = 1080,
    keep_aspect: bool = True,
) -> Tuple[int, int]:
    """
    Compute target size based on mode.

    Args:
        orig_w: Original width
        orig_h: Original height
        mode: "width", "height", or "2x"
        target_w: Target width (used when mode="width")
        target_h: Target height (used when mode="height")
        keep_aspect: Whether to maintain aspect ratio

    Returns:
        (width, height) tuple
    """
    if mode == "2x":
        return (orig_w // 2, orig_h // 2)

    elif mode == "width":
        new_w = target_w
        if keep_aspect:
            new_h = int(orig_h * (target_w / orig_w))
        else:
            new_h = target_h
        return (new_w, new_h)

    elif mode == "height":
        new_h = target_h
        if keep_aspect:
            new_w = int(orig_w * (target_h / orig_h))
        else:
            new_w = target_w
        return (new_w, new_h)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def resize_array(
    img: np.ndarray,
    size: Tuple[int, int],
    kernel: Literal["lanczos", "hermite", "bicubic"] = "lanczos",
    has_alpha: bool = False,
) -> np.ndarray:
    """
    Resize a numpy array image.

    Args:
        img: Input image as numpy array (H, W, C) in [0, 1] range
        size: Target size as (width, height)
        kernel: "lanczos", "hermite", or "bicubic"
        has_alpha: Whether image has alpha channel (4 channels)

    Returns:
        Resized image as numpy array
    """
    # Try OpenCV first (faster)
    try:
        import cv2

        target_w, target_h = size

        if kernel == "lanczos":
            interp = cv2.INTER_LANCZOS4
        elif kernel == "bicubic":
            interp = cv2.INTER_CUBIC
        elif kernel == "hermite":
            # OpenCV doesn't have Hermite, use our implementation
            if has_alpha and img.shape[2] == 4:
                rgb = resize_hermite(img[:, :, :3], size)
                alpha = resize_hermite(img[:, :, 3:4], size)
                return np.concatenate([rgb, alpha], axis=2)
            else:
                return resize_hermite(img[:, :, :3], size)
        else:
            interp = cv2.INTER_LANCZOS4

        # OpenCV resize works with any number of channels
        resized = cv2.resize(img, (target_w, target_h), interpolation=interp)

        return np.clip(resized, 0.0, 1.0)

    except ImportError:
        # Fall back to our implementation
        if kernel == "hermite":
            resize_fn = resize_hermite
        else:
            resize_fn = resize_lanczos

        if has_alpha and img.shape[2] == 4:
            rgb = resize_fn(img[:, :, :3], size)
            alpha = resize_fn(img[:, :, 3:4], size)
            return np.concatenate([rgb, alpha], axis=2)
        else:
            return resize_fn(img[:, :, :3], size)
