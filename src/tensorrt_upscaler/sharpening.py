"""
Contrast Adaptive Sharpening (CAS) - Optimized NumPy implementation.
Based on AMD FidelityFX CAS algorithm.
Reference: https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening

Optimizations:
- In-place operations where possible
- Pre-allocated output arrays
- Optional Numba JIT for pixel-level operations
- Minimized memory allocations
"""

import numpy as np
from PIL import Image

# Try to import Numba for JIT acceleration
NUMBA_AVAILABLE = False
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    pass

EPSILON = 1e-6


# Numba-accelerated CAS kernel (compiled at first use)
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _cas_kernel_numba(padded, output, amount, better_diagonals):
        """Numba JIT-compiled CAS kernel for maximum performance."""
        h, w, c = output.shape
        weight_factor = amount * (1.0/5.0 - 1.0/8.0) + 1.0/8.0
        threshold = 2.0 if better_diagonals else 1.0

        for y in prange(h):
            for x in range(w):
                for ch in range(c):
                    # Get 3x3 neighborhood (padded coords are y+1, x+1)
                    py, px = y + 1, x + 1

                    a = padded[py-1, px-1, ch]
                    b = padded[py-1, px, ch]
                    c_val = padded[py-1, px+1, ch]
                    d = padded[py, px-1, ch]
                    e = padded[py, px, ch]
                    f = padded[py, px+1, ch]
                    g = padded[py+1, px-1, ch]
                    h_val = padded[py+1, px, ch]
                    i = padded[py+1, px+1, ch]

                    # Cross min/max
                    cross_mn = min(min(min(min(b, d), e), f), h_val)
                    cross_mx = max(max(max(max(b, d), e), f), h_val)

                    if better_diagonals:
                        diag_mn = min(min(min(a, c_val), g), i)
                        diag_mx = max(max(max(a, c_val), g), i)
                        mn = min(cross_mn, diag_mn)
                        mx = max(cross_mx, diag_mx)
                    else:
                        mn = cross_mn
                        mx = cross_mx

                    # Adaptive weight
                    inv_mx = 1.0 / (mx + EPSILON)
                    amp = inv_mx * min(mn, threshold - mx)
                    if amp < 0:
                        amp = 0.0
                    amp = amp ** 0.5

                    # Filter weight
                    w_val = -amp * weight_factor

                    # Apply cross filter
                    div = 1.0 / (1.0 + 4.0 * w_val)
                    result = ((b + d + f + h_val) * w_val + e) * div

                    # Clamp
                    if result < 0.0:
                        result = 0.0
                    elif result > 1.0:
                        result = 1.0

                    output[y, x, ch] = result


def cas_sharpen(
    image: np.ndarray,
    amount: float = 0.5,
    better_diagonals: bool = True,
) -> np.ndarray:
    """
    Apply Contrast Adaptive Sharpening to an image.

    Args:
        image: Input image as numpy array (H, W, C) with values in [0, 1]
        amount: Sharpening amount from 0.0 (none) to 1.0 (maximum)
        better_diagonals: Include diagonal neighbors for better quality

    Returns:
        Sharpened image as numpy array (H, W, C)
    """
    if amount <= 0:
        return image

    # Ensure float32 and [0, 1] range
    if image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image

    if img.max() > 1.0:
        img = img / 255.0

    # Pad image with reflection for edge handling
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='reflect')

    # Use Numba kernel if available (5-10x faster)
    if NUMBA_AVAILABLE:
        output = np.empty_like(img)
        _cas_kernel_numba(padded, output, amount, better_diagonals)
        return output

    # Optimized NumPy fallback with reduced memory allocations
    # Extract 3x3 neighborhood using views (no copy)
    b = padded[:-2, 1:-1]  # top-center
    d = padded[1:-1, :-2]  # mid-left
    e = padded[1:-1, 1:-1] # center (original pixel)
    f = padded[1:-1, 2:]   # mid-right
    h = padded[2:, 1:-1]   # bottom-center

    # Calculate min/max from cross pattern - use fmin/fmax for speed
    cross_mn = np.fmin(np.fmin(np.fmin(np.fmin(b, d), e), f), h)
    cross_mx = np.fmax(np.fmax(np.fmax(np.fmax(b, d), e), f), h)

    if better_diagonals:
        a = padded[:-2, :-2]
        c = padded[:-2, 2:]
        g = padded[2:, :-2]
        i = padded[2:, 2:]

        diag_mn = np.fmin(np.fmin(np.fmin(a, c), g), i)
        diag_mx = np.fmax(np.fmax(np.fmax(a, c), g), i)

        mn = np.fmin(cross_mn, diag_mn)
        mx = np.fmax(cross_mx, diag_mx)
        threshold = 2.0
    else:
        mn = cross_mn
        mx = cross_mx
        threshold = 1.0

    # Compute adaptive weight (in-place where possible)
    inv_mx = np.reciprocal(mx + EPSILON)  # faster than 1.0 / x
    amp = inv_mx * np.fmin(mn, threshold - mx)
    np.maximum(amp, 0.0, out=amp)
    np.sqrt(amp, out=amp)

    # Compute filter weight
    weight_factor = amount * (1.0/5.0 - 1.0/8.0) + 1.0/8.0
    w = amp * (-weight_factor)

    # Apply cross-shaped filter
    div = np.reciprocal(1.0 + 4.0 * w)

    # Compute sum of neighbors once
    neighbor_sum = b + d
    neighbor_sum += f
    neighbor_sum += h

    output = neighbor_sum * w
    output += e
    output *= div

    # Clamp to valid range (in-place)
    np.clip(output, 0.0, 1.0, out=output)

    return output


def cas_sharpen_pil(
    image: Image.Image,
    amount: float = 0.5,
    better_diagonals: bool = True,
) -> Image.Image:
    """
    Apply CAS sharpening to a PIL Image.

    Args:
        image: Input PIL Image (RGB or RGBA)
        amount: Sharpening amount (0.0 to 1.0)
        better_diagonals: Include diagonal neighbors

    Returns:
        Sharpened PIL Image
    """
    if amount <= 0:
        return image

    has_alpha = image.mode == 'RGBA'

    if has_alpha:
        # Separate alpha channel
        rgb = image.convert('RGB')
        alpha = image.getchannel('A')
    else:
        rgb = image.convert('RGB')
        alpha = None

    # Convert to numpy [0, 1]
    arr = np.array(rgb).astype(np.float32) / 255.0

    # Apply CAS
    sharpened = cas_sharpen(arr, amount, better_diagonals)

    # Convert back to uint8
    sharpened = (sharpened * 255.0).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(sharpened, mode='RGB')

    if has_alpha:
        result.putalpha(alpha)

    return result


def cas_sharpen_array(
    img: np.ndarray,
    amount: float = 0.5,
    has_alpha: bool = False,
    better_diagonals: bool = True,
) -> np.ndarray:
    """
    Apply CAS sharpening to a numpy array image.

    Args:
        img: Input image as numpy array (H, W, C) in [0, 1] range
        amount: Sharpening amount (0.0 to 1.0)
        has_alpha: Whether image has alpha channel
        better_diagonals: Include diagonal neighbors

    Returns:
        Sharpened image as numpy array
    """
    if amount <= 0:
        return img

    if has_alpha and img.shape[2] == 4:
        # Sharpen only RGB, preserve alpha
        rgb = img[:, :, :3]
        alpha = img[:, :, 3:4]
        rgb_sharp = cas_sharpen(rgb, amount, better_diagonals)
        return np.concatenate([rgb_sharp, alpha], axis=2)
    else:
        return cas_sharpen(img[:, :, :3], amount, better_diagonals)
