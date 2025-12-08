"""
Image Sharpening Algorithms - Optimized NumPy implementations.

Includes:
1. CAS (Contrast Adaptive Sharpening) - AMD FidelityFX algorithm
2. Adaptive Sharpen - bacondither's advanced edge-aware sharpening

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


# =============================================================================
# Adaptive Sharpen - bacondither's edge-aware sharpening algorithm
# Based on: https://github.com/Selur/hybrid-glsl-filters/blob/main/GLSL/parameterized/adaptive-sharpen.glsl
# =============================================================================

# Default parameters for adaptive sharpen
AS_CURVE_HEIGHT = 1.0      # Main sharpening strength (0.3-2.0 recommended)
AS_CURVESLOPE = 0.5        # Sharpening curve slope
AS_L_OVERSHOOT = 0.003     # Light overshoot limit
AS_D_OVERSHOOT = 0.003     # Dark overshoot limit
AS_L_COMPR_LOW = 0.167     # Light compression ratio (low)
AS_L_COMPR_HIGH = 0.334    # Light compression ratio (high, near edges)
AS_D_COMPR_LOW = 0.250     # Dark compression ratio (low)
AS_D_COMPR_HIGH = 0.500    # Dark compression ratio (high, near edges)
AS_SCALE_LIM = 0.1         # Scale limit
AS_SCALE_CS = 0.056        # Scale curve steepness
AS_PM_P = 1.0              # Power mean parameter


def _soft_lim(v: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Attempt soft limiting with tanh-like approximation."""
    # sat(abs(v/s)*(27 + (v/s)^2)/(27 + 9*(v/s)^2)) * s
    ratio = np.abs(v / (s + EPSILON))
    ratio_sq = ratio ** 2
    result = ratio * (27.0 + ratio_sq) / (27.0 + 9.0 * ratio_sq)
    return np.clip(result, 0.0, 1.0) * s


def _wpmean(a: np.ndarray, b: np.ndarray, w: float, pm_p: float = AS_PM_P) -> np.ndarray:
    """Weighted power mean."""
    # pow(w*pow(abs(a), pm_p) + (1-w)*pow(abs(b), pm_p), 1/pm_p)
    return np.power(
        w * np.power(np.abs(a) + EPSILON, pm_p) + (1.0 - w) * np.power(np.abs(b) + EPSILON, pm_p),
        1.0 / pm_p
    )


if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _adaptive_sharpen_kernel_numba(
        padded: np.ndarray,
        output: np.ndarray,
        curve_height: float,
        curveslope: float,
        l_overshoot: float,
        d_overshoot: float,
        l_compr: float,
        d_compr: float,
        scale_lim: float,
        scale_cs: float,
        anime_mode: bool,
    ):
        """Numba JIT-compiled adaptive sharpen kernel."""
        h, w = output.shape[:2]

        for y in prange(h):
            for x in range(w):
                # Padded coordinates (3-pixel border)
                py, px = y + 3, x + 3

                # Sample 25 pixels in the neighborhood pattern
                # c[0] = center, c[1-8] = 3x3 ring, c[9-12] = cardinal +2, etc.
                c = np.empty((25, 3), dtype=np.float32)
                c[0] = padded[py, px]
                c[1] = padded[py-1, px-1]
                c[2] = padded[py-1, px]
                c[3] = padded[py-1, px+1]
                c[4] = padded[py, px-1]
                c[5] = padded[py, px+1]
                c[6] = padded[py+1, px-1]
                c[7] = padded[py+1, px]
                c[8] = padded[py+1, px+1]
                c[9] = padded[py-2, px]
                c[10] = padded[py, px-2]
                c[11] = padded[py, px+2]
                c[12] = padded[py+2, px]
                c[13] = padded[py+3, px]
                c[14] = padded[py+2, px+1]
                c[15] = padded[py+2, px-1]
                c[16] = padded[py, px+3]
                c[17] = padded[py+1, px+2]
                c[18] = padded[py-1, px+2]
                c[19] = padded[py, px-3]
                c[20] = padded[py+1, px-2]
                c[21] = padded[py-1, px-2]
                c[22] = padded[py-3, px]
                c[23] = padded[py-2, px+1]
                c[24] = padded[py-2, px-1]

                # Compute 3x3 Gaussian blur
                blur = (2.0 * (c[2] + c[4] + c[5] + c[7]) +
                       (c[1] + c[3] + c[6] + c[8]) + 4.0 * c[0]) / 16.0

                # Contrast compression factor
                blur_sum = blur[0] + blur[1] + blur[2]
                c_comp = min(max(0.266666681 + 0.9 * (2.0 ** (blur_sum * (-7.4/3.0))), 0.0), 1.0)

                # Edge detection using weighted differences from blur
                edge = 0.0
                for ch in range(3):
                    b_diff_0 = abs(blur[ch] - c[0, ch])
                    b_diff_2 = abs(blur[ch] - c[2, ch])
                    b_diff_4 = abs(blur[ch] - c[4, ch])
                    b_diff_5 = abs(blur[ch] - c[5, ch])
                    b_diff_7 = abs(blur[ch] - c[7, ch])
                    b_diff_1 = abs(blur[ch] - c[1, ch])
                    b_diff_3 = abs(blur[ch] - c[3, ch])
                    b_diff_6 = abs(blur[ch] - c[6, ch])
                    b_diff_8 = abs(blur[ch] - c[8, ch])
                    b_diff_9 = abs(blur[ch] - c[9, ch])
                    b_diff_10 = abs(blur[ch] - c[10, ch])
                    b_diff_11 = abs(blur[ch] - c[11, ch])
                    b_diff_12 = abs(blur[ch] - c[12, ch])

                    edge_ch = (1.38 * b_diff_0 +
                              1.15 * (b_diff_2 + b_diff_4 + b_diff_5 + b_diff_7) +
                              0.92 * (b_diff_1 + b_diff_3 + b_diff_6 + b_diff_8) +
                              0.23 * (b_diff_9 + b_diff_10 + b_diff_11 + b_diff_12))
                    edge += edge_ch * edge_ch

                edge = (edge ** 0.5) * c_comp

                # Use red channel as luma approximation for simplicity
                luma = np.empty(25, dtype=np.float32)
                for i in range(25):
                    luma[i] = 0.299 * c[i, 0] + 0.587 * c[i, 1] + 0.114 * c[i, 2]

                c0_Y = luma[0]

                # Simplified negative laplacian (weighted average of neighbors)
                neg_laplace = (luma[1] + luma[2] + luma[3] + luma[4] +
                              luma[5] + luma[6] + luma[7] + luma[8]) / 8.0

                # Adaptive sharpening strength
                sharpen_val = curve_height / (curve_height * curveslope * (edge ** 3.5) + 0.625)

                # Compute sharp difference
                sharpdiff = (c0_Y - neg_laplace) * sharpen_val

                # Find local min/max using partial sort
                sorted_luma = np.sort(luma)
                nmin = sorted_luma[1]   # Second smallest
                nmax = sorted_luma[23]  # Second largest

                # Blend with center
                nmax = (max(nmax, c0_Y) * 3.0 + sorted_luma[24]) / 4.0
                nmin = (min(nmin, c0_Y) * 3.0 + sorted_luma[0]) / 4.0

                # Compute overshoot limits
                min_dist = min(abs(nmax - c0_Y), abs(c0_Y - nmin))
                pos_scale = min_dist + l_overshoot
                neg_scale = min_dist + d_overshoot

                pos_scale = min(pos_scale, scale_lim * (1.0 - scale_cs) + pos_scale * scale_cs)
                neg_scale = min(neg_scale, scale_lim * (1.0 - scale_cs) + neg_scale * scale_cs)

                # Apply soft limiting
                # anime_mode: only darken edges (no brightening)
                if sharpdiff > 0:
                    if anime_mode:
                        sharpdiff = 0.0
                    else:
                        ratio = abs(sharpdiff / (pos_scale + 1e-6))
                        ratio_sq = ratio * ratio
                        soft_val = ratio * (27.0 + ratio_sq) / (27.0 + 9.0 * ratio_sq)
                        soft_val = min(max(soft_val, 0.0), 1.0) * pos_scale
                        # Weighted power mean blend
                        sharpdiff = l_compr * abs(sharpdiff) + (1.0 - l_compr) * abs(soft_val)
                else:
                    ratio = abs(sharpdiff / (neg_scale + 1e-6))
                    ratio_sq = ratio * ratio
                    soft_val = ratio * (27.0 + ratio_sq) / (27.0 + 9.0 * ratio_sq)
                    soft_val = min(max(soft_val, 0.0), 1.0) * neg_scale
                    sharpdiff = -(d_compr * abs(sharpdiff) + (1.0 - d_compr) * abs(soft_val))

                # Limit sharpening to valid range
                sharpdiff_lim = min(max(c0_Y + sharpdiff, 0.0), 1.0) - c0_Y

                # Compute saturation multiplier
                satmul = (c0_Y + max(sharpdiff_lim * 0.9, sharpdiff_lim) * 1.03 + 0.03) / (c0_Y + 0.03)

                # Apply to each channel
                for ch in range(3):
                    result = c0_Y + (sharpdiff_lim * 3.0 + sharpdiff) / 4.0 + (c[0, ch] - c0_Y) * satmul
                    output[y, x, ch] = min(max(result, 0.0), 1.0)


def adaptive_sharpen(
    image: np.ndarray,
    strength: float = 1.0,
    overshoot_ctrl: bool = False,
    anime_mode: bool = False,
) -> np.ndarray:
    """
    Apply Adaptive Sharpen to an image.

    This is an advanced edge-aware sharpening algorithm that adapts its strength
    based on local contrast and edge detection. It includes anti-ringing
    (overshoot control) to prevent halos.

    Args:
        image: Input image as numpy array (H, W, C) with values in [0, 1]
        strength: Sharpening strength multiplier (0.0-2.0, default 1.0)
        overshoot_ctrl: Enable advanced overshoot control near edges
        anime_mode: Only darken edges (no brightening), better for anime/cartoon

    Returns:
        Sharpened image as numpy array (H, W, C)
    """
    if strength <= 0:
        return image

    # Ensure float32 and [0, 1] range
    if image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image.copy()

    if img.max() > 1.0:
        img = img / 255.0

    h, w, c = img.shape

    # Pad image with 3 pixels for 7x7 neighborhood sampling
    padded = np.pad(img, ((3, 3), (3, 3), (0, 0)), mode='reflect')

    # Compression ratios based on overshoot control
    if overshoot_ctrl:
        l_compr = AS_L_COMPR_HIGH
        d_compr = AS_D_COMPR_HIGH
    else:
        l_compr = AS_L_COMPR_LOW
        d_compr = AS_D_COMPR_LOW

    curve_height = AS_CURVE_HEIGHT * strength

    # Use Numba kernel if available
    if NUMBA_AVAILABLE:
        output = np.empty_like(img)
        _adaptive_sharpen_kernel_numba(
            padded, output,
            curve_height, AS_CURVESLOPE,
            AS_L_OVERSHOOT, AS_D_OVERSHOOT,
            l_compr, d_compr,
            AS_SCALE_LIM, AS_SCALE_CS,
            anime_mode,
        )
        return output

    # NumPy fallback (slower but works without Numba)
    output = np.empty_like(img)

    # Sample offsets for the 25-point pattern
    offsets = [
        (0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1), (-2, 0),
        (0, -2), (0, 2), (2, 0), (3, 0), (2, 1),
        (2, -1), (0, 3), (1, 2), (-1, 2), (0, -3),
        (1, -2), (-1, -2), (-3, 0), (-2, 1), (-2, -1),
    ]

    # Process each pixel
    for y in range(h):
        for x in range(w):
            py, px = y + 3, x + 3

            # Sample 25 pixels
            samples = np.array([padded[py + dy, px + dx] for dy, dx in offsets])

            # 3x3 Gaussian blur
            blur = (2.0 * (samples[2] + samples[4] + samples[5] + samples[7]) +
                   (samples[1] + samples[3] + samples[6] + samples[8]) +
                   4.0 * samples[0]) / 16.0

            # Contrast compression
            blur_sum = blur.sum()
            c_comp = np.clip(0.266666681 + 0.9 * np.exp2(blur_sum * (-7.4/3.0)), 0.0, 1.0)

            # Edge detection
            b_diffs = np.abs(blur - samples[:13])
            weights = np.array([1.38, 0.92, 1.15, 0.92, 1.15, 1.15, 0.92, 1.15, 0.92, 0.23, 0.23, 0.23, 0.23])
            edge = np.linalg.norm((b_diffs * weights[:, np.newaxis]).sum(axis=0)) * c_comp

            # Luma values
            luma = 0.299 * samples[:, 0] + 0.587 * samples[:, 1] + 0.114 * samples[:, 2]
            c0_Y = luma[0]

            # Simplified negative laplacian
            neg_laplace = luma[1:9].mean()

            # Adaptive strength
            sharpen_val = curve_height / (curve_height * AS_CURVESLOPE * (edge ** 3.5) + 0.625)
            sharpdiff = (c0_Y - neg_laplace) * sharpen_val

            # Local min/max
            sorted_luma = np.sort(luma)
            nmax = (max(sorted_luma[23], c0_Y) * 3.0 + sorted_luma[24]) / 4.0
            nmin = (min(sorted_luma[1], c0_Y) * 3.0 + sorted_luma[0]) / 4.0

            # Overshoot limits
            min_dist = min(abs(nmax - c0_Y), abs(c0_Y - nmin))
            pos_scale = min(min_dist + AS_L_OVERSHOOT, AS_SCALE_LIM * (1.0 - AS_SCALE_CS) + (min_dist + AS_L_OVERSHOOT) * AS_SCALE_CS)
            neg_scale = min(min_dist + AS_D_OVERSHOOT, AS_SCALE_LIM * (1.0 - AS_SCALE_CS) + (min_dist + AS_D_OVERSHOOT) * AS_SCALE_CS)

            # Soft limiting
            # anime_mode: only darken edges (no brightening)
            if sharpdiff > 0:
                if anime_mode:
                    sharpdiff = 0.0
                else:
                    soft_val = _soft_lim(np.array([sharpdiff]), np.array([pos_scale]))[0]
                    sharpdiff = l_compr * abs(sharpdiff) + (1.0 - l_compr) * abs(soft_val)
            else:
                soft_val = _soft_lim(np.array([sharpdiff]), np.array([neg_scale]))[0]
                sharpdiff = -(d_compr * abs(sharpdiff) + (1.0 - d_compr) * abs(soft_val))

            # Apply sharpening
            sharpdiff_lim = np.clip(c0_Y + sharpdiff, 0.0, 1.0) - c0_Y
            satmul = (c0_Y + max(sharpdiff_lim * 0.9, sharpdiff_lim) * 1.03 + 0.03) / (c0_Y + 0.03)

            result = c0_Y + (sharpdiff_lim * 3.0 + sharpdiff) / 4.0 + (samples[0] - c0_Y) * satmul
            output[y, x] = np.clip(result, 0.0, 1.0)

    return output


def adaptive_sharpen_pil(
    image: Image.Image,
    strength: float = 1.0,
    overshoot_ctrl: bool = False,
    anime_mode: bool = False,
) -> Image.Image:
    """
    Apply Adaptive Sharpen to a PIL Image.

    Args:
        image: Input PIL Image (RGB or RGBA)
        strength: Sharpening strength (0.0-2.0, default 1.0)
        overshoot_ctrl: Enable advanced overshoot control
        anime_mode: Only darken edges (no brightening), better for anime/cartoon

    Returns:
        Sharpened PIL Image
    """
    if strength <= 0:
        return image

    has_alpha = image.mode == 'RGBA'

    if has_alpha:
        rgb = image.convert('RGB')
        alpha = image.getchannel('A')
    else:
        rgb = image.convert('RGB')
        alpha = None

    # Convert to numpy [0, 1]
    arr = np.array(rgb).astype(np.float32) / 255.0

    # Apply adaptive sharpen
    sharpened = adaptive_sharpen(arr, strength, overshoot_ctrl, anime_mode)

    # Convert back to uint8
    sharpened = (sharpened * 255.0).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(sharpened, mode='RGB')

    if has_alpha:
        result.putalpha(alpha)

    return result


def adaptive_sharpen_array(
    img: np.ndarray,
    strength: float = 1.0,
    has_alpha: bool = False,
    overshoot_ctrl: bool = False,
    anime_mode: bool = False,
) -> np.ndarray:
    """
    Apply Adaptive Sharpen to a numpy array image.

    Args:
        img: Input image as numpy array (H, W, C) in [0, 1] range
        strength: Sharpening strength (0.0-2.0, default 1.0)
        has_alpha: Whether image has alpha channel
        overshoot_ctrl: Enable advanced overshoot control
        anime_mode: Only darken edges (no brightening), better for anime/cartoon

    Returns:
        Sharpened image as numpy array
    """
    if strength <= 0:
        return img

    if has_alpha and img.shape[2] == 4:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3:4]
        rgb_sharp = adaptive_sharpen(rgb, strength, overshoot_ctrl, anime_mode)
        return np.concatenate([rgb_sharp, alpha], axis=2)
    else:
        return adaptive_sharpen(img[:, :, :3], strength, overshoot_ctrl, anime_mode)
