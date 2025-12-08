# TensorRT Upscaler v2 - Minimal dependency image upscaler
# Uses direct TensorRT inference without VapourSynth

__version__ = "2.0.0"

from .engine import TensorRTEngine
from .dml_engine import DirectMLEngine, is_directml_available
from .upscaler import ImageUpscaler
from .sharpening import cas_sharpen, cas_sharpen_pil, cas_sharpen_array
from .resize import resize_pil, resize_hermite, resize_lanczos, resize_array, compute_scaled_size
from .config import Config, get_config, save_config
from .fast_io import (
    load_image_fast,
    save_image_fast,
    AsyncImageSaver,
    AsyncImageLoader,
    PipelinedProcessor,
    CV2_AVAILABLE,
)

__all__ = [
    "TensorRTEngine",
    "DirectMLEngine",
    "is_directml_available",
    "ImageUpscaler",
    "cas_sharpen",
    "cas_sharpen_pil",
    "cas_sharpen_array",
    "resize_pil",
    "resize_hermite",
    "resize_lanczos",
    "resize_array",
    "compute_scaled_size",
    "Config",
    "get_config",
    "save_config",
    "load_image_fast",
    "save_image_fast",
    "AsyncImageSaver",
    "AsyncImageLoader",
    "PipelinedProcessor",
    "CV2_AVAILABLE",
]


def _prewarm_numba():
    """
    Pre-warm Numba JIT compilation in a background thread.
    This eliminates the JIT compilation delay on first use.
    """
    import threading

    def _warm():
        try:
            import numpy as np

            # Pre-warm sharpening kernels
            try:
                from .sharpening import NUMBA_AVAILABLE as SHARP_NUMBA
                if SHARP_NUMBA:
                    from .sharpening import _cas_kernel_numba
                    # Trigger compilation with small test array
                    test = np.zeros((10, 10, 3), dtype=np.float32)
                    padded = np.pad(test, ((1, 1), (1, 1), (0, 0)), mode='reflect')
                    output = np.empty_like(test)
                    _cas_kernel_numba(padded, output, 0.5, True)
            except Exception:
                pass

            # Pre-warm resize kernels
            try:
                from .resize import NUMBA_AVAILABLE as RESIZE_NUMBA
                if RESIZE_NUMBA:
                    from .resize import _resize_1d_hermite_numba
                    test = np.zeros((10, 10, 3), dtype=np.float32)
                    _resize_1d_hermite_numba(test, 5, 0)
            except Exception:
                pass

            # Pre-warm upscaler blend kernels
            try:
                from .upscaler import NUMBA_AVAILABLE as UP_NUMBA
                if UP_NUMBA:
                    from .upscaler import (
                        _apply_weighted_blend_numba,
                        _normalize_output_numba,
                        _create_blend_weight_numba,
                    )
                    output = np.zeros((10, 10, 3), dtype=np.float32)
                    weights = np.zeros((10, 10, 1), dtype=np.float32)
                    result = np.zeros((5, 5, 3), dtype=np.float32)
                    weight = np.ones((5, 5, 1), dtype=np.float32)
                    _apply_weighted_blend_numba(output, weights, result, weight, 0, 0, 5, 5)
                    _normalize_output_numba(output, weights)
                    _create_blend_weight_numba(10, 10, 2)
            except Exception:
                pass

        except Exception:
            pass

    # Start warming in background thread
    thread = threading.Thread(target=_warm, daemon=True)
    thread.start()


# Start Numba pre-warming on module import
_prewarm_numba()
