"""
Base protocol for inference engines.

All engines (TensorRT, DirectML, PyTorch) implement this interface.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol defining the common interface for all inference engines.

    Implementations: TensorRTEngine, DirectMLEngine, PyTorchEngine.
    """

    model_scale: int

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on an input array.

        Args:
            input_array: (H, W, C) or (N, C, H, W) float32 in [0, 1].

        Returns:
            Upscaled array (H*scale, W*scale, C) float32 in [0, 1].
        """
        ...

    def infer_nchw(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on a pre-transposed NCHW input array.

        Args:
            input_array: (N, C, H, W) float32 in [0, 1].

        Returns:
            Upscaled array (H*scale, W*scale, C) float32 in [0, 1].
        """
        ...
