"""
DirectML inference engine for ONNX super-resolution models.
Alternative backend for AMD/Intel GPUs using ONNX Runtime with DirectML.

This provides a vendor-agnostic GPU acceleration option that works on any
DirectX 12 compatible GPU (NVIDIA, AMD, Intel).
"""

import os
import numpy as np
from typing import Optional, Tuple

# Check for ONNX Runtime DirectML
ONNXRUNTIME_DML_AVAILABLE = False
ort = None

try:
    import onnxruntime as ort
    # Check if DirectML execution provider is available
    available_providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in available_providers:
        ONNXRUNTIME_DML_AVAILABLE = True
except ImportError:
    pass


class DirectMLEngine:
    """
    DirectML engine for super-resolution inference using ONNX Runtime.

    Supports:
    - ONNX model loading (no engine building required)
    - DirectML GPU acceleration (works on AMD, Intel, NVIDIA)
    - Dynamic input shapes

    Note: Generally slower than TensorRT on NVIDIA GPUs but provides
    broader hardware compatibility.
    """

    def __init__(
        self,
        onnx_path: str,
        fp16: bool = False,
        device_id: int = 0,
    ):
        """
        Initialize DirectML engine from ONNX model.

        Args:
            onnx_path: Path to ONNX super-resolution model
            fp16: Enable FP16 precision (if supported)
            device_id: GPU device ID to use
        """
        if not ONNXRUNTIME_DML_AVAILABLE:
            raise RuntimeError(
                "ONNX Runtime with DirectML is not available. "
                "Install with: pip install onnxruntime-directml"
            )

        self.onnx_path = onnx_path
        self.fp16 = fp16
        self.device_id = device_id
        self.model_scale = None

        self._detect_model_scale()
        self._create_session()

    def _detect_model_scale(self):
        """Detect model scale from filename (e.g., HAT_L_2x, RealESRGAN_4x)."""
        basename = os.path.basename(self.onnx_path).lower()
        for scale in [8, 4, 2, 1]:
            patterns = [f"{scale}x_", f"_{scale}x", f"x{scale}", f"-{scale}x", f"{scale}x."]
            for pattern in patterns:
                if pattern in basename:
                    self.model_scale = scale
                    print(f"[DirectML] Detected model scale: {scale}x")
                    return
        self.model_scale = 4
        print(f"[DirectML] Using default model scale: 4x")

    def _create_session(self):
        """Create ONNX Runtime inference session with DirectML."""
        print(f"[DirectML] Loading ONNX model: {self.onnx_path}")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        # DirectML provider options
        provider_options = {
            'device_id': self.device_id,
        }

        # Create session with DirectML provider
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_options,
            providers=[
                ('DmlExecutionProvider', provider_options),
                'CPUExecutionProvider',  # Fallback
            ]
        )

        # Get input/output names and types
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.output_name = self.session.get_outputs()[0].name

        # Detect input data type
        input_type = input_info.type
        if 'float16' in input_type:
            self.input_dtype = np.float16
            print(f"[DirectML] Model uses FP16 precision")
        else:
            self.input_dtype = np.float32
            print(f"[DirectML] Model uses FP32 precision")

        print(f"[DirectML] Session created successfully")
        print(f"[DirectML] Input: {self.input_name}, Output: {self.output_name}")
        print(f"[DirectML] Active provider: {self.session.get_providers()[0]}")

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on input array.

        Args:
            input_array: Input image as numpy array (H, W, C) or (N, C, H, W)
                        Values should be in range [0, 1] as float32

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        # Ensure NCHW format
        if input_array.ndim == 3:
            input_array = np.transpose(input_array, (2, 0, 1))[np.newaxis, ...]

        # Convert to model's expected dtype
        input_array = np.ascontiguousarray(input_array.astype(self.input_dtype))

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_array}
        )

        output_array = outputs[0]

        # Convert back to HWC and float32
        output_array = np.transpose(output_array[0], (1, 2, 0)).astype(np.float32)
        np.clip(output_array, 0.0, 1.0, out=output_array)

        return output_array

    def infer_nchw(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on pre-transposed NCHW input (skips format conversion).

        Args:
            input_array: Input image as numpy array (N, C, H, W), contiguous float32
                        Values should be in range [0, 1]

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        # Convert to model's expected dtype if needed
        if input_array.dtype != self.input_dtype:
            input_array = input_array.astype(self.input_dtype)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_array}
        )

        output_array = outputs[0]

        # Convert back to HWC and float32
        output_array = np.transpose(output_array[0], (1, 2, 0)).astype(np.float32)
        np.clip(output_array, 0.0, 1.0, out=output_array)

        return output_array

    def __del__(self):
        """Cleanup resources."""
        self.session = None


def get_directml_devices() -> list:
    """Get list of available DirectML devices."""
    devices = []

    if not ONNXRUNTIME_DML_AVAILABLE:
        return devices

    # DirectML doesn't have a direct API to enumerate devices,
    # but we can try to create sessions on different device IDs
    # For now, just report device 0 as available if DML works
    try:
        # Simple check - if DML provider is available, report default device
        devices.append({
            "index": 0,
            "name": "DirectML Device 0 (Default GPU)",
            "backend": "DirectML",
        })
    except Exception:
        pass

    return devices


def is_directml_available() -> bool:
    """Check if DirectML backend is available."""
    return ONNXRUNTIME_DML_AVAILABLE
