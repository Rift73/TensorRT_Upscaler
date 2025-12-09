"""
PyTorch inference engine for super-resolution models.
Uses spandrel for automatic architecture detection and model loading.

Supports .pth, .safetensors, and .ckpt model files.

Precision modes:
- FP32: Full precision (default, most compatible)
- FP16: Half precision (faster, less VRAM)
- BF16: BFloat16 (Ampere+ GPUs, better numerical stability than FP16)

VRAM modes:
- normal: Full GPU inference, OOM = error
- auto: Try GPU first, fallback to offloading on OOM
- low_vram: Move entire model CPU<->GPU per tile
- ramtorch: Layer-by-layer offloading using RamTorch (requires ramtorch package)

Optimization options:
- TF32: Enable TensorFloat32 for matmuls/convolutions (Ampere+, ~2-3x faster)
- channels_last: Use NHWC memory format (faster for CNNs on tensor cores)
- inference_mode: More aggressive than no_grad (faster)
"""

import os
import gc
import numpy as np
from typing import Optional, Tuple, Callable

# Check for PyTorch
TORCH_AVAILABLE = False
torch = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Check for spandrel
SPANDREL_AVAILABLE = False
spandrel = None

try:
    import spandrel
    SPANDREL_AVAILABLE = True
    # Try to load extra architectures
    try:
        import spandrel_extra_arches
        spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)
    except ImportError:
        pass
except ImportError:
    pass

# Check for HuggingFace Accelerate (optional, for better offloading)
ACCELERATE_AVAILABLE = False
try:
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import remove_hook_from_module
    ACCELERATE_AVAILABLE = True
except ImportError:
    pass

# Check for RamTorch (optional, layer-by-layer offloading)
RAMTORCH_AVAILABLE = False
replace_linear_with_ramtorch = None
try:
    from ramtorch.helpers import replace_linear_with_ramtorch
    RAMTORCH_AVAILABLE = True
except ImportError:
    pass


class PyTorchEngine:
    """
    PyTorch engine for super-resolution inference using spandrel.

    Supports:
    - Automatic architecture detection from model files
    - .pth, .safetensors, .ckpt formats
    - FP16/BF16/FP32 precision
    - Layer offloading for low VRAM situations
    - TF32, channels_last, torch.compile optimizations

    VRAM modes:
    - normal: Full model on GPU
    - auto: GPU first, offload on OOM
    - low_vram: Move model CPU<->GPU per tile
    - ramtorch: Layer-by-layer offloading via RamTorch
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        half: bool = False,
        bf16: bool = True,
        vram_mode: str = "normal",  # normal, auto, low_vram, ramtorch
        # Optimization options
        enable_tf32: bool = True,  # TensorFloat32 for matmuls/convolutions
        channels_last: bool = True,  # NHWC memory format
    ):
        """
        Initialize PyTorch engine from model file.

        Args:
            model_path: Path to .pth, .safetensors, or .ckpt model
            device: Device to use ("cuda", "cuda:0", "cpu")
            half: Use FP16 precision (faster, less VRAM)
            bf16: Use BF16 precision (Ampere+ GPUs, better stability than FP16)
            vram_mode: VRAM management mode (normal/auto/low_vram/ramtorch)
            enable_tf32: Enable TensorFloat32 for matmuls/convolutions (Ampere+)
            channels_last: Use NHWC memory format (faster for CNNs)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available. "
                "Install with: pip install torch"
            )

        if not SPANDREL_AVAILABLE:
            raise RuntimeError(
                "Spandrel is not available. "
                "Install with: pip install spandrel spandrel_extra_arches"
            )

        self.model_path = model_path
        self.device = device
        # Precision: BF16 takes priority over FP16, both require GPU
        self.bf16 = bf16 and device != "cpu"
        self.half = half and device != "cpu" and not self.bf16  # FP16 only if not BF16
        self.vram_mode = vram_mode
        self.model_scale = None
        self._offload_hooks_registered = False
        self._using_accelerate = False
        self._using_ramtorch = False
        self._manual_offload = False
        self._original_device = device

        # Optimization options
        self.enable_tf32 = enable_tf32 and device != "cpu"
        self.channels_last = channels_last and device != "cpu"

        # Check BF16 support
        if self.bf16:
            if not self._check_bf16_support():
                print("[PyTorch] Warning: BF16 not supported on this GPU, falling back to FP16")
                self.bf16 = False
                self.half = True

        # Apply global backend settings before loading model
        self._apply_backend_settings()

        self._load_model()

    def _check_bf16_support(self) -> bool:
        """Check if current GPU supports BF16."""
        if not torch.cuda.is_available():
            return False

        try:
            # Get device index
            if self.device == "cuda":
                device_idx = 0
            elif self.device.startswith("cuda:"):
                device_idx = int(self.device.split(":")[1])
            else:
                return False

            # Check compute capability (BF16 requires SM 80+ / Ampere+)
            major, minor = torch.cuda.get_device_capability(device_idx)
            # SM 8.0+ (Ampere, Ada Lovelace, Hopper)
            return major >= 8
        except Exception:
            return False

    def _check_tf32_support(self) -> bool:
        """Check if current GPU supports TF32."""
        if not torch.cuda.is_available():
            return False

        try:
            if self.device == "cuda":
                device_idx = 0
            elif self.device.startswith("cuda:"):
                device_idx = int(self.device.split(":")[1])
            else:
                return False

            # TF32 requires SM 80+ (Ampere+)
            major, minor = torch.cuda.get_device_capability(device_idx)
            return major >= 8
        except Exception:
            return False

    def _apply_backend_settings(self):
        """Apply PyTorch backend settings for optimization."""
        if not torch.cuda.is_available() or self.device == "cpu":
            return

        settings_applied = []

        # TF32 settings (Ampere+ GPUs)
        if self.enable_tf32 and self._check_tf32_support():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            settings_applied.append("TF32")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if settings_applied:
            print(f"[PyTorch] Backend settings: {', '.join(settings_applied)}")

    def _load_model(self):
        """Load model using spandrel."""
        print(f"[PyTorch] Loading model: {self.model_path}")

        # Load model with spandrel (auto-detects architecture)
        self.model_descriptor = spandrel.ModelLoader().load_from_file(
            self.model_path
        )

        # Get model and scale
        self.model = self.model_descriptor.model
        self.model_scale = self.model_descriptor.scale

        print(f"[PyTorch] Architecture: {self.model_descriptor.architecture}")
        print(f"[PyTorch] Scale: {self.model_scale}x")

        # Set to eval mode
        self.model.eval()

        # Apply precision and device based on vram_mode
        if self.vram_mode == "low_vram":
            self._setup_manual_offloading()
        elif self.vram_mode == "ramtorch":
            self._setup_ramtorch()
        else:
            # Normal or auto mode: try to put on GPU
            self._move_to_device()

    def _move_to_device(self):
        """Move model to target device with optional FP16/BF16."""
        # Keep model in FP32 - autocast handles precision during forward pass
        # This is more compatible with models that have FP32 buffers (attention masks, etc.)
        self.model = self.model.to(self.device)

        # Apply channels_last memory format (NHWC) for CNN optimization
        if self.channels_last:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                print(f"[PyTorch] Using channels_last memory format")
            except Exception as e:
                print(f"[PyTorch] Warning: channels_last not supported: {e}")
                self.channels_last = False

        if self.bf16:
            print(f"[PyTorch] Using BF16 precision (via autocast)")
        elif self.half:
            print(f"[PyTorch] Using FP16 precision (via autocast)")
        else:
            print(f"[PyTorch] Using FP32 precision")

        print(f"[PyTorch] Model loaded on {self.device}")

    def _setup_ramtorch(self):
        """Setup RamTorch layer-by-layer offloading."""
        if not RAMTORCH_AVAILABLE:
            print("[PyTorch] Warning: RamTorch not available, falling back to Low VRAM mode")
            self._setup_manual_offloading()
            return

        print(f"[PyTorch] Setting up RamTorch layer offloading...")

        # Get target device
        target_device = self.device if self.device != "cpu" else "cuda"

        # Replace nn.Linear layers with RamTorch CPUBouncingLinear (CPU offloaded)
        # Weights stay on CPU and transfer to GPU during forward pass
        self.model = replace_linear_with_ramtorch(self.model, device=target_device)

        # Keep model in FP32 - autocast handles precision during forward pass
        self.model = self.model.to(target_device)

        # Apply channels_last if enabled
        if self.channels_last:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                print(f"[PyTorch] Using channels_last memory format")
            except Exception:
                self.channels_last = False

        self._using_ramtorch = True

        precision = "BF16" if self.bf16 else ("FP16" if self.half else "FP32")
        print(f"[PyTorch] RamTorch offloading enabled (target: {target_device}, {precision} via autocast)")

    def _setup_manual_offloading(self):
        """Manual offloading - keep model on CPU, move to GPU per-tile."""
        print(f"[PyTorch] Using manual CPU offloading (per-tile transfer)")

        # Keep model in FP32 on CPU - autocast handles precision during forward pass
        self.model = self.model.to("cpu")

        # Mark that we're using manual offloading (not hooks, but per-tile transfer)
        self._manual_offload = True

        if self.bf16:
            print(f"[PyTorch] Model on CPU, will transfer to GPU per-tile (BF16 via autocast)")
        elif self.half:
            print(f"[PyTorch] Model on CPU, will transfer to GPU per-tile (FP16 via autocast)")
        else:
            print(f"[PyTorch] Model on CPU, will transfer to GPU per-tile (FP32)")

    def _infer_with_offload(self, input_tensor: "torch.Tensor") -> "torch.Tensor":
        """Run inference with manual CPU<->GPU offloading per tile."""
        target_device = self.device if self.device != "cpu" else "cuda"

        # Move model to GPU
        self.model.to(target_device)

        # Apply channels_last on GPU if enabled
        if self.channels_last:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                input_tensor = input_tensor.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # Move input to GPU
        input_tensor = input_tensor.to(target_device)

        # Run inference with autocast for BF16/FP16
        try:
            if torch.cuda.is_available() and (self.half or self.bf16):
                autocast_dtype = torch.bfloat16 if self.bf16 else torch.float16
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    output_tensor = self.model(input_tensor)
            else:
                output_tensor = self.model(input_tensor)

            # Move output to CPU before moving model
            output_tensor = output_tensor.cpu()

        finally:
            # Always move model back to CPU and clear cache
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return output_tensor

    def _switch_to_offloading(self):
        """Switch from normal mode to offloading mode (for auto mode OOM recovery)."""
        print(f"[PyTorch] OOM detected, switching to layer offloading...")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Move model to CPU first
        self.model = self.model.to("cpu")

        # Setup manual offloading (fallback for auto mode)
        self._manual_offload = True
        print(f"[PyTorch] Switched to manual CPU offloading")

    def _run_inference(self, input_tensor: "torch.Tensor") -> "torch.Tensor":
        """Core inference logic with all optimizations applied."""
        # Apply channels_last to input if model uses it
        if self.channels_last and not self._manual_offload:
            input_tensor = input_tensor.to(memory_format=torch.channels_last)

        # Move input to device
        input_tensor = input_tensor.to(self.device)

        # Run with autocast for BF16/FP16 (handles mixed precision properly)
        if torch.cuda.is_available() and self.device != "cpu" and (self.half or self.bf16):
            autocast_dtype = torch.bfloat16 if self.bf16 else torch.float16
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                output_tensor = self.model(input_tensor)
        else:
            output_tensor = self.model(input_tensor)

        return output_tensor

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

        # Convert to tensor (always start with float32, autocast handles precision)
        input_tensor = torch.from_numpy(input_array.astype(np.float32))

        # Run inference based on mode
        try:
            # Use inference_mode for maximum performance (more aggressive than no_grad)
            with torch.inference_mode():
                if self._manual_offload:
                    # Manual offloading: move model to GPU, run, move back to CPU
                    output_tensor = self._infer_with_offload(input_tensor)
                elif self._using_ramtorch or self._using_accelerate:
                    # RamTorch/Accelerate handle device placement
                    output_tensor = self._run_inference(input_tensor)
                else:
                    # Normal mode: model already on GPU
                    output_tensor = self._run_inference(input_tensor)

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg and "allocate" in error_msg:
                if self.vram_mode == "auto" and not self._manual_offload and not self._using_ramtorch and not self._using_accelerate:
                    # Auto mode: switch to offloading and retry
                    self._switch_to_offloading()
                    return self.infer(input_array)  # Retry with offloading
                else:
                    raise RuntimeError(
                        f"CUDA out of memory. Try reducing tile size or enabling Low VRAM mode. "
                        f"Original error: {e}"
                    )
            else:
                raise

        # Convert back to numpy (ensure contiguous for output)
        output_array = output_tensor.float().cpu().contiguous().numpy()

        # Convert from NCHW to HWC
        output_array = np.transpose(output_array[0], (1, 2, 0))
        np.clip(output_array, 0.0, 1.0, out=output_array)

        return output_array

    def infer_nchw(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on pre-transposed NCHW input.

        Args:
            input_array: Input image as numpy array (N, C, H, W), contiguous float32

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        # Convert to tensor (always start with float32, autocast handles precision)
        input_tensor = torch.from_numpy(input_array)

        # Run inference based on mode
        try:
            # Use inference_mode for maximum performance
            with torch.inference_mode():
                if self._manual_offload:
                    output_tensor = self._infer_with_offload(input_tensor)
                elif self._using_ramtorch or self._using_accelerate:
                    output_tensor = self._run_inference(input_tensor)
                else:
                    output_tensor = self._run_inference(input_tensor)

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg and "allocate" in error_msg:
                if self.vram_mode == "auto" and not self._manual_offload and not self._using_ramtorch and not self._using_accelerate:
                    self._switch_to_offloading()
                    return self.infer_nchw(input_array)
                else:
                    raise RuntimeError(
                        f"CUDA out of memory. Try reducing tile size or enabling Low VRAM mode. "
                        f"Original error: {e}"
                    )
            else:
                raise

        # Convert back to numpy (ensure contiguous)
        output_array = output_tensor.float().cpu().contiguous().numpy()

        # Convert from NCHW to HWC
        output_array = np.transpose(output_array[0], (1, 2, 0))
        np.clip(output_array, 0.0, 1.0, out=output_array)

        return output_array

    def release_vram(self):
        """Release VRAM by moving model to CPU."""
        if not hasattr(self, 'model') or self.model is None:
            return

        try:
            if self._using_accelerate:
                # Remove accelerate hooks and move to CPU
                try:
                    remove_hook_from_module(self.model, recurse=True)
                except Exception:
                    pass
                self._using_accelerate = False

            # Move model to CPU
            self.model = self.model.to("cpu")

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"[PyTorch] VRAM released")
        except Exception as e:
            # Silently handle errors during cleanup
            print(f"[PyTorch] Warning: Error during VRAM release: {e}")

    def __del__(self):
        """Cleanup resources."""
        # Silent cleanup - don't call release_vram() which prints and does heavy ops
        # Just drop references and let Python/CUDA handle cleanup
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Just set to None, don't move to CPU (can cause issues during thread cleanup)
                self.model = None
        except Exception:
            pass


def is_pytorch_available() -> bool:
    """Check if PyTorch backend is available."""
    return TORCH_AVAILABLE and SPANDREL_AVAILABLE


def get_pytorch_devices() -> list:
    """Get list of available PyTorch devices."""
    devices = []

    if not TORCH_AVAILABLE:
        return devices

    # Always add CPU
    devices.append({
        "index": -1,
        "name": "CPU",
        "device": "cpu",
    })

    # Add CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "device": f"cuda:{i}",
                "memory": props.total_memory // (1024 * 1024),  # MB
            })

    return devices


def get_supported_extensions() -> list:
    """Get list of supported model file extensions."""
    return [".pth", ".safetensors", ".ckpt", ".pt"]
