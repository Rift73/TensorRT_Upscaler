"""
Direct TensorRT inference engine for ONNX super-resolution models.
No VapourSynth or vsmlrt dependency - pure TensorRT Python API.

Precision modes:
- FP16: Fast, widely supported
- BF16: Good balance of speed and quality (default)
- TF32: Tensor Float 32 mode

TensorRT optimizations (MAXED OUT):
- Builder optimization level 5 (maximum tactics exploration)
- 100GB workspace memory (practically unlimited)
- All tactic sources enabled (cuBLAS, cuBLASLt, cuDNN, EdgeMask, JIT)
- Hardware compatibility NONE (optimized for current GPU only)
- CUDA graphs DISABLED (Myelin backend doesn't support capture)
- Precision constraints preferred
- Direct I/O enabled
- Sparse weights optimization enabled

Performance optimizations:
- Pinned (page-locked) host memory for faster H2D/D2H transfers
- Double-buffered async transfers with dual CUDA streams
- Pre-allocated persistent buffers
- Overlapped compute and memory transfers
"""

import os
import ctypes
import numpy as np
import tensorrt as trt
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor

# Try to import cuda-python (newer API: cuda.bindings.runtime)
CUDART_AVAILABLE = False
cudart = None

try:
    from cuda.bindings import runtime as cudart
    CUDART_AVAILABLE = True
except ImportError:
    try:
        # Older cuda-python API
        import cuda.cudart as cudart
        CUDART_AVAILABLE = True
    except ImportError:
        pass

# Fallback to pycuda
PYCUDA_AVAILABLE = False
cuda = None

if not CUDART_AVAILABLE:
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        PYCUDA_AVAILABLE = True
    except ImportError:
        pass

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def check_cuda_err(err):
    """Check CUDA runtime error."""
    if CUDART_AVAILABLE and cudart is not None:
        if isinstance(err, tuple):
            if hasattr(cudart, 'cudaError_t'):
                if err[0] != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"CUDA Error: {err[0]}")
            elif err[0] != 0:
                raise RuntimeError(f"CUDA Error: {err[0]}")
            return err[1] if len(err) > 1 else None
    return err


class TensorRTEngine:
    """
    Direct TensorRT engine for super-resolution inference.

    Supports:
    - ONNX model loading and TRT engine building
    - Engine caching for faster subsequent loads
    - FP16/BF16/TF32 precision modes
    - Dynamic input shapes

    Performance features:
    - Pinned host memory for ~2x faster PCIe transfers
    - Double-buffered async execution for overlapped compute/transfer
    - Pre-allocated buffers to eliminate allocation overhead
    """

    def __init__(
        self,
        onnx_path: str,
        fp16: bool = False,
        bf16: bool = True,
        tf32: bool = False,
        min_shape: tuple = (1, 3, 64, 64),
        opt_shape: tuple = (1, 3, 512, 512),
        max_shape: tuple = (1, 3, 1920, 1088),
    ):
        """
        Initialize TensorRT engine from ONNX model.

        Args:
            onnx_path: Path to ONNX super-resolution model
            fp16: Enable FP16 precision
            bf16: Enable BF16 precision (default, best speed/quality)
            tf32: Enable TF32 precision
            min_shape: Minimum input shape (N, C, H, W)
            opt_shape: Optimal input shape for engine tuning
            max_shape: Maximum input shape
        """
        self.onnx_path = onnx_path
        self.fp16 = fp16
        self.bf16 = bf16
        self.tf32 = tf32
        self.min_shape = min_shape
        self.opt_shape = opt_shape
        self.max_shape = max_shape

        self.engine = None
        self.context = None
        self.stream = None
        self.stream2 = None  # Second stream for double-buffering
        self.input_name = None
        self.output_name = None
        self.model_scale = None

        # GPU memory buffers (double-buffered)
        self.d_input = [None, None]
        self.d_output = [None, None]
        self.input_nbytes = 0
        self.output_nbytes = 0

        # Pinned host memory buffers (double-buffered)
        self.h_input = [None, None]
        self.h_output = [None, None]
        self.h_input_nbytes = 0
        self.h_output_nbytes = 0

        # Current buffer index for double-buffering
        self._buf_idx = 0

        # Thread pool for async output processing
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._detect_model_scale()
        self._build_or_load_engine()

        if self.engine is None:
            raise RuntimeError(f"Failed to build or load TensorRT engine from: {onnx_path}")
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self._create_streams()

    def _detect_model_scale(self):
        """Detect model scale from filename (e.g., HAT_L_2x, RealESRGAN_4x)."""
        basename = os.path.basename(self.onnx_path).lower()
        for scale in [8, 4, 2, 1]:
            patterns = [f"{scale}x_", f"_{scale}x", f"x{scale}", f"-{scale}x", f"{scale}x."]
            for pattern in patterns:
                if pattern in basename:
                    self.model_scale = scale
                    print(f"Detected model scale: {scale}x")
                    return
        self.model_scale = 4
        print(f"Using default model scale: 4x")

    def _get_cache_path(self) -> str:
        """Generate engine cache path based on ONNX path and settings."""
        base = os.path.splitext(self.onnx_path)[0]
        if self.fp16:
            precision = "fp16"
        elif self.bf16:
            precision = "bf16"
        else:
            precision = "fp32"
        shape_str = f"{self.max_shape[3]}x{self.max_shape[2]}"
        return f"{base}_{precision}_{shape_str}.trt"

    def _build_or_load_engine(self):
        """Build TensorRT engine from ONNX or load from cache."""
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            onnx_mtime = os.path.getmtime(self.onnx_path)
            if cache_mtime > onnx_mtime:
                print(f"Loading cached TensorRT engine: {cache_path}")
                self.engine = self._load_engine(cache_path)
                if self.engine:
                    self._create_context()
                    return

        print(f"Building TensorRT engine from: {self.onnx_path}")
        self.engine = self._build_engine()

        if self.engine:
            self._save_engine(cache_path)
            self._create_context()

    def _build_engine(self):
        """Build TensorRT engine from ONNX model."""
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        print(f"Parsing ONNX model: {self.onnx_path}")
        with open(self.onnx_path, "rb") as f:
            onnx_data = f.read()

        if not parser.parse(onnx_data):
            print("ERROR: Failed to parse ONNX model!")
            for i in range(parser.num_errors):
                print(f"  ONNX Parse Error {i+1}: {parser.get_error(i)}")
            return None
        print(f"ONNX model parsed successfully. Network has {network.num_layers} layers.")

        config = builder.create_builder_config()

        # === MAXED OUT SETTINGS ===

        # Builder optimization level 5 (maximum) - TensorRT 8.6+
        # Level 5 = longest build time but best runtime performance
        # Tries more tactics and kernel options
        if hasattr(config, 'builder_optimization_level'):
            config.builder_optimization_level = 5
            print("Builder optimization level: 5 (MAXIMUM)")

        # Set massive workspace - practically unlimited (100GB)
        # TensorRT will only use what it needs, this just removes the limit
        workspace_size = 100 * (1 << 30)  # 100GB
        if hasattr(config, 'set_memory_pool_limit'):
            # TensorRT 8.4+ API
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
            # Tactic shared memory has a hard limit of < 1GB
            if hasattr(trt.MemoryPoolType, 'TACTIC_SHARED_MEMORY'):
                tactic_shmem_size = 1023 * (1 << 20)  # 1023MB (just under 1GB limit)
                config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY, tactic_shmem_size)
            print(f"Memory pools: Workspace={workspace_size // (1 << 30)}GB, TacticSHMEM=1023MB")
        else:
            # Older API fallback
            config.max_workspace_size = workspace_size
            print(f"Workspace limit: {workspace_size // (1 << 30)}GB")

        # Tactic sources - enable ALL available tactics for best performance
        # This includes cuBLAS, cuBLASLt, cuDNN, and edge mask convolutions
        if hasattr(config, 'set_tactic_sources'):
            tactic_sources = 0
            if hasattr(trt.TacticSource, 'CUBLAS'):
                tactic_sources |= 1 << int(trt.TacticSource.CUBLAS)
            if hasattr(trt.TacticSource, 'CUBLAS_LT'):
                tactic_sources |= 1 << int(trt.TacticSource.CUBLAS_LT)
            if hasattr(trt.TacticSource, 'CUDNN'):
                tactic_sources |= 1 << int(trt.TacticSource.CUDNN)
            if hasattr(trt.TacticSource, 'EDGE_MASK_CONVOLUTIONS'):
                tactic_sources |= 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
            if hasattr(trt.TacticSource, 'JIT_CONVOLUTIONS'):
                tactic_sources |= 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
            config.set_tactic_sources(tactic_sources)
            print("Tactic sources: ALL ENABLED (cuBLAS, cuBLASLt, cuDNN, EdgeMask, JIT)")

        # Hardware compatibility level - fastest for current GPU only
        if hasattr(config, 'hardware_compatibility_level'):
            if hasattr(trt.HardwareCompatibilityLevel, 'NONE'):
                config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.NONE
                print("Hardware compatibility: NONE (optimized for current GPU)")

        # Precision constraints - use PREFER (OBEY is mutually exclusive)
        if hasattr(trt.BuilderFlag, 'PREFER_PRECISION_CONSTRAINTS'):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            print("Enabled: PREFER_PRECISION_CONSTRAINTS")

        # Enable direct I/O for potentially faster transfers
        if hasattr(trt.BuilderFlag, 'DIRECT_IO'):
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
            print("Enabled: DIRECT_IO")

        # Reject empty algorithms (more robust builds)
        if hasattr(trt.BuilderFlag, 'REJECT_EMPTY_ALGORITHMS'):
            config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

        # Sparse weights optimization if available
        if hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            print("Enabled: SPARSE_WEIGHTS")

        # Set precision flags
        if self.bf16 and hasattr(trt.BuilderFlag, 'BF16') and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.BF16)
            print("Using BF16 precision")
        elif self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        if self.tf32:
            config.set_flag(trt.BuilderFlag.TF32)
            print("Using TF32 precision")
        if not self.fp16 and not self.bf16 and not self.tf32:
            print("Using FP32 precision")

        # Configure dynamic shapes
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        self.input_name = input_tensor.name

        profile.set_shape(
            self.input_name,
            self.min_shape,
            self.opt_shape,
            self.max_shape
        )
        config.add_optimization_profile(profile)

        output_tensor = network.get_output(0)
        self.output_name = output_tensor.name

        print(f"Building engine with shapes: min={self.min_shape}, opt={self.opt_shape}, max={self.max_shape}")
        print(f"Precision: FP16={self.fp16}, BF16={self.bf16}, TF32={self.tf32}")
        print("This may take several minutes...")

        try:
            serialized_engine = builder.build_serialized_network(network, config)
        except Exception as e:
            print(f"TensorRT build exception: {e}")
            return None

        if serialized_engine is None:
            print("ERROR: Failed to build TensorRT engine!")
            return None

        print("Engine build successful, deserializing...")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            print("ERROR: Failed to deserialize TensorRT engine!")
            return None

        print("TensorRT engine ready!")
        return engine

    def _load_engine(self, path: str):
        """Load serialized TensorRT engine from file."""
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _save_engine(self, path: str):
        """Save serialized TensorRT engine to file."""
        if self.engine:
            with open(path, "wb") as f:
                f.write(self.engine.serialize())
            print(f"Saved TensorRT engine to: {path}")

    def _create_context(self):
        """Create execution context from engine."""
        self.context = self.engine.create_execution_context()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        print(f"Input tensor: {self.input_name}")
        print(f"Output tensor: {self.output_name}")

    def _create_streams(self):
        """Create CUDA streams for async execution."""
        if CUDART_AVAILABLE:
            self.stream = check_cuda_err(cudart.cudaStreamCreate())
            self.stream2 = check_cuda_err(cudart.cudaStreamCreate())
        elif PYCUDA_AVAILABLE:
            self.stream = cuda.Stream()
            self.stream2 = cuda.Stream()
        else:
            self.stream = None
            self.stream2 = None

    def _allocate_buffers(self, input_shape, output_shape):
        """Allocate GPU and pinned host buffers if needed (double-buffered)."""
        input_nbytes = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
        output_nbytes = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

        # Allocate GPU buffers (both slots)
        if input_nbytes > self.input_nbytes:
            self._free_gpu_buffers()
            if CUDART_AVAILABLE:
                self.d_input[0] = check_cuda_err(cudart.cudaMalloc(input_nbytes))
                self.d_input[1] = check_cuda_err(cudart.cudaMalloc(input_nbytes))
            elif PYCUDA_AVAILABLE:
                self.d_input[0] = cuda.mem_alloc(input_nbytes)
                self.d_input[1] = cuda.mem_alloc(input_nbytes)
            self.input_nbytes = input_nbytes

        if output_nbytes > self.output_nbytes:
            if CUDART_AVAILABLE:
                for i in range(2):
                    if self.d_output[i]:
                        cudart.cudaFree(self.d_output[i])
                self.d_output[0] = check_cuda_err(cudart.cudaMalloc(output_nbytes))
                self.d_output[1] = check_cuda_err(cudart.cudaMalloc(output_nbytes))
            elif PYCUDA_AVAILABLE:
                for i in range(2):
                    if self.d_output[i]:
                        self.d_output[i].free()
                self.d_output[0] = cuda.mem_alloc(output_nbytes)
                self.d_output[1] = cuda.mem_alloc(output_nbytes)
            self.output_nbytes = output_nbytes

        # Allocate pinned host buffers (both slots)
        if input_nbytes > self.h_input_nbytes:
            self._free_host_buffers()
            if CUDART_AVAILABLE:
                # Allocate pinned memory for faster transfers
                self.h_input[0] = check_cuda_err(cudart.cudaMallocHost(input_nbytes))
                self.h_input[1] = check_cuda_err(cudart.cudaMallocHost(input_nbytes))
            self.h_input_nbytes = input_nbytes

        if output_nbytes > self.h_output_nbytes:
            if CUDART_AVAILABLE:
                for i in range(2):
                    if self.h_output[i]:
                        cudart.cudaFreeHost(self.h_output[i])
                self.h_output[0] = check_cuda_err(cudart.cudaMallocHost(output_nbytes))
                self.h_output[1] = check_cuda_err(cudart.cudaMallocHost(output_nbytes))
            self.h_output_nbytes = output_nbytes

    def _free_gpu_buffers(self):
        """Free GPU buffers."""
        if CUDART_AVAILABLE:
            for i in range(2):
                if self.d_input[i]:
                    cudart.cudaFree(self.d_input[i])
                if self.d_output[i]:
                    cudart.cudaFree(self.d_output[i])
        elif PYCUDA_AVAILABLE:
            for i in range(2):
                if self.d_input[i]:
                    self.d_input[i].free()
                if self.d_output[i]:
                    self.d_output[i].free()

        self.d_input = [None, None]
        self.d_output = [None, None]
        self.input_nbytes = 0
        self.output_nbytes = 0

    def _free_host_buffers(self):
        """Free pinned host buffers."""
        if CUDART_AVAILABLE:
            for i in range(2):
                if self.h_input[i]:
                    cudart.cudaFreeHost(self.h_input[i])
                if self.h_output[i]:
                    cudart.cudaFreeHost(self.h_output[i])

        self.h_input = [None, None]
        self.h_output = [None, None]
        self.h_input_nbytes = 0
        self.h_output_nbytes = 0

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on input array.

        Args:
            input_array: Input image as numpy array (H, W, C) or (N, C, H, W)
                        Values should be in range [0, 1] as float32

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        if self.context is None:
            raise RuntimeError("TensorRT context not initialized.")

        # Ensure NCHW format
        if input_array.ndim == 3:
            input_array = np.transpose(input_array, (2, 0, 1))[np.newaxis, ...]

        input_array = np.ascontiguousarray(input_array.astype(np.float32))
        batch, channels, height, width = input_array.shape
        input_shape = input_array.shape

        if height > self.max_shape[2] or width > self.max_shape[3]:
            raise ValueError(
                f"Input size {width}x{height} exceeds engine max shape "
                f"{self.max_shape[3]}x{self.max_shape[2]}."
            )

        out_height = height * self.model_scale
        out_width = width * self.model_scale
        output_shape = (batch, channels, out_height, out_width)

        self._allocate_buffers(input_shape, output_shape)

        idx = self._buf_idx

        if CUDART_AVAILABLE:
            # Use pinned memory path for faster transfers
            if self.h_input[idx] is not None:
                # Copy input to pinned buffer using ctypes memmove
                ctypes.memmove(self.h_input[idx], input_array.ctypes.data, input_array.nbytes)

                # Async copy from pinned to device
                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.d_input[idx],
                    self.h_input[idx],
                    input_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream
                ))
            else:
                # Fallback: regular copy
                check_cuda_err(cudart.cudaMemcpy(
                    self.d_input[idx],
                    input_array.ctypes.data,
                    input_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                ))

            # Execute inference
            self.context.set_tensor_address(self.input_name, self.d_input[idx])
            self.context.set_tensor_address(self.output_name, self.d_output[idx])
            self.context.set_input_shape(self.input_name, input_shape)
            self.context.execute_async_v3(self.stream)

            # Copy output back
            if self.h_output[idx] is not None:
                # Async copy from device to pinned
                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.h_output[idx],
                    self.d_output[idx],
                    int(np.prod(output_shape) * 4),
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream
                ))
                check_cuda_err(cudart.cudaStreamSynchronize(self.stream))

                # Copy from pinned to output array using ctypes memmove
                output_array = np.empty(output_shape, dtype=np.float32)
                ctypes.memmove(output_array.ctypes.data, self.h_output[idx], output_array.nbytes)
            else:
                # Fallback: synchronize and regular copy
                check_cuda_err(cudart.cudaStreamSynchronize(self.stream))
                output_array = np.empty(output_shape, dtype=np.float32)
                check_cuda_err(cudart.cudaMemcpy(
                    output_array.ctypes.data,
                    self.d_output[idx],
                    output_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
                ))

        elif PYCUDA_AVAILABLE:
            cuda.memcpy_htod(self.d_input[idx], input_array)

            self.context.set_tensor_address(self.input_name, int(self.d_input[idx]))
            self.context.set_tensor_address(self.output_name, int(self.d_output[idx]))
            self.context.set_input_shape(self.input_name, input_shape)
            self.context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()

            output_array = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_array, self.d_output[idx])

        else:
            raise RuntimeError("No CUDA runtime available. Install cuda-python or pycuda.")

        # Convert back to HWC
        output_array = np.transpose(output_array[0], (1, 2, 0))
        np.clip(output_array, 0.0, 1.0, out=output_array)

        # Alternate buffer index for next call (enables potential overlap)
        self._buf_idx = 1 - self._buf_idx

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
        if self.context is None:
            raise RuntimeError("TensorRT context not initialized.")

        # Input is already NCHW and contiguous - skip conversion
        batch, channels, height, width = input_array.shape
        input_shape = input_array.shape

        if height > self.max_shape[2] or width > self.max_shape[3]:
            raise ValueError(
                f"Input size {width}x{height} exceeds engine max shape "
                f"{self.max_shape[3]}x{self.max_shape[2]}."
            )

        out_height = height * self.model_scale
        out_width = width * self.model_scale
        output_shape = (batch, channels, out_height, out_width)

        self._allocate_buffers(input_shape, output_shape)

        idx = self._buf_idx

        if CUDART_AVAILABLE:
            if self.h_input[idx] is not None:
                ctypes.memmove(self.h_input[idx], input_array.ctypes.data, input_array.nbytes)

                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.d_input[idx],
                    self.h_input[idx],
                    input_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream
                ))
            else:
                check_cuda_err(cudart.cudaMemcpy(
                    self.d_input[idx],
                    input_array.ctypes.data,
                    input_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                ))

            self.context.set_tensor_address(self.input_name, self.d_input[idx])
            self.context.set_tensor_address(self.output_name, self.d_output[idx])
            self.context.set_input_shape(self.input_name, input_shape)
            self.context.execute_async_v3(self.stream)

            if self.h_output[idx] is not None:
                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.h_output[idx],
                    self.d_output[idx],
                    int(np.prod(output_shape) * 4),
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream
                ))
                check_cuda_err(cudart.cudaStreamSynchronize(self.stream))

                output_array = np.empty(output_shape, dtype=np.float32)
                ctypes.memmove(output_array.ctypes.data, self.h_output[idx], output_array.nbytes)
            else:
                check_cuda_err(cudart.cudaStreamSynchronize(self.stream))
                output_array = np.empty(output_shape, dtype=np.float32)
                check_cuda_err(cudart.cudaMemcpy(
                    output_array.ctypes.data,
                    self.d_output[idx],
                    output_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
                ))

        elif PYCUDA_AVAILABLE:
            cuda.memcpy_htod(self.d_input[idx], input_array)

            self.context.set_tensor_address(self.input_name, int(self.d_input[idx]))
            self.context.set_tensor_address(self.output_name, int(self.d_output[idx]))
            self.context.set_input_shape(self.input_name, input_shape)
            self.context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()

            output_array = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_array, self.d_output[idx])

        else:
            raise RuntimeError("No CUDA runtime available. Install cuda-python or pycuda.")

        # Convert back to HWC
        output_array = np.transpose(output_array[0], (1, 2, 0))
        np.clip(output_array, 0.0, 1.0, out=output_array)

        self._buf_idx = 1 - self._buf_idx

        return output_array

    def infer_async_start(self, input_array: np.ndarray, buf_idx: int = 0) -> Tuple[int, tuple]:
        """
        Start async inference (non-blocking). Use infer_async_wait to get result.

        Args:
            input_array: Input image as numpy array (H, W, C)
            buf_idx: Buffer index (0 or 1) for double-buffering

        Returns:
            (buf_idx, output_shape) to pass to infer_async_wait
        """
        if self.context is None:
            raise RuntimeError("TensorRT context not initialized.")

        # Ensure NCHW format
        if input_array.ndim == 3:
            input_array = np.transpose(input_array, (2, 0, 1))[np.newaxis, ...]

        input_array = np.ascontiguousarray(input_array.astype(np.float32))
        batch, channels, height, width = input_array.shape
        input_shape = input_array.shape

        out_height = height * self.model_scale
        out_width = width * self.model_scale
        output_shape = (batch, channels, out_height, out_width)

        self._allocate_buffers(input_shape, output_shape)

        stream = self.stream if buf_idx == 0 else self.stream2

        if CUDART_AVAILABLE:
            if self.h_input[buf_idx] is not None:
                ctypes.memmove(self.h_input[buf_idx], input_array.ctypes.data, input_array.nbytes)

                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.d_input[buf_idx],
                    self.h_input[buf_idx],
                    input_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream
                ))

            self.context.set_tensor_address(self.input_name, self.d_input[buf_idx])
            self.context.set_tensor_address(self.output_name, self.d_output[buf_idx])
            self.context.set_input_shape(self.input_name, input_shape)
            self.context.execute_async_v3(stream)

            if self.h_output[buf_idx] is not None:
                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.h_output[buf_idx],
                    self.d_output[buf_idx],
                    int(np.prod(output_shape) * 4),
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream
                ))

        return buf_idx, output_shape

    def infer_async_start_nchw(self, input_array: np.ndarray, buf_idx: int = 0) -> Tuple[int, tuple]:
        """
        Start async inference with pre-transposed NCHW input (skips format conversion).

        Args:
            input_array: Input image as numpy array (N, C, H, W), contiguous float32
            buf_idx: Buffer index (0 or 1) for double-buffering

        Returns:
            (buf_idx, output_shape) to pass to infer_async_wait
        """
        if self.context is None:
            raise RuntimeError("TensorRT context not initialized.")

        # Input is already NCHW and contiguous - skip conversion
        batch, channels, height, width = input_array.shape
        input_shape = input_array.shape

        out_height = height * self.model_scale
        out_width = width * self.model_scale
        output_shape = (batch, channels, out_height, out_width)

        self._allocate_buffers(input_shape, output_shape)

        stream = self.stream if buf_idx == 0 else self.stream2

        if CUDART_AVAILABLE:
            if self.h_input[buf_idx] is not None:
                ctypes.memmove(self.h_input[buf_idx], input_array.ctypes.data, input_array.nbytes)

                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.d_input[buf_idx],
                    self.h_input[buf_idx],
                    input_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream
                ))

            self.context.set_tensor_address(self.input_name, self.d_input[buf_idx])
            self.context.set_tensor_address(self.output_name, self.d_output[buf_idx])
            self.context.set_input_shape(self.input_name, input_shape)
            self.context.execute_async_v3(stream)

            if self.h_output[buf_idx] is not None:
                check_cuda_err(cudart.cudaMemcpyAsync(
                    self.h_output[buf_idx],
                    self.d_output[buf_idx],
                    int(np.prod(output_shape) * 4),
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream
                ))

        return buf_idx, output_shape

    def infer_async_wait(self, buf_idx: int, output_shape: tuple) -> np.ndarray:
        """
        Wait for async inference to complete and get result.

        Args:
            buf_idx: Buffer index from infer_async_start
            output_shape: Output shape from infer_async_start

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, C)
        """
        stream = self.stream if buf_idx == 0 else self.stream2

        if CUDART_AVAILABLE:
            check_cuda_err(cudart.cudaStreamSynchronize(stream))

            output_array = np.empty(output_shape, dtype=np.float32)
            if self.h_output[buf_idx] is not None:
                ctypes.memmove(output_array.ctypes.data, self.h_output[buf_idx], output_array.nbytes)
            else:
                check_cuda_err(cudart.cudaMemcpy(
                    output_array.ctypes.data,
                    self.d_output[buf_idx],
                    output_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
                ))

        elif PYCUDA_AVAILABLE:
            stream.synchronize()
            output_array = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_array, self.d_output[buf_idx])

        else:
            raise RuntimeError("No CUDA runtime available.")

        output_array = np.transpose(output_array[0], (1, 2, 0))
        np.clip(output_array, 0.0, 1.0, out=output_array)

        return output_array

    def __del__(self):
        """Cleanup resources."""
        self._free_gpu_buffers()
        self._free_host_buffers()
        if CUDART_AVAILABLE:
            if self.stream:
                cudart.cudaStreamDestroy(self.stream)
            if self.stream2:
                cudart.cudaStreamDestroy(self.stream2)
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        self.context = None
        self.engine = None


def get_available_gpus() -> list:
    """Get list of available CUDA GPUs."""
    gpus = []
    if CUDART_AVAILABLE:
        count = check_cuda_err(cudart.cudaGetDeviceCount())
        for i in range(count):
            props = check_cuda_err(cudart.cudaGetDeviceProperties(i))
            gpus.append({
                "index": i,
                "name": props.name.decode() if isinstance(props.name, bytes) else props.name,
                "memory": props.totalGlobalMem // (1024 * 1024),
            })
    elif PYCUDA_AVAILABLE:
        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            gpus.append({
                "index": i,
                "name": dev.name(),
                "memory": dev.total_memory() // (1024 * 1024),
            })
    return gpus
