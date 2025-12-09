"""
Background worker threads for image processing.
Handles upscaling, clipboard operations, and progress reporting.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Callable

from PySide6.QtCore import Signal, QThread

from ..upscaler import ImageUpscaler
from ..sharpening import cas_sharpen_array, adaptive_sharpen_array
from ..resize import resize_array, compute_scaled_size
from ..fast_io import (
    load_image_fast,
    save_image_fast,
    AsyncImageSaver,
    AsyncImageLoader,
    extract_image_metadata,
)
from ..utils import (
    generate_output_path,
    optimize_png,
    should_skip_image,
)


class UpscaleWorker(QThread):
    """Background worker for upscaling images with full feature support.

    Optimizations:
    - Async prefetching overlaps I/O with GPU inference
    - Async saving writes PNGs in background thread
    - Progress callbacks emit only when needed (reduces GIL contention)
    """

    progress = Signal(int, int)  # current_tile, total_tiles
    file_progress = Signal(int, int, str)  # current_file, total_files, current_file_path
    finished = Signal(bool, str)  # success, message
    file_done = Signal(str, str, float)  # input_path, output_path, elapsed_time
    file_skipped = Signal(str, str)  # input_path, reason
    checkpoint_updated = Signal(int, list)  # current_index, remaining_files

    def __init__(self, files: List[str], config, onnx_path: str, input_root: str = "", start_index: int = 0):
        super().__init__()
        self.files = files
        self.config = config
        self.onnx_path = onnx_path
        self.input_root = input_root
        self._cancelled = False
        self._skipped_count = 0
        self._start_index = start_index  # For resume support

    def run(self):
        try:
            cfg = self.config

            # Build engine only if upscaling is enabled
            upscaler = None
            if cfg.upscale_enabled:
                upscaler = ImageUpscaler(
                    onnx_path=self.onnx_path,
                    tile_size=(cfg.tile_width, cfg.tile_height),
                    overlap=cfg.tile_overlap,
                    fp16=cfg.use_fp16,
                    bf16=cfg.use_bf16,
                    tf32=cfg.use_tf32,
                    backend=cfg.backend,
                    disable_tile_limit=cfg.disable_tile_limit,
                    # TensorRT-specific options
                    trt_cuda_graphs=getattr(cfg, 'trt_cuda_graphs', False),
                    trt_builder_optimization=getattr(cfg, 'trt_builder_optimization', 5),
                    # PyTorch-specific options
                    pytorch_model_path=getattr(cfg, 'pytorch_model_path', ''),
                    pytorch_device=getattr(cfg, 'pytorch_device', 'cuda'),
                    pytorch_half=getattr(cfg, 'pytorch_half', False),
                    pytorch_bf16=getattr(cfg, 'pytorch_bf16', True),
                    pytorch_vram_mode=getattr(cfg, 'pytorch_vram_mode', 'normal'),
                    # PyTorch optimization options
                    pytorch_enable_tf32=getattr(cfg, 'pytorch_enable_tf32', True),
                    pytorch_channels_last=getattr(cfg, 'pytorch_channels_last', True),
                    pytorch_cudnn_benchmark=getattr(cfg, 'pytorch_cudnn_benchmark', True),
                    pytorch_torch_compile=getattr(cfg, 'pytorch_torch_compile', 'off'),
                )

            # Setup async saver for background PNG writing
            async_saver = AsyncImageSaver(max_queue=6)
            async_saver.start()

            # Setup async loader for prefetching
            async_loader = AsyncImageLoader()

            try:
                total_files = len(self.files)

                # Prefetch first file
                if total_files > 0:
                    async_loader.prefetch(self.files[0])

                for i, input_path in enumerate(self.files):
                    # Skip already processed files (for resume)
                    if i < self._start_index:
                        continue

                    if self._cancelled:
                        # Emit checkpoint before exiting
                        remaining = self.files[i:]
                        self.checkpoint_updated.emit(i, remaining)
                        self.finished.emit(False, "Cancelled")
                        return

                    file_start = time.perf_counter()
                    self.file_progress.emit(i + 1, total_files, input_path)

                    # Emit checkpoint for auto-save
                    if cfg.auto_checkpoint:
                        remaining = self.files[i:]
                        self.checkpoint_updated.emit(i, remaining)

                    # Generate output path first (needed for skip check)
                    output_dir = cfg.last_output_path or os.path.dirname(input_path)
                    output_path = generate_output_path(
                        input_path=input_path,
                        output_dir=output_dir,
                        suffix=cfg.same_dir_suffix,
                        save_next_to_input=cfg.save_next_to_input,
                        manga_folder_mode=cfg.manga_folder_mode,
                        append_model_suffix=cfg.append_model_suffix,
                        model_name=Path(self.onnx_path).stem if cfg.append_model_suffix else "",
                        overwrite=cfg.overwrite_existing,
                        input_root=self.input_root,
                        output_format="png",
                    )

                    # Check if file should be skipped
                    should_skip, skip_reason = should_skip_image(
                        image_path=input_path,
                        output_path=output_path,
                        skip_existing=cfg.skip_existing,
                        conditional_enabled=cfg.conditional_enabled,
                        min_width=cfg.conditional_min_width,
                        min_height=cfg.conditional_min_height,
                        max_width=cfg.conditional_max_width,
                        max_height=cfg.conditional_max_height,
                        aspect_filter_enabled=cfg.aspect_filter_enabled,
                        aspect_mode=cfg.aspect_filter_mode,
                        aspect_min_ratio=cfg.aspect_filter_min_ratio,
                        aspect_max_ratio=cfg.aspect_filter_max_ratio,
                    )

                    if should_skip:
                        self._skipped_count += 1
                        self.file_skipped.emit(input_path, skip_reason)
                        # Prefetch next file if we're skipping
                        if i + 1 < total_files:
                            async_loader.prefetch(self.files[i + 1])
                        continue

                    # Load image using fast I/O (or prefetched)
                    img, img_has_alpha = async_loader.get(input_path)
                    height, width = img.shape[:2]

                    # Extract metadata (ICC profile, EXIF) if preservation is enabled
                    img_metadata = None
                    if cfg.preserve_metadata:
                        img_metadata = extract_image_metadata(input_path)

                    # Start prefetching next image immediately after loading current
                    if i + 1 < total_files:
                        async_loader.prefetch(self.files[i + 1])

                    # Feature #31-35: Pre-scale
                    if cfg.prescale_enabled:
                        new_size = compute_scaled_size(
                            width, height,
                            cfg.prescale_mode,
                            cfg.prescale_width,
                            cfg.prescale_height,
                        )
                        img = resize_array(img, new_size, cfg.prescale_kernel, img_has_alpha)
                        height, width = img.shape[:2]

                    # Feature #14: Upscale (can be disabled)
                    if cfg.upscale_enabled and upscaler:
                        img = upscaler.upscale_array(
                            img,
                            has_alpha=img_has_alpha,
                            progress_callback=lambda c, t: self.progress.emit(c, t)
                        )
                        height, width = img.shape[:2]

                    # Feature #20-25: Custom resolution
                    if cfg.custom_res_enabled:
                        new_size = compute_scaled_size(
                            width, height,
                            cfg.custom_res_mode,
                            cfg.custom_res_width,
                            cfg.custom_res_height,
                            keep_aspect=cfg.custom_res_keep_aspect,
                        )
                        img = resize_array(img, new_size, cfg.custom_res_kernel, img_has_alpha)

                    # Feature #18-19: Sharpening
                    if cfg.sharpen_enabled and cfg.sharpen_value > 0:
                        sharpen_method = getattr(cfg, 'sharpen_method', 'cas')
                        if sharpen_method == 'adaptive':
                            anime_mode = getattr(cfg, 'sharpen_anime_mode', False)
                            img = adaptive_sharpen_array(
                                img, cfg.sharpen_value, img_has_alpha,
                                overshoot_ctrl=False, anime_mode=anime_mode
                            )
                        else:  # cas or legacy
                            img = cas_sharpen_array(img, cfg.sharpen_value, img_has_alpha)

                    # Output path was already generated above for skip check
                    # Save as PNG asynchronously (background thread)
                    # Pass metadata for ICC profile and EXIF preservation
                    async_saver.save(img, output_path, img_has_alpha, img_metadata)

                    # Feature #47-49: PNG optimization (must wait for save to complete)
                    # Note: optimization runs after async save completes
                    if cfg.png_quantize_enabled or cfg.png_optimize_enabled:
                        # Wait for this file to be saved before optimizing
                        async_saver.wait_pending()
                        optimize_png(
                            output_path,
                            quantize=cfg.png_quantize_enabled,
                            quantize_colors=cfg.png_quantize_colors,
                            optimize=cfg.png_optimize_enabled,
                        )

                    # Feature #26-30: Secondary output
                    if cfg.secondary_enabled:
                        sec_size = compute_scaled_size(
                            img.shape[1], img.shape[0],
                            cfg.secondary_mode,
                            cfg.secondary_width,
                            cfg.secondary_height,
                        )
                        sec_img = resize_array(img, sec_size, cfg.secondary_kernel, img_has_alpha)

                        # Secondary output path (also PNG)
                        sec_dir = os.path.dirname(output_path)
                        sec_name = Path(output_path).stem + "_secondary.png"
                        sec_path = os.path.join(sec_dir, sec_name)
                        async_saver.save(sec_img, sec_path, img_has_alpha, img_metadata)

                    file_elapsed = time.perf_counter() - file_start
                    self.file_done.emit(input_path, output_path, file_elapsed)

                # Wait for all saves to complete
                async_saver.wait_pending()

                processed = total_files - self._skipped_count
                if self._skipped_count > 0:
                    self.finished.emit(True, f"Completed {processed} files ({self._skipped_count} skipped)")
                else:
                    self.finished.emit(True, f"Completed {processed} files")

            finally:
                async_saver.stop()
                async_loader.shutdown()
                # Explicitly release VRAM before thread ends
                if upscaler is not None and hasattr(upscaler, 'engine'):
                    try:
                        if hasattr(upscaler.engine, 'release_vram'):
                            upscaler.engine.release_vram()
                    except Exception:
                        pass

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))

    def cancel(self):
        self._cancelled = True


class ClipboardWorker(QThread):
    """Background worker for upscaling a single image to clipboard."""

    progress = Signal(int, int)  # current_tile, total_tiles
    finished = Signal(bool, str, object)  # success, message, result_image (QImage or None)

    def __init__(self, image_path: str, config, onnx_path: str):
        super().__init__()
        self.image_path = image_path
        self.config = config
        self.onnx_path = onnx_path
        self._cancelled = False

    def run(self):
        try:
            import numpy as np
            from PIL import Image as PILImage

            cfg = self.config

            # Load image
            img, img_has_alpha = load_image_fast(self.image_path)

            # Build upscaler
            upscaler = ImageUpscaler(
                onnx_path=self.onnx_path,
                tile_size=(cfg.tile_width, cfg.tile_height),
                overlap=cfg.tile_overlap,
                fp16=cfg.use_fp16,
                bf16=cfg.use_bf16,
                tf32=cfg.use_tf32,
                backend=cfg.backend,
                disable_tile_limit=cfg.disable_tile_limit,
                # TensorRT-specific options
                trt_cuda_graphs=getattr(cfg, 'trt_cuda_graphs', False),
                trt_builder_optimization=getattr(cfg, 'trt_builder_optimization', 5),
                # PyTorch-specific options
                pytorch_model_path=getattr(cfg, 'pytorch_model_path', ''),
                pytorch_device=getattr(cfg, 'pytorch_device', 'cuda'),
                pytorch_half=getattr(cfg, 'pytorch_half', False),
                pytorch_bf16=getattr(cfg, 'pytorch_bf16', True),
                pytorch_vram_mode=getattr(cfg, 'pytorch_vram_mode', 'normal'),
                # PyTorch optimization options
                pytorch_enable_tf32=getattr(cfg, 'pytorch_enable_tf32', True),
                pytorch_channels_last=getattr(cfg, 'pytorch_channels_last', True),
                pytorch_cudnn_benchmark=getattr(cfg, 'pytorch_cudnn_benchmark', True),
                pytorch_torch_compile=getattr(cfg, 'pytorch_torch_compile', 'off'),
            )

            try:
                # Upscale with progress
                def on_progress(current, total):
                    if not self._cancelled:
                        self.progress.emit(current, total)

                result = upscaler.upscale_array(img, img_has_alpha, on_progress)

                if self._cancelled:
                    self.finished.emit(False, "Cancelled", None)
                    return

                # Convert to PIL then QImage
                result_uint8 = (result * 255.0).clip(0, 255).astype(np.uint8)
                if img_has_alpha and result.shape[2] == 4:
                    pil_img = PILImage.fromarray(result_uint8, mode="RGBA")
                else:
                    pil_img = PILImage.fromarray(result_uint8, mode="RGB")

                # Convert PIL to QImage
                from PIL.ImageQt import ImageQt
                qimg = ImageQt(pil_img).copy()  # .copy() ensures data persists

                self.finished.emit(True, "Upscaled and copied to clipboard", qimg)

            finally:
                # Explicitly release VRAM before thread ends
                if hasattr(upscaler, 'engine') and hasattr(upscaler.engine, 'release_vram'):
                    try:
                        upscaler.engine.release_vram()
                    except Exception:
                        pass

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None)

    def cancel(self):
        self._cancelled = True
