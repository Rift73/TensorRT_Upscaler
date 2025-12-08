"""
TensorRT Upscaler v2 GUI - Full-featured PySide6 interface.
Layout matches vapoursynth_image_upscaler for consistency.
"""

import os
import sys
import time
import tempfile
import re
import subprocess
import json
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, Signal, QThread, QSize, QTimer, QMimeData, QUrl, QEvent, QFileSystemWatcher
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QPixmap, QImage, QPalette, QColor,
    QClipboard, QKeySequence, QShortcut, QAction, QGuiApplication,
    QIcon,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QSplitter,
    QMenu,
    QMenuBar,
    QStatusBar,
    QSizePolicy,
    QFrame,
    QScrollArea,
    QSystemTrayIcon,
    QStyle,
)

from .config import get_config, save_config
from .utils import (
    IMAGE_EXTENSIONS,
    natural_sort_key,
    format_time_hms,
    has_alpha,
    is_animated,
    generate_output_path,
    download_url,
    extract_url_from_text,
    collect_files,
    optimize_png,
    should_skip_image,
)
from .dialogs import (
    CustomResolutionDialog,
    AnimatedOutputDialog,
    PngOptionsDialog,
    SettingsDialog,
    NotificationsDialog,
    LogDialog,
    ModelQueueDialog,
    ComparisonDialog,
    CropPreviewDialog,
)
from .dependencies_window import DependenciesWindow
from .upscaler import ImageUpscaler
from .sharpening import cas_sharpen_pil, cas_sharpen_array
from .resize import resize_pil, resize_array, compute_scaled_size
from .fast_io import (
    load_image_fast,
    save_image_fast,
    AsyncImageSaver,
    AsyncImageLoader,
    array_to_pil,
    extract_image_metadata,
    CV2_AVAILABLE,
)
from .theme import ThemeManager, AVAILABLE_THEMES, DEFAULT_THEME


class DropLineEdit(QLineEdit):
    """Line edit that accepts dropped folders."""

    folder_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self.setText(path)
                self.folder_dropped.emit(path)
            elif os.path.isfile(path):
                # If file dropped, use its parent directory
                self.setText(os.path.dirname(path))
                self.folder_dropped.emit(os.path.dirname(path))


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
            )

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

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None)

    def cancel(self):
        self._cancelled = True


class ThumbnailLabel(QLabel):
    """Label that displays image with pan and zoom support.

    - Ctrl+Scroll to zoom in/out
    - Click and drag to pan when zoomed in
    - Double-click to reset to fit view
    """

    MAX_WIDTH = 1280
    MAX_HEIGHT = 640
    ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # Available zoom levels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(QSize(400, 200))
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)
        self._clear()
        self._current_path = None
        self._dimensions = None
        self._original_pixmap: Optional[QPixmap] = None
        self._pil_image = None  # Cache full resolution image

        # Zoom and pan state
        self._zoom_index = 0  # Index into ZOOM_LEVELS, 0 = fit
        self._zoom_scale = 1.0  # Current zoom scale (1.0 = 100%)
        self._is_fit_mode = True  # True = fit to view, False = custom zoom
        self._pan_x = 0.5  # Pan position (0-1, center of view in image)
        self._pan_y = 0.5
        self._dragging = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_pan_x = 0.0
        self._drag_start_pan_y = 0.0

    def _clear(self):
        self.setText("(no thumbnail)")
        self.setPixmap(QPixmap())
        self._current_path = None
        self._dimensions = None
        self._original_pixmap = None
        self._pil_image = None
        self._zoom_index = 0
        self._zoom_scale = 1.0
        self._is_fit_mode = True
        self._pan_x = 0.5
        self._pan_y = 0.5

    def load_image(self, path: str):
        """Load and display thumbnail using PIL with Lanczos resampling."""
        try:
            self._current_path = path

            # Load with PIL to get dimensions and do Lanczos resize
            from PIL import Image as PILImage
            from PIL.ImageQt import ImageQt
            pil_img = PILImage.open(path)

            # Get original dimensions before any processing
            self._dimensions = (pil_img.width, pil_img.height)

            # Ensure we have a standard mode for display
            if pil_img.mode not in ("RGB", "RGBA"):
                if pil_img.mode == "P" and "transparency" in pil_img.info:
                    pil_img = pil_img.convert("RGBA")
                elif pil_img.mode in ("LA", "PA"):
                    pil_img = pil_img.convert("RGBA")
                else:
                    pil_img = pil_img.convert("RGB")

            # Cache full resolution for zoom
            self._pil_image = pil_img.copy()

            # Reset to fit mode
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5
            self._update_display()

        except Exception as e:
            self.setText(f"Error: {e}")
            self._dimensions = None
            self._pil_image = None

    def _update_display(self):
        """Update the displayed image based on current zoom and pan."""
        if not self._pil_image:
            return

        from PIL import Image as PILImage
        from PIL.ImageQt import ImageQt

        pil_img = self._pil_image
        img_w, img_h = pil_img.width, pil_img.height
        view_w, view_h = self.MAX_WIDTH, self.MAX_HEIGHT

        if self._is_fit_mode:
            # Fit to view
            scale = min(view_w / img_w, view_h / img_h, 1.0)
            self._zoom_scale = scale
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            if scale < 1.0:
                resized = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            else:
                resized = pil_img

            qimg = ImageQt(resized)
            pix = QPixmap.fromImage(qimg)
            self._original_pixmap = pix
            self.setPixmap(pix)
            self.setCursor(Qt.ArrowCursor)
        else:
            # Custom zoom - crop visible region
            scale = self._zoom_scale

            # Size of view in image coordinates
            view_img_w = view_w / scale
            view_img_h = view_h / scale

            # Calculate crop region centered on pan position
            center_x = self._pan_x * img_w
            center_y = self._pan_y * img_h

            left = center_x - view_img_w / 2
            top = center_y - view_img_h / 2

            # Clamp to image bounds
            left = max(0, min(left, img_w - view_img_w))
            top = max(0, min(top, img_h - view_img_h))
            right = min(img_w, left + view_img_w)
            bottom = min(img_h, top + view_img_h)

            # Update pan to reflect clamped position
            if view_img_w < img_w:
                self._pan_x = (left + view_img_w / 2) / img_w
            if view_img_h < img_h:
                self._pan_y = (top + view_img_h / 2) / img_h

            # Crop and scale
            cropped = pil_img.crop((int(left), int(top), int(right), int(bottom)))

            # Scale to view size
            out_w = int(cropped.width * scale)
            out_h = int(cropped.height * scale)

            if scale != 1.0:
                resized = cropped.resize((out_w, out_h), PILImage.Resampling.LANCZOS)
            else:
                resized = cropped

            qimg = ImageQt(resized)
            pix = QPixmap.fromImage(qimg)
            self.setPixmap(pix)

            # Show grab cursor if image is larger than view
            if img_w * scale > view_w or img_h * scale > view_h:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

        self.setText("")

    def toggle_zoom(self):
        """Toggle between fit-to-view and 100% zoom."""
        if not self._pil_image:
            return

        if self._is_fit_mode:
            # Switch to 100% zoom
            self._is_fit_mode = False
            self._zoom_scale = 1.0
        else:
            # Switch to fit mode
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5

        self._update_display()
        return 0 if self._is_fit_mode else 1

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming (Ctrl+Scroll)."""
        if not self._pil_image:
            return

        # Check for Ctrl modifier
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()

            if delta > 0:
                # Zoom in
                self._zoom_scale = min(self._zoom_scale * 1.25, 8.0)
            else:
                # Zoom out
                self._zoom_scale = max(self._zoom_scale / 1.25, 0.1)

            # Exit fit mode when zooming
            self._is_fit_mode = False

            # Check if we should return to fit mode
            fit_scale = min(self.MAX_WIDTH / self._pil_image.width,
                           self.MAX_HEIGHT / self._pil_image.height, 1.0)
            if abs(self._zoom_scale - fit_scale) < 0.05:
                self._is_fit_mode = True
                self._zoom_scale = fit_scale

            self._update_display()
            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        """Start dragging to pan."""
        if event.button() == Qt.LeftButton and not self._is_fit_mode:
            self._dragging = True
            self._drag_start_x = event.pos().x()
            self._drag_start_y = event.pos().y()
            self._drag_start_pan_x = self._pan_x
            self._drag_start_pan_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        """Pan the image while dragging."""
        if self._dragging and self._pil_image:
            dx = event.pos().x() - self._drag_start_x
            dy = event.pos().y() - self._drag_start_y

            # Convert pixel delta to image coordinates
            img_w, img_h = self._pil_image.width, self._pil_image.height

            # Pan amount in image fraction
            pan_dx = -dx / (img_w * self._zoom_scale)
            pan_dy = -dy / (img_h * self._zoom_scale)

            self._pan_x = max(0, min(1, self._drag_start_pan_x + pan_dx))
            self._pan_y = max(0, min(1, self._drag_start_pan_y + pan_dy))

            self._update_display()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            if not self._is_fit_mode:
                self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset to fit view."""
        if event.button() == Qt.LeftButton:
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5
            self._update_display()
            event.accept()
        else:
            event.ignore()

    def get_zoom_level(self) -> int:
        """Return current zoom level (0=fit, 1=zoomed)."""
        return 0 if self._is_fit_mode else 1

    def get_zoom_scale(self) -> float:
        """Return current zoom scale (1.0 = 100%)."""
        return self._zoom_scale

    def get_dimensions(self):
        """Return (width, height) of loaded image."""
        return self._dimensions

    def get_current_path(self):
        """Return the path of the currently displayed image."""
        return self._current_path

    def clear_image(self):
        self._clear()


class MainWindow(QMainWindow):
    """Main application window with full feature support.

    Layout matches vapoursynth_image_upscaler for consistency.
    Uses QGridLayout with thumbnail preview on right side.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorRT Upscaler v2")
        self.setAcceptDrops(True)

        self.config = get_config()
        self.worker: Optional[UpscaleWorker] = None
        self._clipboard_worker: Optional[ClipboardWorker] = None
        self.files: List[str] = []
        self.input_root: str = ""

        # Track current input dimensions
        self._current_input_width: int = 0
        self._current_input_height: int = 0

        # Timing state
        self._elapsed_timer: Optional[QTimer] = None
        self._smooth_timer: Optional[QTimer] = None  # Fast timer for smooth progress
        self._batch_start_time: float = 0.0
        self._total_files_in_batch: int = 0
        self._completed_files_in_batch: int = 0
        self._current_avg_per_image: float = 0.0
        self._current_image_start_time: float = 0.0

        # Smooth progress interpolation state
        self._last_tile_current: int = 0
        self._last_tile_total: int = 1
        self._last_tile_update_time: float = 0.0
        self._tile_time_per_unit: float = 0.0  # Estimated time per tile
        self._interpolated_tile_progress: float = 0.0  # 0.0 to 1.0
        self._last_image_duration: float = 0.0  # Duration of last completed image (for extrapolation)

        # QoL: Processing log
        self._log_entries: List[str] = []
        self._last_output_path: str = ""  # Track last output for opening folder

        # Resume/checkpoint state
        self._checkpoint_files: List[str] = []
        self._checkpoint_index: int = 0
        self._checkpoint_onnx: str = ""
        self._checkpoint_input_root: str = ""

        # Multi-model queue state
        self._model_queue: List[str] = []
        self._model_queue_index: int = 0
        self._original_files: List[str] = []  # Store original file list for multi-model processing

        # System tray icon
        self._tray_icon: Optional[QSystemTrayIcon] = None

        # Watch folder mode
        self._watch_folder: Optional[str] = None
        self._file_watcher: Optional[QFileSystemWatcher] = None
        self._watch_pending_files: List[str] = []
        self._watch_processing: bool = False
        self._watch_delay_timer: Optional[QTimer] = None
        self._watch_existing_files: set = set()

        self._create_widgets()
        self._create_menus()
        self._build_ui()
        self._connect_signals()
        self._load_settings()
        self._setup_tray_icon()
        self._setup_keyboard_shortcuts()

        # Enable drag & drop and Ctrl+V handling
        self._input_edit.installEventFilter(self)

    def _create_widgets(self):
        """Create all UI widgets."""
        # Input/output line edits
        self._input_edit = QLineEdit()
        self._output_edit = DropLineEdit()  # Supports drag-drop folders
        self._output_edit.setPlaceholderText("Drop folder here or browse...")
        self.onnx_edit = QLineEdit()

        # Tile combos (matching vapoursynth_image_upscaler style)
        self._tile_w_combo = QComboBox()
        self._tile_h_combo = QComboBox()
        for combo in (self._tile_w_combo, self._tile_h_combo):
            combo.setEditable(True)
            combo.addItems(["512", "768", "1024", "1088", "1536", "1920"])
            combo.setFixedWidth(100)
        self._tile_w_combo.setCurrentText("1088")
        self._tile_h_combo.setCurrentText("1920")

        # Auto-detect tile size button
        self._btn_auto_tile = QPushButton("Auto")
        self._btn_auto_tile.setToolTip("Auto-detect optimal tile size based on GPU VRAM")
        self._btn_auto_tile.setFixedWidth(60)

        # Checkboxes
        self._same_dir_check = QCheckBox("Save next to input with suffix:")
        self._same_dir_suffix_edit = QLineEdit("_upscaled")
        self._manga_folder_check = QCheckBox("Manga folder")
        self._manga_folder_check.setToolTip("Output: ParentFolder_suffix/Subfolder/.../Image.png")
        self._append_model_suffix_check = QCheckBox("Append model suffix")
        self._overwrite_check = QCheckBox("Overwrite")
        self._overwrite_check.setChecked(True)

        # Precision checkboxes
        self._fp16_check = QCheckBox("fp16")
        self._bf16_check = QCheckBox("bf16")
        self._bf16_check.setChecked(True)

        # Sharpening widgets
        self._sharpen_check = QCheckBox("Sharpen")
        self._sharpen_value_edit = QLineEdit("0.50")
        self._sharpen_value_edit.setFixedWidth(50)
        self._sharpen_value_edit.setEnabled(False)

        # Labels
        self._current_file_label = QLabel("Current file: (none)")
        self._progress_label = QLabel("Idle")
        self._time_label = QLabel("Elapsed: 00:00:00 | ETA: --:--:--")
        self._avg_label = QLabel("Avg per image: -")

        # Dual progress bars: batch (file) progress and tile progress
        # Use 1000 steps for smooth visual interpolation
        self._batch_progress_bar = QProgressBar()
        self._batch_progress_bar.setRange(0, 1000)
        self._batch_progress_bar.setFormat("Batch: %p%")
        self._batch_progress_bar.setTextVisible(True)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setFormat("Tiles: %p%")
        self._progress_bar.setTextVisible(True)
        self._thumbnail_label = ThumbnailLabel()
        self._image_info_label = QLabel("")
        self._image_info_label.setAlignment(Qt.AlignCenter)

        # Zoom button for thumbnail
        self._btn_zoom = QPushButton("Zoom 100%")
        self._btn_zoom.setToolTip("Toggle between fit-to-view and 100% zoom (click image or press Z)")

        # Compare button for before/after
        self._btn_compare = QPushButton("Compare")
        self._btn_compare.setToolTip("Open before/after split comparison view")

        # Crop preview button
        self._btn_crop_preview = QPushButton("Crop")
        self._btn_crop_preview.setToolTip("Select a region to preview upscale quality")

        # Buttons
        self._btn_in_file = QPushButton("Browse File")
        self._btn_in_folder = QPushButton("Browse Folder")
        self._btn_out = QPushButton("Browse")
        self._btn_clear_output = QPushButton("Clear")
        self._btn_onnx = QPushButton("Browse ONNX")
        self._btn_recent_onnx = QPushButton("Recent")
        self._btn_recent_onnx.setToolTip("Select from recently used ONNX models")
        self._btn_recent_onnx.setFixedWidth(70)
        self._btn_model_queue = QPushButton("Queue")
        self._btn_model_queue.setToolTip("Configure model queue (process with multiple models)")
        self._btn_model_queue.setFixedWidth(70)
        self._upscale_check = QCheckBox("Upscale")
        self._upscale_check.setChecked(True)
        self._upscale_check.setToolTip("Enable SR upscaling. When disabled, only applies resolution/alpha processing.")
        self._start_button = QPushButton("Start")
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._resume_button = QPushButton("Resume")
        self._resume_button.setToolTip("Resume interrupted batch from checkpoint")
        self._resume_button.setEnabled(False)
        self._clipboard_button = QPushButton("To Clipboard")
        self._clipboard_button.setToolTip("Upscale and copy result to clipboard (single image only)")
        self._custom_res_button = QPushButton("Resolution")
        self._animated_output_button = QPushButton("Animated Output")
        self._animated_output_button.setToolTip("Configure output format for animated content (GIF, WebP, AVIF)")
        self._png_options_button = QPushButton("PNG Options")
        self._png_options_button.setToolTip("Configure PNG optimization (pngquant, pingo)")

        # Theme dropdown
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(AVAILABLE_THEMES)
        self._theme_combo.setToolTip("Select UI theme")
        self._theme_combo.setFixedWidth(80)

        # QoL buttons
        self._settings_button = QPushButton("Settings")
        self._settings_button.setToolTip("Processing settings, filters, and presets")
        self._notifications_button = QPushButton("Notifications")
        self._notifications_button.setToolTip("Notification and window behavior settings")
        self._open_output_button = QPushButton("Open Output")
        self._open_output_button.setToolTip("Open output folder in Explorer")
        self._log_button = QPushButton("Log")
        self._log_button.setToolTip("View processing log and export batch report")

        # Watch folder button
        self._watch_button = QPushButton("Watch Folder")
        self._watch_button.setToolTip("Enable watch mode to auto-process new files in a folder")
        self._watch_button.setCheckable(True)

        # File list preview panel
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._file_list.setToolTip("Double-click to preview, right-click for options")
        self._file_list_label = QLabel("Files: 0")
        self._btn_clear_files = QPushButton("Clear")
        self._btn_clear_files.setToolTip("Clear file list")
        self._btn_remove_selected = QPushButton("Remove")
        self._btn_remove_selected.setToolTip("Remove selected files from list")

    def _create_menus(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_output_action = QAction("&Open Output Folder", self)
        open_output_action.setShortcut("Ctrl+O")
        open_output_action.triggered.connect(self._open_output_folder)
        file_menu.addAction(open_output_action)

        copy_path_action = QAction("&Copy Output Path", self)
        copy_path_action.setShortcut("Ctrl+Shift+C")
        copy_path_action.triggered.connect(self._copy_output_path)
        file_menu.addAction(copy_path_action)

        file_menu.addSeparator()

        log_action = QAction("Processing &Log...", self)
        log_action.setShortcut("Ctrl+L")
        log_action.triggered.connect(self._open_log_dialog)
        file_menu.addAction(log_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")

        settings_action = QAction("&Processing Settings...", self)
        settings_action.triggered.connect(self._open_settings_dialog)
        settings_menu.addAction(settings_action)

        notifications_action = QAction("&Notifications...", self)
        notifications_action.triggered.connect(self._open_notifications_dialog)
        settings_menu.addAction(notifications_action)

        settings_menu.addSeparator()

        resolution_action = QAction("&Resolution...", self)
        resolution_action.triggered.connect(self._open_resolution_dialog)
        settings_menu.addAction(resolution_action)

        animated_action = QAction("&Animated Output...", self)
        animated_action.triggered.connect(self._open_animated_dialog)
        settings_menu.addAction(animated_action)

        png_action = QAction("&PNG Options...", self)
        png_action.triggered.connect(self._open_png_dialog)
        settings_menu.addAction(png_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Dependencies action
        deps_action = QAction("&Dependencies...", self)
        deps_action.setStatusTip("Install required dependencies")
        deps_action.triggered.connect(self._show_dependencies_window)
        tools_menu.addAction(deps_action)

    def _show_dependencies_window(self):
        """Show the dependencies installation window."""
        dialog = DependenciesWindow(self)
        dialog.exec()

    def _build_ui(self):
        """Build the main UI layout (QGridLayout matching vapoursynth_image_upscaler)."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QGridLayout()
        central.setLayout(main_layout)

        row = 0

        # Input row
        main_layout.addWidget(QLabel("Input file/folder(s):"), row, 0)
        main_layout.addWidget(self._input_edit, row, 1)
        main_layout.addWidget(self._btn_in_file, row, 2)
        main_layout.addWidget(self._btn_in_folder, row, 3)
        row += 1

        # Output row
        main_layout.addWidget(QLabel("Output folder:"), row, 0)
        main_layout.addWidget(self._output_edit, row, 1)
        main_layout.addWidget(self._btn_out, row, 2)
        main_layout.addWidget(self._btn_clear_output, row, 3)
        row += 1

        # Hint
        hint = QLabel(
            "Tip: You can drag & drop multiple images/folders/URLs, "
            "or paste an image from clipboard into the input box (Ctrl+V)."
        )
        main_layout.addWidget(hint, row, 0, 1, 4)
        row += 1

        # ONNX row with Recent and Queue buttons
        main_layout.addWidget(QLabel("ONNX Model:"), row, 0)
        main_layout.addWidget(self.onnx_edit, row, 1)
        onnx_btn_layout = QHBoxLayout()
        onnx_btn_layout.setSpacing(2)
        onnx_btn_layout.addWidget(self._btn_onnx)
        onnx_btn_layout.addWidget(self._btn_recent_onnx)
        onnx_btn_layout.addWidget(self._btn_model_queue)
        onnx_btn_container = QWidget()
        onnx_btn_container.setLayout(onnx_btn_layout)
        main_layout.addWidget(onnx_btn_container, row, 2)
        main_layout.addWidget(self._upscale_check, row, 3)
        row += 1

        # Tile group
        tile_box = QGroupBox("Tile")
        tile_layout = QHBoxLayout()
        tile_layout.setSpacing(4)
        tile_box.setLayout(tile_layout)
        tile_layout.addWidget(QLabel("Width:"))
        tile_layout.addWidget(self._tile_w_combo)
        tile_layout.addSpacing(10)
        tile_layout.addWidget(QLabel("Height:"))
        tile_layout.addWidget(self._tile_h_combo)
        tile_layout.addWidget(self._btn_auto_tile)
        tile_layout.addStretch()
        main_layout.addWidget(tile_box, row, 0, 1, 4)
        row += 1

        # Precision options and sharpening in one row
        options_row = QHBoxLayout()

        prec_box = QGroupBox("Precision")
        prec_layout = QHBoxLayout()
        prec_box.setLayout(prec_layout)
        prec_layout.addWidget(self._fp16_check)
        prec_layout.addWidget(self._bf16_check)
        options_row.addWidget(prec_box)

        sharpen_box = QGroupBox("Sharpen")
        sharpen_layout = QHBoxLayout()
        sharpen_box.setLayout(sharpen_layout)
        sharpen_layout.addWidget(self._sharpen_check)
        sharpen_layout.addWidget(self._sharpen_value_edit)
        sharpen_layout.addWidget(QLabel("(0-1)"))
        options_row.addWidget(sharpen_box)

        options_row.addStretch()

        options_container = QWidget()
        options_container.setLayout(options_row)
        main_layout.addWidget(options_container, row, 0, 1, 4)
        row += 1

        # Same-dir + suffix + manga folder
        same_dir_layout = QHBoxLayout()
        same_dir_container = QWidget()
        same_dir_container.setLayout(same_dir_layout)
        same_dir_layout.addWidget(self._same_dir_check)
        same_dir_layout.addWidget(self._same_dir_suffix_edit)
        same_dir_layout.addWidget(self._manga_folder_check)
        same_dir_layout.addStretch()
        main_layout.addWidget(same_dir_container, row, 0, 1, 4)
        row += 1

        # Options row
        opts_layout = QHBoxLayout()
        opts_container = QWidget()
        opts_container.setLayout(opts_layout)
        opts_layout.addWidget(self._overwrite_check)
        opts_layout.addWidget(self._append_model_suffix_check)
        opts_layout.addWidget(self._custom_res_button)
        opts_layout.addWidget(self._animated_output_button)
        opts_layout.addWidget(self._png_options_button)
        opts_layout.addWidget(self._settings_button)
        opts_layout.addWidget(self._notifications_button)
        opts_layout.addStretch()
        main_layout.addWidget(opts_container, row, 0, 1, 4)
        row += 1

        # Info labels
        main_layout.addWidget(self._current_file_label, row, 0, 1, 4)
        row += 1
        main_layout.addWidget(self._progress_label, row, 0, 1, 4)
        row += 1

        # Progress bars: batch progress (files) and tile progress
        main_layout.addWidget(self._batch_progress_bar, row, 0, 1, 4)
        row += 1
        main_layout.addWidget(self._progress_bar, row, 0, 1, 4)
        row += 1

        main_layout.addWidget(self._time_label, row, 0, 1, 4)
        row += 1
        main_layout.addWidget(self._avg_label, row, 0, 1, 4)
        row += 1

        # Right panel with thumbnail and file list
        right_panel = QVBoxLayout()

        # Thumbnail
        thumb_box = QGroupBox("Preview")
        thumb_layout = QVBoxLayout()
        thumb_box.setLayout(thumb_layout)
        thumb_layout.addWidget(self._thumbnail_label)
        thumb_info_row = QHBoxLayout()
        thumb_info_row.addWidget(self._image_info_label)
        thumb_info_row.addStretch()
        thumb_info_row.addWidget(self._btn_zoom)
        thumb_info_row.addWidget(self._btn_compare)
        thumb_info_row.addWidget(self._btn_crop_preview)
        thumb_layout.addLayout(thumb_info_row)
        right_panel.addWidget(thumb_box)

        # File list panel
        file_list_box = QGroupBox("File Queue")
        file_list_layout = QVBoxLayout()
        file_list_box.setLayout(file_list_layout)
        file_list_layout.addWidget(self._file_list)
        file_list_btn_row = QHBoxLayout()
        file_list_btn_row.addWidget(self._file_list_label)
        file_list_btn_row.addStretch()
        file_list_btn_row.addWidget(self._btn_remove_selected)
        file_list_btn_row.addWidget(self._btn_clear_files)
        file_list_layout.addLayout(file_list_btn_row)
        right_panel.addWidget(file_list_box)

        right_container = QWidget()
        right_container.setLayout(right_panel)
        main_layout.addWidget(right_container, 0, 4, row, 1)

        # Buttons at bottom (Start, Cancel, Resume, To Clipboard, Watch, Open Output, Log, Theme)
        btn_layout = QHBoxLayout()
        btn_container = QWidget()
        btn_container.setLayout(btn_layout)
        btn_layout.addWidget(self._start_button, 2)
        btn_layout.addWidget(self._cancel_button, 1)
        btn_layout.addWidget(self._resume_button, 1)
        btn_layout.addWidget(self._clipboard_button, 1)
        btn_layout.addWidget(self._watch_button, 1)
        btn_layout.addWidget(self._open_output_button, 1)
        btn_layout.addWidget(self._log_button, 1)
        btn_layout.addWidget(QLabel("Theme:"))
        btn_layout.addWidget(self._theme_combo)
        main_layout.addWidget(btn_container, row, 0, 1, 5)

        self.resize(1400, 750)

    def _connect_signals(self):
        """Connect widget signals to slots."""
        self._btn_in_file.clicked.connect(self._browse_input_file)
        self._btn_in_folder.clicked.connect(self._browse_input_folder)
        self._btn_out.clicked.connect(self._browse_output)
        self._btn_clear_output.clicked.connect(self._clear_output_folder)
        self._btn_onnx.clicked.connect(self._browse_onnx)
        self._btn_recent_onnx.clicked.connect(self._show_recent_onnx)
        self._btn_model_queue.clicked.connect(self._open_model_queue_dialog)
        self._start_button.clicked.connect(self._start_upscaling)
        self._cancel_button.clicked.connect(self._cancel)
        self._clipboard_button.clicked.connect(self._copy_to_clipboard)
        self._custom_res_button.clicked.connect(self._open_resolution_dialog)
        self._animated_output_button.clicked.connect(self._open_animated_dialog)
        self._png_options_button.clicked.connect(self._open_png_dialog)
        self._sharpen_check.toggled.connect(self._on_sharpen_toggled)
        self._manga_folder_check.toggled.connect(self._on_manga_folder_toggled)
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)

        # QoL button connections
        self._settings_button.clicked.connect(self._open_settings_dialog)
        self._notifications_button.clicked.connect(self._open_notifications_dialog)
        self._open_output_button.clicked.connect(self._open_output_folder)
        self._log_button.clicked.connect(self._open_log_dialog)

        # File list connections
        self._file_list.itemDoubleClicked.connect(self._on_file_list_double_click)
        self._file_list.itemSelectionChanged.connect(self._on_file_list_selection_changed)
        self._btn_clear_files.clicked.connect(self._clear_file_list)
        self._btn_remove_selected.clicked.connect(self._remove_selected_files)

        # Auto tile size
        self._btn_auto_tile.clicked.connect(self._auto_detect_tile_size)

        # Watch folder
        self._watch_button.clicked.connect(self._toggle_watch_folder)

        # Resume button
        self._resume_button.clicked.connect(self._resume_batch)

        # Zoom button
        self._btn_zoom.clicked.connect(self._toggle_zoom)

        # Compare button
        self._btn_compare.clicked.connect(self._open_comparison_dialog)

        # Crop preview button
        self._btn_crop_preview.clicked.connect(self._open_crop_preview_dialog)

    def _on_sharpen_toggled(self, checked: bool):
        """Handle sharpen checkbox toggle."""
        self._sharpen_value_edit.setEnabled(checked)

    def _on_manga_folder_toggled(self, checked: bool):
        """Handle manga folder checkbox toggle - disables same_dir when enabled."""
        if checked:
            self._same_dir_check.setChecked(False)
            self._same_dir_check.setEnabled(False)
        else:
            self._same_dir_check.setEnabled(True)

    def _on_theme_changed(self, theme_name: str):
        """Handle theme dropdown change."""
        ThemeManager.apply_theme(theme_name)
        self.config.theme = theme_name
        save_config()

    def _update_file_list_ui(self):
        """Update the file list widget with current files."""
        self._file_list.clear()
        for file_path in self.files:
            item = QListWidgetItem(os.path.basename(file_path))
            item.setData(Qt.UserRole, file_path)  # Store full path
            item.setToolTip(file_path)
            self._file_list.addItem(item)
        self._file_list_label.setText(f"Files: {len(self.files)}")

    def _on_file_list_double_click(self, item: QListWidgetItem):
        """Handle double-click on file list item - show preview."""
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            self._update_thumbnail(file_path)

    def _on_file_list_selection_changed(self):
        """Handle selection change - update preview with first selected."""
        selected = self._file_list.selectedItems()
        if selected:
            file_path = selected[0].data(Qt.UserRole)
            if file_path and os.path.exists(file_path):
                self._update_thumbnail(file_path)

    def _clear_file_list(self):
        """Clear all files from the list."""
        self.files.clear()
        self._update_file_list_ui()
        self._input_edit.clear()
        self._thumbnail_label.clear()
        self._image_info_label.setText("")

    def _remove_selected_files(self):
        """Remove selected files from the list."""
        selected = self._file_list.selectedItems()
        if not selected:
            return

        # Get paths to remove
        paths_to_remove = {item.data(Qt.UserRole) for item in selected}

        # Remove from files list
        self.files = [f for f in self.files if f not in paths_to_remove]

        # Update UI
        self._update_file_list_ui()

        # Update input edit if needed
        if self.files:
            self._input_edit.setText(f"{len(self.files)} files")
        else:
            self._input_edit.clear()

    def _toggle_zoom(self):
        """Toggle zoom level on thumbnail preview."""
        zoom_level = self._thumbnail_label.toggle_zoom()
        if zoom_level == 0:
            self._btn_zoom.setText("Zoom 100%")
        else:
            self._btn_zoom.setText("Fit View")

    def _auto_detect_tile_size(self):
        """Auto-detect optimal tile size based on GPU VRAM."""
        try:
            # Try to get GPU VRAM using torch/cuda
            vram_gb = None
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    vram_gb = props.total_memory / (1024**3)
            except ImportError:
                pass

            if vram_gb is None:
                # Fallback: try pynvml
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_gb = info.total / (1024**3)
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

            if vram_gb is None:
                QMessageBox.warning(
                    self, "Detection Failed",
                    "Could not detect GPU VRAM. Please set tile size manually.\n\n"
                    "Tip: Higher VRAM allows larger tiles:\n"
                    "- 6GB: 512x512\n"
                    "- 8GB: 768x768\n"
                    "- 12GB: 1024x1024\n"
                    "- 16GB+: 1088x1920"
                )
                return

            # Determine tile size based on VRAM
            # These are conservative estimates for 4x upscaling models
            if vram_gb >= 16:
                tile_w, tile_h = 1088, 1920
            elif vram_gb >= 12:
                tile_w, tile_h = 1024, 1024
            elif vram_gb >= 10:
                tile_w, tile_h = 768, 1024
            elif vram_gb >= 8:
                tile_w, tile_h = 768, 768
            elif vram_gb >= 6:
                tile_w, tile_h = 512, 768
            else:
                tile_w, tile_h = 512, 512

            self._tile_w_combo.setCurrentText(str(tile_w))
            self._tile_h_combo.setCurrentText(str(tile_h))

            QMessageBox.information(
                self, "Auto-Detect",
                f"Detected {vram_gb:.1f} GB VRAM\n\n"
                f"Recommended tile size: {tile_w}x{tile_h}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Auto-detect failed: {e}")

    def _browse_input_file(self):
        """Open file browser for input image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*.*)",
        )
        if file_path:
            self._set_inputs_from_paths([Path(file_path)])

    def _browse_input_folder(self):
        """Open folder browser for input folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self._set_inputs_from_paths([Path(folder)])

    def _clear_output_folder(self):
        """Clear the output path."""
        self._output_edit.clear()

    def _set_inputs_from_paths(self, paths: List[Path]):
        """Set input paths and update UI."""
        self.files = []
        self.input_root = ""
        is_folder_input = False

        for p in paths:
            if p.is_file():
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.files.append(str(p))
            elif p.is_dir():
                is_folder_input = True
                self.input_root = str(p)
                found = collect_files([str(p)], recursive=True)
                self.files.extend(found)

        if not self.files:
            self._input_edit.clear()
            self._current_file_label.setText("Current file: (none)")
            self._thumbnail_label.clear_image()
            self._image_info_label.setText("")
            return

        # Auto-enable manga folder mode when a folder is dropped
        if is_folder_input:
            self._manga_folder_check.setChecked(True)

        # Update UI
        if len(self.files) == 1:
            self._input_edit.setText(self.files[0])
            self._update_thumbnail(self.files[0])
            self._current_file_label.setText(f"Current file: {Path(self.files[0]).name}")
        else:
            self._input_edit.setText(f"{len(self.files)} files")
            self._update_thumbnail(self.files[0])
            self._current_file_label.setText(f"Current file: (multiple, first: {Path(self.files[0]).name})")

        # Set default output path display (informational)
        # When manga folder mode: output goes to <input_root>_upscaled/...
        # When single file: output goes to <parent>/upscaled/...
        if is_folder_input and self.input_root:
            root = Path(self.input_root)
            output_preview = str(root.parent / (root.name + "_upscaled"))
            self._output_edit.setText(output_preview)
        elif not self._output_edit.text().strip():
            first = Path(self.files[0])
            self._output_edit.setText(str(first.parent / "upscaled"))

        # Update file list panel
        self._update_file_list_ui()

    def _update_thumbnail(self, image_path: str):
        """Update the thumbnail preview."""
        self._thumbnail_label.load_image(image_path)
        dims = self._thumbnail_label.get_dimensions()
        if dims:
            self._current_input_width = dims[0]
            self._current_input_height = dims[1]
            file_ext = Path(image_path).suffix.upper().lstrip(".")
            if not file_ext:
                file_ext = "Unknown"
            self._image_info_label.setText(f"{dims[0]} × {dims[1]} {file_ext}")
        else:
            self._image_info_label.setText("")

    def eventFilter(self, obj, event):
        """Handle keyboard events for input field (Ctrl+V paste)."""
        if obj is self._input_edit and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
                self._handle_clipboard_paste()
                return True
        return super().eventFilter(obj, event)

    def _handle_clipboard_paste(self):
        """Handle Ctrl+V paste in input field."""
        cb = QGuiApplication.clipboard()
        md = cb.mimeData()

        # Image directly in clipboard
        img = cb.image()
        if not img.isNull():
            try:
                temp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(temp_dir, f"clipboard-image-{int(time.time()*1000)}.png")
                img.save(tmp_path, "PNG")
                self._set_inputs_from_paths([Path(tmp_path)])
            except Exception as e:
                print(f"Clipboard image error: {e}")
            return

        # URLs
        urls = md.urls()
        if urls:
            u = urls[0]
            if u.isLocalFile():
                p = Path(u.toLocalFile())
                if p.exists():
                    self._set_inputs_from_paths([p])
                    return
            else:
                s = u.toString()
                if s.lower().startswith(("http://", "https://")):
                    path = download_url(s)
                    if path:
                        self._set_inputs_from_paths([Path(path)])
                        return

        # Plain text URL
        text = md.text().strip() if md.hasText() else ""
        if text.lower().startswith(("http://", "https://")):
            path = download_url(text)
            if path:
                self._set_inputs_from_paths([Path(path)])

    def _load_settings(self):
        """Load settings from config."""
        cfg = self.config

        # Model
        if cfg.onnx_path and os.path.exists(cfg.onnx_path):
            self.onnx_edit.setText(cfg.onnx_path)
        else:
            default_onnx = r"C:\Executables\models\HAT\HAT_L_4x_bf16.onnx"
            if os.path.exists(default_onnx):
                self.onnx_edit.setText(default_onnx)

        self._tile_w_combo.setCurrentText(str(cfg.tile_width))
        self._tile_h_combo.setCurrentText(str(cfg.tile_height))
        self._upscale_check.setChecked(cfg.upscale_enabled)
        self._fp16_check.setChecked(cfg.use_fp16)
        self._bf16_check.setChecked(cfg.use_bf16)

        # Output
        if cfg.last_output_path:
            self._output_edit.setText(cfg.last_output_path)
        self._same_dir_suffix_edit.setText(cfg.same_dir_suffix)
        self._same_dir_check.setChecked(cfg.save_next_to_input)
        self._manga_folder_check.setChecked(cfg.manga_folder_mode)
        self._overwrite_check.setChecked(cfg.overwrite_existing)
        self._append_model_suffix_check.setChecked(cfg.append_model_suffix)

        # Sharpening
        self._sharpen_check.setChecked(cfg.sharpen_enabled)
        self._sharpen_value_edit.setText(f"{cfg.sharpen_value:.2f}")
        self._sharpen_value_edit.setEnabled(cfg.sharpen_enabled)

        # Theme
        theme_name = cfg.theme.capitalize() if cfg.theme else DEFAULT_THEME
        if theme_name in AVAILABLE_THEMES:
            self._theme_combo.setCurrentText(theme_name)
            ThemeManager.initialize(QApplication.instance(), theme_name)
        else:
            ThemeManager.initialize(QApplication.instance(), DEFAULT_THEME)

        # Apply window flags (always on top)
        if cfg.always_on_top:
            self._apply_window_flags()

    def _save_settings(self):
        """Save settings to config."""
        cfg = self.config

        # Model
        cfg.onnx_path = self.onnx_edit.text()
        try:
            cfg.tile_width = int(self._tile_w_combo.currentText())
            cfg.tile_height = int(self._tile_h_combo.currentText())
        except ValueError:
            pass
        cfg.upscale_enabled = self._upscale_check.isChecked()
        cfg.use_fp16 = self._fp16_check.isChecked()
        cfg.use_bf16 = self._bf16_check.isChecked()

        # Output
        cfg.last_output_path = self._output_edit.text()
        cfg.same_dir_suffix = self._same_dir_suffix_edit.text()
        cfg.save_next_to_input = self._same_dir_check.isChecked()
        cfg.manga_folder_mode = self._manga_folder_check.isChecked()
        cfg.overwrite_existing = self._overwrite_check.isChecked()
        cfg.append_model_suffix = self._append_model_suffix_check.isChecked()

        # Sharpening
        cfg.sharpen_enabled = self._sharpen_check.isChecked()
        try:
            cfg.sharpen_value = float(self._sharpen_value_edit.text())
        except ValueError:
            pass

        # Theme
        cfg.theme = self._theme_combo.currentText().lower()

        save_config()

    def _browse_onnx(self):
        """Browse for ONNX model file."""
        start_dir = self.config.last_onnx_directory or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ONNX Model",
            start_dir,
            "ONNX Models (*.onnx);;All Files (*.*)"
        )
        if path:
            self.onnx_edit.setText(path)
            self.config.last_onnx_directory = os.path.dirname(path)
            self._add_to_recent_onnx(path)

    def _add_to_recent_onnx(self, path: str):
        """Add path to recent ONNX list."""
        try:
            recent = json.loads(self.config.recent_onnx_paths)
        except json.JSONDecodeError:
            recent = []

        # Remove if already exists (to move to top)
        if path in recent:
            recent.remove(path)

        # Add to front
        recent.insert(0, path)

        # Limit to max items
        max_items = self.config.max_recent_items
        recent = recent[:max_items]

        self.config.recent_onnx_paths = json.dumps(recent)
        save_config()

    def _show_recent_onnx(self):
        """Show dropdown menu with recent ONNX models."""
        try:
            recent = json.loads(self.config.recent_onnx_paths)
        except json.JSONDecodeError:
            recent = []

        if not recent:
            QMessageBox.information(self, "No Recent", "No recent ONNX models found.")
            return

        # Create popup menu
        menu = QMenu(self)
        for path in recent:
            if os.path.exists(path):
                # Show just filename but store full path
                action = menu.addAction(os.path.basename(path))
                action.setToolTip(path)
                action.setData(path)

        if menu.isEmpty():
            QMessageBox.information(self, "No Recent", "No recent ONNX models found (files may have been moved).")
            return

        # Show menu below button
        action = menu.exec_(self._btn_recent_onnx.mapToGlobal(
            self._btn_recent_onnx.rect().bottomLeft()
        ))

        if action:
            path = action.data()
            self.onnx_edit.setText(path)
            self._add_to_recent_onnx(path)  # Move to top

    def _browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._output_edit.text() or ""
        )
        if path:
            self._output_edit.setText(path)

    # Drag and drop support
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter - accept files and URLs."""
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasText() or mime.hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop - files, folders, URLs."""
        mime = event.mimeData()
        paths_to_add: List[Path] = []

        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    paths_to_add.append(Path(url.toLocalFile()))
                else:
                    # Remote URL - download
                    url_str = url.toString()
                    path = download_url(url_str)
                    if path:
                        paths_to_add.append(Path(path))

        elif mime.hasText():
            text = mime.text().strip()
            url = extract_url_from_text(text)
            if url:
                path = download_url(url)
                if path:
                    paths_to_add.append(Path(path))

        if paths_to_add:
            self._set_inputs_from_paths(paths_to_add)

    def _copy_to_clipboard(self):
        """Upscale first image and copy result to clipboard."""
        if not self.files:
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return

        onnx_path = self.onnx_edit.text().strip()
        if not onnx_path or not os.path.exists(onnx_path):
            QMessageBox.warning(self, "No Model", "Please select an ONNX model first.")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Please wait for current operation to finish.")
            return

        # Disable button during processing
        self._clipboard_button.setEnabled(False)
        self._progress_label.setText("Upscaling to clipboard...")
        self._progress_bar.setValue(0)

        # Start clipboard worker
        self._clipboard_worker = ClipboardWorker(self.files[0], self.config, onnx_path)
        self._clipboard_worker.progress.connect(self._on_clipboard_progress)
        self._clipboard_worker.finished.connect(self._on_clipboard_finished)
        self._clipboard_worker.start()

    def _on_clipboard_progress(self, current: int, total: int):
        """Update progress bar for clipboard upscale."""
        if total > 0:
            pct = int(current / total * 1000)
            self._progress_bar.setValue(pct)

    def _on_clipboard_finished(self, success: bool, message: str, qimg):
        """Handle clipboard upscale completion."""
        self._clipboard_button.setEnabled(True)
        self._progress_bar.setValue(1000 if success else 0)

        if success and qimg is not None:
            cb = QGuiApplication.clipboard()
            cb.setImage(qimg)
            self._progress_label.setText("Upscaled image copied to clipboard")
        else:
            self._progress_label.setText(f"Clipboard failed: {message}")
            if not success:
                QMessageBox.warning(self, "Error", f"Failed to upscale: {message}")

    # Dialog openers
    def _open_resolution_dialog(self):
        """Open resolution settings dialog."""
        dlg = CustomResolutionDialog(self)
        dlg.exec_()

    def _open_animated_dialog(self):
        """Open animated output settings dialog."""
        dlg = AnimatedOutputDialog(self)
        dlg.exec_()

    def _open_png_dialog(self):
        """Open PNG options dialog."""
        dlg = PngOptionsDialog(self)
        dlg.exec_()

    def _open_settings_dialog(self):
        """Open settings dialog (filters, presets, etc.)."""
        dlg = SettingsDialog(self)
        if dlg.exec_():
            # Reload UI from config if preset was loaded
            self._load_settings()

    def _open_notifications_dialog(self):
        """Open notifications and window behavior dialog."""
        dlg = NotificationsDialog(self)
        if dlg.exec_():
            # Apply always-on-top setting immediately
            self._apply_window_flags()

    def _open_log_dialog(self):
        """Open processing log dialog."""
        dlg = LogDialog(self._log_entries, self)
        dlg.exec_()

    def _open_model_queue_dialog(self):
        """Open model queue dialog."""
        dlg = ModelQueueDialog(self)
        dlg.exec_()

    def _open_comparison_dialog(self):
        """Open before/after comparison dialog with in-memory upscaling support."""
        try:
            before_path = ""
            after_path = ""

            # Get current preview image as before
            current_path = self._thumbnail_label.get_current_path()
            if current_path and os.path.exists(current_path):
                # If showing output, try to find the original input
                if self._last_output_path and current_path == self._last_output_path:
                    # We're showing the output - use last input as before
                    if self.files:
                        before_path = self.files[-1] if len(self.files) == 1 else ""
                    after_path = self._last_output_path
                else:
                    before_path = current_path
                    if self._last_output_path and os.path.exists(self._last_output_path):
                        after_path = self._last_output_path

            # If no before path but we have files, use the first one
            if not before_path and self.files:
                before_path = self.files[0]

            # Get ONNX path for in-memory upscaling
            onnx_path = self.onnx_edit.text().strip()

            # Pass config and onnx_path for in-memory upscaling capability
            dlg = ComparisonDialog(
                before_path=before_path,
                after_path=after_path,
                parent=self,
                config=self.config,
                onnx_path=onnx_path if onnx_path and os.path.exists(onnx_path) else ""
            )
            dlg.exec()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to open comparison dialog: {e}")

    def _open_crop_preview_dialog(self):
        """Open crop region preview dialog."""
        # Get current image path
        image_path = self._thumbnail_label.get_current_path()
        if not image_path or not os.path.exists(image_path):
            # Try to use first file in queue
            if self.files:
                image_path = self.files[0]
            else:
                QMessageBox.warning(self, "No Image", "Please load an image first.")
                return

        # Get ONNX path
        onnx_path = self.onnx_edit.text()
        if not onnx_path or not os.path.exists(onnx_path):
            QMessageBox.warning(self, "No Model", "Please select a valid ONNX model first.")
            return

        dlg = CropPreviewDialog(image_path, onnx_path, self.config, self)
        dlg.exec_()

    def _apply_window_flags(self):
        """Apply window flags based on settings."""
        cfg = self.config
        flags = self.windowFlags()
        if cfg.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()  # Need to show again after changing flags

    def _open_output_folder(self):
        """Open output folder in file explorer."""
        # Try last processed output path first
        if self._last_output_path and os.path.exists(os.path.dirname(self._last_output_path)):
            folder = os.path.dirname(self._last_output_path)
        elif self._output_edit.text() and os.path.exists(self._output_edit.text()):
            folder = self._output_edit.text()
        else:
            QMessageBox.information(self, "No Output", "No output folder available yet.")
            return

        # Open folder in Explorer (Windows)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])

    def _copy_output_path(self):
        """Copy last output path to clipboard."""
        if self._last_output_path:
            cb = QGuiApplication.clipboard()
            cb.setText(self._last_output_path)
            self._progress_label.setText(f"Copied: {self._last_output_path}")
        elif self._output_edit.text():
            cb = QGuiApplication.clipboard()
            cb.setText(self._output_edit.text())
            self._progress_label.setText(f"Copied: {self._output_edit.text()}")
        else:
            QMessageBox.information(self, "No Path", "No output path available yet.")

    def _start_upscaling(self):
        """Start the upscaling process."""
        if not self.files:
            QMessageBox.warning(self, "No Files", "Please add files to upscale.")
            return

        onnx_path = self.onnx_edit.text()
        if self._upscale_check.isChecked() and (not onnx_path or not os.path.exists(onnx_path)):
            QMessageBox.warning(self, "No Model", "Please select a valid ONNX model.")
            return

        # Update config from UI
        self._save_settings()

        # Check for multi-model queue mode
        cfg = self.config
        if cfg.model_queue_enabled and self._model_queue_index == 0:
            # First model in queue - setup queue state
            try:
                model_queue = json.loads(cfg.model_queue)
                if model_queue and len(model_queue) > 0:
                    # Use queue instead of single model
                    self._model_queue = [m for m in model_queue if os.path.exists(m)]
                    if self._model_queue:
                        self._model_queue_index = 0
                        self._original_files = self.files.copy()
                        onnx_path = self._model_queue[0]
                        self.onnx_edit.setText(onnx_path)
                        # Force append model suffix for multi-model
                        self._append_model_suffix_check.setChecked(True)
                        cfg.append_model_suffix = True
            except json.JSONDecodeError:
                pass

        # Initialize timing state
        self._batch_start_time = time.perf_counter()
        self._total_files_in_batch = len(self.files)
        self._completed_files_in_batch = 0
        self._current_avg_per_image = 0.0
        self._current_image_start_time = time.perf_counter()

        # Initialize smooth progress state
        self._last_tile_current = 0
        self._last_tile_total = 1
        self._last_tile_update_time = time.perf_counter()
        self._tile_time_per_unit = 0.0
        self._interpolated_tile_progress = 0.0
        # Keep _last_image_duration from previous batch for first image estimation
        # Only reset if it's the first time running
        if self._last_image_duration == 0:
            self._last_image_duration = 0.0

        # Start elapsed timer (1 second for time display)
        if self._elapsed_timer is None:
            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.timeout.connect(self._update_time_display)
        self._elapsed_timer.start(1000)

        # Start smooth progress timer (60Hz for smooth visual updates)
        if self._smooth_timer is None:
            self._smooth_timer = QTimer(self)
            self._smooth_timer.timeout.connect(self._update_smooth_progress)
        self._smooth_timer.start(16)  # ~16.67ms = 60Hz

        self._start_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

        # Initialize both progress bars (1000 steps for smooth interpolation)
        self._batch_progress_bar.setRange(0, 1000)
        self._batch_progress_bar.setValue(0)
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)

        self._progress_label.setText("Starting...")
        self._current_file_label.setText(f"Processing 0/{self._total_files_in_batch}")
        self._time_label.setText("Elapsed: 00:00:00 | ETA: --:--:--")
        self._avg_label.setText("Avg per image: -")

        # Sort files naturally
        sorted_files = sorted(self.files, key=natural_sort_key)

        self.worker = UpscaleWorker(
            files=sorted_files,
            config=self.config,
            onnx_path=onnx_path,
            input_root=self.input_root,
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.file_skipped.connect(self._on_file_skipped)
        self.worker.checkpoint_updated.connect(self._on_checkpoint_updated)
        self.worker.start()

    def _cancel(self):
        """Cancel the current operation."""
        if self.worker:
            self.worker.cancel()
            self._progress_label.setText("Cancelling...")

    def _update_time_display(self):
        """Update elapsed time and ETA labels."""
        if self._batch_start_time <= 0:
            return

        elapsed = time.perf_counter() - self._batch_start_time
        elapsed_str = format_time_hms(elapsed)

        if self._current_avg_per_image > 0 and self._completed_files_in_batch > 0:
            remaining_files = self._total_files_in_batch - self._completed_files_in_batch
            eta_seconds = remaining_files * self._current_avg_per_image
            eta_str = format_time_hms(eta_seconds)
        else:
            eta_str = "--:--:--"

        self._time_label.setText(f"Elapsed: {elapsed_str} | ETA: {eta_str}")

    def _on_progress(self, current: int, total: int):
        """Update tile progress - records timing for smooth interpolation."""
        now = time.perf_counter()

        # Calculate time per tile unit for interpolation
        if current > self._last_tile_current and self._last_tile_current > 0:
            delta_tiles = current - self._last_tile_current
            delta_time = now - self._last_tile_update_time
            if delta_tiles > 0 and delta_time > 0:
                # Exponential moving average for smoother estimation
                new_rate = delta_time / delta_tiles
                if self._tile_time_per_unit > 0:
                    self._tile_time_per_unit = 0.7 * self._tile_time_per_unit + 0.3 * new_rate
                else:
                    self._tile_time_per_unit = new_rate

        # Store current state for next update
        self._last_tile_current = current
        self._last_tile_total = total
        self._last_tile_update_time = now

        # Set base progress (will be interpolated by smooth timer)
        self._interpolated_tile_progress = current / total if total > 0 else 0.0

    def _update_smooth_progress(self):
        """Interpolate and update progress bars smoothly (called every 50ms)."""
        now = time.perf_counter()
        time_on_current = now - self._current_image_start_time

        # Interpolate tile progress using multiple strategies
        tile_progress = 0.0

        # Strategy 1: Use tile-based timing if we have real tile progress
        if self._tile_time_per_unit > 0 and self._last_tile_total > 0 and self._last_tile_current > 0:
            # Calculate how much progress we expect since last update
            time_since_update = now - self._last_tile_update_time
            expected_progress = time_since_update / self._tile_time_per_unit

            # Interpolate: base progress + fraction of next tile
            base_progress = self._last_tile_current / self._last_tile_total
            tile_progress = base_progress + (expected_progress / self._last_tile_total)

            # Clamp to reasonable bounds (don't exceed 100%, don't go below base)
            tile_progress = min(tile_progress, 1.0)
            tile_progress = max(tile_progress, base_progress)

        # Strategy 2: Use time-based extrapolation from previous image duration
        elif self._last_image_duration > 0:
            # Estimate progress based on how long previous image took
            # Use eased progress curve for more natural feel (ease-out)
            raw_progress = time_on_current / self._last_image_duration
            # Apply ease-out curve: progress accelerates at start, slows near end
            # This looks more natural and avoids overshooting
            raw_progress = min(raw_progress, 0.98)  # Cap at 98% to avoid premature completion
            # Ease-out quadratic: 1 - (1-t)^2
            tile_progress = 1.0 - (1.0 - raw_progress) ** 2
            tile_progress = max(0.0, min(tile_progress, 0.98))

        # Strategy 3: Use average time per image if no other data
        elif self._current_avg_per_image > 0:
            raw_progress = time_on_current / self._current_avg_per_image
            raw_progress = min(raw_progress, 0.98)
            tile_progress = 1.0 - (1.0 - raw_progress) ** 2
            tile_progress = max(0.0, min(tile_progress, 0.98))

        # Strategy 4: Fallback to discrete progress
        else:
            tile_progress = self._interpolated_tile_progress

        tile_value = int(tile_progress * 1000)
        self._progress_bar.setValue(tile_value)

        # Interpolate batch progress using time-based estimation
        if self._total_files_in_batch > 0:
            # Base: completed files
            completed_fraction = self._completed_files_in_batch / self._total_files_in_batch

            # Current file progress contribution
            current_file_fraction = tile_progress / self._total_files_in_batch

            # Time-based interpolation for smoother batch progress
            # If we have average time per image, estimate additional progress
            time_interpolation = 0.0
            if self._current_avg_per_image > 0 and self._current_image_start_time > 0:
                # How long have we been on current image?
                time_on_current = now - self._current_image_start_time
                # Estimate progress through current image based on time
                time_based_progress = min(time_on_current / self._current_avg_per_image, 1.0)

                # Blend tile-based and time-based progress (favor tile-based when available)
                if tile_progress > 0:
                    # Weight tile progress more heavily as it's more accurate
                    blended_progress = 0.7 * tile_progress + 0.3 * time_based_progress
                else:
                    # No tile progress yet, use time-based
                    blended_progress = time_based_progress

                current_file_fraction = blended_progress / self._total_files_in_batch

            # Smooth batch progress
            batch_progress = completed_fraction + current_file_fraction
            batch_value = int(batch_progress * 1000)
            batch_value = min(batch_value, 1000)

            self._batch_progress_bar.setValue(batch_value)

            # Update batch progress bar format to show file count
            files_done = self._completed_files_in_batch
            files_total = self._total_files_in_batch
            pct = int(batch_progress * 100)
            self._batch_progress_bar.setFormat(f"Batch: {pct}% ({files_done}/{files_total} files)")

    def _on_file_progress(self, current: int, total: int, file_path: str):
        """Update file progress and thumbnail for current image."""
        # Reset tile progress state for new file
        self._last_tile_current = 0
        self._last_tile_total = 1
        self._last_tile_update_time = time.perf_counter()
        self._interpolated_tile_progress = 0.0
        self._current_image_start_time = time.perf_counter()

        # Update thumbnail to show current file being processed
        if file_path and file_path != self._thumbnail_label.get_current_path():
            self._update_thumbnail(file_path)

        # Update labels
        file_name = Path(file_path).name if file_path else ""
        self._progress_label.setText(f"Processing image {current}/{total}: {file_name}")
        self._current_file_label.setText(f"Processing {current}/{total}")

    def _on_file_done(self, input_path: str, output_path: str, elapsed_time: float):
        """Handle completed file."""
        self._completed_files_in_batch += 1
        self._last_output_path = output_path

        # Record this image's duration for smooth progress extrapolation
        image_duration = time.perf_counter() - self._current_image_start_time
        if image_duration > 0:
            # Use exponential moving average for smoother estimation
            if self._last_image_duration > 0:
                self._last_image_duration = 0.7 * self._last_image_duration + 0.3 * image_duration
            else:
                self._last_image_duration = image_duration

        # Update average using total elapsed time (includes all overhead)
        total_elapsed = time.perf_counter() - self._batch_start_time
        self._current_avg_per_image = total_elapsed / self._completed_files_in_batch

        # Reset tile progress for next file (smooth timer will update bars)
        self._interpolated_tile_progress = 0.0
        self._last_tile_current = 0

        # Add to processing log
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({elapsed_time:.2f}s)"
        self._log_entries.append(log_entry)

        # Update display
        self._current_file_label.setText(
            f"Processed {self._completed_files_in_batch}/{self._total_files_in_batch}"
        )
        self._avg_label.setText(f"Avg per image: {self._current_avg_per_image:.2f}s")

        # Update time display immediately
        self._update_time_display()

        # Progress label
        self._progress_label.setText(f"Saved: {os.path.basename(output_path)}")

    def _on_file_skipped(self, input_path: str, reason: str):
        """Handle skipped file."""
        # Add to processing log
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] SKIPPED: {os.path.basename(input_path)} ({reason})"
        self._log_entries.append(log_entry)

        # Update progress label briefly
        self._progress_label.setText(f"Skipped: {os.path.basename(input_path)} ({reason})")

    def _on_checkpoint_updated(self, current_index: int, remaining_files: list):
        """Handle checkpoint update - save state for resume."""
        self._checkpoint_index = current_index
        self._checkpoint_files = list(remaining_files)
        self._checkpoint_onnx = self.onnx_edit.text()
        self._checkpoint_input_root = self.input_root

        # Enable resume button if we have checkpoint data
        if self._checkpoint_files:
            self._resume_button.setEnabled(True)

    def _resume_batch(self):
        """Resume processing from last checkpoint."""
        if not self._checkpoint_files:
            QMessageBox.information(self, "No Checkpoint", "No interrupted batch to resume.")
            return

        # Restore checkpoint state
        self.files = self._checkpoint_files
        self.input_root = self._checkpoint_input_root

        # Update UI
        self._update_file_list_ui()
        self._input_edit.setText(f"{len(self.files)} files (resumed)")

        if self._checkpoint_onnx:
            self.onnx_edit.setText(self._checkpoint_onnx)

        # Start processing from where we left off
        self._progress_label.setText(f"Resuming from file {self._checkpoint_index + 1}...")
        self._start_upscaling()

        # Disable resume button after starting
        self._resume_button.setEnabled(False)

    def _on_finished(self, success: bool, message: str):
        """Handle completion."""
        # Stop timers
        if self._elapsed_timer:
            self._elapsed_timer.stop()
        if self._smooth_timer:
            self._smooth_timer.stop()

        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)

        # Clear checkpoint on successful completion, keep for resume on cancel/error
        if success:
            self._checkpoint_files.clear()
            self._checkpoint_index = 0
            self._resume_button.setEnabled(False)
        else:
            # Keep checkpoint for resume
            self._resume_button.setEnabled(bool(self._checkpoint_files))

        # Update both progress bars
        if success:
            self._batch_progress_bar.setValue(1000)
            self._batch_progress_bar.setFormat(
                f"Batch: 100% ({self._total_files_in_batch}/{self._total_files_in_batch} files)"
            )
            self._progress_bar.setValue(1000)
        else:
            self._progress_bar.setValue(0)

        self._progress_label.setText(message)

        cfg = self.config

        if success:
            # Show final stats
            total_elapsed = time.perf_counter() - self._batch_start_time
            elapsed_str = format_time_hms(total_elapsed)
            self._time_label.setText(f"Total time: {elapsed_str}")

            # QoL: Show output preview (last processed image)
            if self._last_output_path and os.path.exists(self._last_output_path):
                self._update_thumbnail(self._last_output_path)
                self._image_info_label.setText(f"Output: {os.path.basename(self._last_output_path)}")

            # QoL: Play sound on completion
            if cfg.sound_on_complete:
                self._play_completion_sound()

            # QoL: Open output folder on completion
            if cfg.open_output_on_complete:
                self._open_output_folder()

            # Show notification (system tray or message box)
            if cfg.notify_on_complete:
                QMessageBox.information(
                    self,
                    "Complete",
                    f"{message}\n\n"
                    f"Total time: {elapsed_str}\n"
                    f"Average per image: {self._current_avg_per_image:.2f}s"
                )

            # QoL: Auto-shutdown after completion (not in watch mode)
            if cfg.auto_shutdown_enabled and not self._watch_folder:
                self._execute_auto_shutdown()
        else:
            QMessageBox.warning(self, "Error", message)

        self.worker = None

        # Multi-model queue: check for next model
        if success and self._model_queue and self._model_queue_index < len(self._model_queue) - 1:
            # More models to process
            self._model_queue_index += 1
            next_model = self._model_queue[self._model_queue_index]
            self.onnx_edit.setText(next_model)
            # Restore original files for next model
            self.files = self._original_files.copy()
            self._update_file_list_ui()

            # Add log entry
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self._log_entries.append(f"[{timestamp}] Starting model {self._model_queue_index + 1}/{len(self._model_queue)}: {os.path.basename(next_model)}")

            self._progress_label.setText(f"Model {self._model_queue_index + 1}/{len(self._model_queue)}: {os.path.basename(next_model)}")

            # Start next model after short delay
            QTimer.singleShot(500, self._start_upscaling)
            return  # Don't trigger completion notifications yet
        elif self._model_queue:
            # All models completed - reset queue state
            self._model_queue.clear()
            self._model_queue_index = 0
            self._original_files.clear()

        # Watch mode: reset processing flag and check for more files
        if self._watch_folder:
            self._watch_processing = False
            # Check for more pending files after a short delay
            if self._watch_pending_files:
                QTimer.singleShot(500, self._process_watch_pending)

    def _play_completion_sound(self):
        """Play a sound when processing completes."""
        cfg = self.config
        try:
            if cfg.sound_file_path and os.path.exists(cfg.sound_file_path):
                # Play custom sound file
                if sys.platform == "win32":
                    import winsound
                    winsound.PlaySound(cfg.sound_file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Play system default sound
                if sys.platform == "win32":
                    import winsound
                    winsound.MessageBeep(winsound.MB_OK)
        except Exception as e:
            print(f"Failed to play sound: {e}")

    def _execute_auto_shutdown(self):
        """Execute auto-shutdown based on settings."""
        cfg = self.config
        mode = cfg.auto_shutdown_mode

        # Show countdown warning
        reply = QMessageBox.question(
            self,
            "Auto-Shutdown",
            f"Processing complete. System will {mode} in 30 seconds.\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            if sys.platform == "win32":
                if mode == "sleep":
                    # SetSuspendState(hibernate, force, disableWakeEvent)
                    subprocess.run(["powercfg", "-h", "off"], capture_output=True)  # Disable hibernate first
                    subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
                elif mode == "hibernate":
                    subprocess.run(["shutdown", "/h"])
                elif mode == "shutdown":
                    subprocess.run(["shutdown", "/s", "/t", "30"])
            else:
                # Linux/Mac
                if mode == "sleep":
                    subprocess.run(["systemctl", "suspend"])
                elif mode == "hibernate":
                    subprocess.run(["systemctl", "hibernate"])
                elif mode == "shutdown":
                    subprocess.run(["shutdown", "-h", "+1"])
        except Exception as e:
            QMessageBox.warning(self, "Shutdown Failed", f"Failed to execute {mode}: {e}")

    def _setup_tray_icon(self):
        """Setup system tray icon for minimize to tray feature."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray_icon = QSystemTrayIcon(self)
        # Use application icon or default
        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self._tray_icon.setIcon(icon)
        self._tray_icon.setToolTip("TensorRT Upscaler v2")

        # Create tray menu
        tray_menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self._show_from_tray)
        tray_menu.addAction(show_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)

        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)

    def _show_from_tray(self):
        """Show window from tray."""
        self.showNormal()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_from_tray()

    def changeEvent(self, event):
        """Handle window state changes for minimize to tray."""
        if event.type() == QEvent.WindowStateChange:
            if self.isMinimized() and self.config.minimize_to_tray and self._tray_icon:
                # Hide to tray
                QTimer.singleShot(0, self._hide_to_tray)
        super().changeEvent(event)

    def _hide_to_tray(self):
        """Hide window to system tray."""
        self.hide()
        if self._tray_icon:
            self._tray_icon.show()
            self._tray_icon.showMessage(
                "TensorRT Upscaler",
                "Minimized to tray. Double-click to restore.",
                QSystemTrayIcon.Information,
                2000
            )

    def _setup_keyboard_shortcuts(self):
        """Setup global keyboard shortcuts."""
        if not self.config.shortcuts_enabled:
            return

        # Start/Stop processing - Enter/Escape
        start_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        start_shortcut.activated.connect(self._shortcut_start)

        cancel_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        cancel_shortcut.activated.connect(self._shortcut_cancel)

        # File operations
        # Ctrl+O - Open file
        open_file_shortcut = QShortcut(QKeySequence("Ctrl+I"), self)
        open_file_shortcut.activated.connect(self._browse_input_file)

        # Ctrl+Shift+O - Open folder
        open_folder_shortcut = QShortcut(QKeySequence("Ctrl+Shift+I"), self)
        open_folder_shortcut.activated.connect(self._browse_input_folder)

        # Ctrl+E - Open output folder
        open_output_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        open_output_shortcut.activated.connect(self._open_output_folder)

        # Ctrl+Shift+C - Copy output path
        copy_path_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_path_shortcut.activated.connect(self._copy_output_path)

        # Ctrl+L - Open log
        log_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        log_shortcut.activated.connect(self._open_log_dialog)

        # Ctrl+, - Open settings
        settings_shortcut = QShortcut(QKeySequence("Ctrl+,"), self)
        settings_shortcut.activated.connect(self._open_settings_dialog)

        # Ctrl+W - Toggle watch folder
        watch_shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        watch_shortcut.activated.connect(self._toggle_watch_folder)

        # Delete - Remove selected files from list
        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        delete_shortcut.activated.connect(self._remove_selected_files)

        # Ctrl+Delete - Clear file list
        clear_shortcut = QShortcut(QKeySequence("Ctrl+Delete"), self)
        clear_shortcut.activated.connect(self._clear_file_list)

        # F5 - Refresh/reload input
        refresh_shortcut = QShortcut(QKeySequence(Qt.Key_F5), self)
        refresh_shortcut.activated.connect(self._refresh_input)

        # Ctrl+T - Toggle always on top
        always_on_top_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        always_on_top_shortcut.activated.connect(self._toggle_always_on_top)

        # Z - Toggle zoom on preview
        zoom_shortcut = QShortcut(QKeySequence(Qt.Key_Z), self)
        zoom_shortcut.activated.connect(self._toggle_zoom)

    def _shortcut_start(self):
        """Handle Enter shortcut - start if not running."""
        if self._start_button.isEnabled():
            self._start_upscaling()

    def _shortcut_cancel(self):
        """Handle Escape shortcut - cancel if running."""
        if self._cancel_button.isEnabled():
            self._cancel()

    def _refresh_input(self):
        """Refresh input - re-collect files from current path."""
        input_text = self._input_edit.text()
        if not input_text or "files" in input_text:
            # Multiple files already loaded, nothing to refresh
            return
        if os.path.exists(input_text):
            self._set_inputs_from_paths([Path(input_text)])

    def _toggle_always_on_top(self):
        """Toggle always on top setting."""
        self.config.always_on_top = not self.config.always_on_top
        save_config()
        self._apply_window_flags()
        status = "ON" if self.config.always_on_top else "OFF"
        self._progress_label.setText(f"Always on top: {status}")

    # ===== Watch Folder Mode =====

    def _toggle_watch_folder(self):
        """Toggle watch folder mode."""
        if self._watch_folder:
            self._stop_watch_folder()
        else:
            self._start_watch_folder()

    def _start_watch_folder(self):
        """Start watching a folder for new images."""
        # Get folder to watch
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Watch",
            self._input_edit.text() or ""
        )
        if not folder:
            self._watch_button.setChecked(False)
            return

        # Validate ONNX model is set
        onnx_path = self.onnx_edit.text()
        if self._upscale_check.isChecked() and (not onnx_path or not os.path.exists(onnx_path)):
            QMessageBox.warning(self, "No Model", "Please select a valid ONNX model before starting watch mode.")
            self._watch_button.setChecked(False)
            return

        # Setup file watcher
        self._watch_folder = folder
        self._file_watcher = QFileSystemWatcher([folder], self)
        self._file_watcher.directoryChanged.connect(self._on_watch_folder_changed)

        # Setup delay timer for debouncing file changes
        self._watch_delay_timer = QTimer(self)
        self._watch_delay_timer.setSingleShot(True)
        self._watch_delay_timer.timeout.connect(self._process_watch_pending)

        # Update UI
        self._watch_button.setChecked(True)
        self._watch_button.setText("Stop Watch")
        self._progress_label.setText(f"Watching: {folder}")

        # Get initial file list to track new additions
        self._watch_existing_files = set(
            str(p) for p in Path(folder).glob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        # Add to log
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_entries.append(f"[{timestamp}] Started watching: {folder}")

    def _stop_watch_folder(self):
        """Stop watching folder."""
        if self._file_watcher:
            self._file_watcher.deleteLater()
            self._file_watcher = None

        if self._watch_delay_timer:
            self._watch_delay_timer.stop()
            self._watch_delay_timer = None

        self._watch_folder = None
        self._watch_pending_files.clear()
        self._watch_processing = False

        # Update UI
        self._watch_button.setChecked(False)
        self._watch_button.setText("Watch Folder")
        self._progress_label.setText("Watch mode stopped")

        # Add to log
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_entries.append(f"[{timestamp}] Stopped watch mode")

    def _on_watch_folder_changed(self, path: str):
        """Handle folder content change."""
        if not self._watch_folder:
            return

        # Find new files
        current_files = set(
            str(p) for p in Path(self._watch_folder).glob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        new_files = current_files - self._watch_existing_files
        self._watch_existing_files = current_files

        if new_files:
            # Add to pending queue
            for f in sorted(new_files, key=natural_sort_key):
                if f not in self._watch_pending_files:
                    self._watch_pending_files.append(f)

            # Start/restart delay timer (wait for file to finish writing)
            if self._watch_delay_timer:
                self._watch_delay_timer.start(1000)  # 1 second delay

    def _process_watch_pending(self):
        """Process pending files from watch folder."""
        if not self._watch_pending_files or self._watch_processing:
            return

        if self.worker and self.worker.isRunning():
            # Already processing, wait for completion
            return

        # Get files that are ready (fully written)
        ready_files = []
        still_pending = []

        for file_path in self._watch_pending_files:
            if self._is_file_ready(file_path):
                ready_files.append(file_path)
            else:
                still_pending.append(file_path)

        self._watch_pending_files = still_pending

        if ready_files:
            self._watch_processing = True
            # Set these as input and start processing
            self.files = ready_files
            self._update_file_list_ui()
            self._input_edit.setText(f"{len(ready_files)} new files")

            # Start processing
            self._start_upscaling()

    def _is_file_ready(self, file_path: str) -> bool:
        """Check if file is ready (not still being written)."""
        try:
            # Try to open file exclusively to check if it's still being written
            if not os.path.exists(file_path):
                return False

            # Check file size is stable
            size1 = os.path.getsize(file_path)
            time.sleep(0.1)
            size2 = os.path.getsize(file_path)

            return size1 == size2 and size1 > 0
        except (OSError, PermissionError):
            return False

    def closeEvent(self, event):
        """Save settings on close."""
        self._save_settings()
        # Stop watch mode if active
        if self._watch_folder:
            self._stop_watch_folder()
        if self._tray_icon:
            self._tray_icon.hide()
        event.accept()


def main():
    """Main entry point for GUI."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Theme is initialized in MainWindow._load_settings()
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
