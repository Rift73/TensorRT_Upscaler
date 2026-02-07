"""
Before/after comparison dialog with split view.
Supports in-memory upscaling for quick comparison without saving to disk.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QProgressBar,
)


class SplitCompareWidget(QWidget):
    """
    Widget for comparing two images with a draggable split slider.
    Shows 'before' on left and 'after' on right of the slider.

    Features:
    - Scroll to zoom in/out
    - Right-click drag to pan when zoomed
    - Left-click drag to move split slider
    - Double-click to reset to fit view
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_pil = None  # Full resolution PIL images
        self._after_pil = None
        self._split_position: float = 0.5  # 0.0 to 1.0
        self._dragging_split: bool = False
        self._dragging_pan: bool = False
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable focus for wheel events

        # Zoom and pan state
        self._zoom_scale = 1.0  # Display scale (1.0 = fit)
        self._fit_scale = 1.0  # Scale needed to fit
        self._is_fit_mode = True
        self._pan_x = 0.5  # Pan position (0-1)
        self._pan_y = 0.5
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_pan_x = 0.0
        self._drag_start_pan_y = 0.0

        # Cached display pixmaps
        self._before_pixmap: QPixmap = None
        self._after_pixmap: QPixmap = None
        self._display_w = 0
        self._display_h = 0

        # High-quality render timer (2 second delay after interaction stops)
        self._hq_render_timer = QTimer(self)
        self._hq_render_timer.setSingleShot(True)
        self._hq_render_timer.setInterval(2000)  # 2 seconds
        self._hq_render_timer.timeout.connect(self._update_display_hq)
        self._use_hq = False  # Flag for current render quality

    def set_images(self, before_path: str, after_path: str):
        """Load before and after images from file paths."""
        from PIL import Image as PILImage

        try:
            # Load before image
            pil_before = PILImage.open(before_path)
            if pil_before.mode not in ("RGB", "RGBA"):
                pil_before = pil_before.convert("RGB")

            # Load after image
            pil_after = PILImage.open(after_path)
            if pil_after.mode not in ("RGB", "RGBA"):
                pil_after = pil_after.convert("RGB")

            self.set_pil_images(pil_before, pil_after)

        except Exception as e:
            print(f"Failed to load comparison images: {e}")

    def set_pil_images(self, before_pil, after_pil):
        """Set before and after images from PIL Image objects (in-memory)."""
        from PIL import Image as PILImage

        # Ensure correct mode
        if before_pil.mode not in ("RGB", "RGBA"):
            before_pil = before_pil.convert("RGB")
        if after_pil.mode not in ("RGB", "RGBA"):
            after_pil = after_pil.convert("RGB")

        # Cache full resolution - scale before to match after size for 1:1 comparison
        self._after_pil = after_pil.copy()
        # Pre-scale before image to after size (do this once, not on every update)
        if before_pil.size != after_pil.size:
            self._before_pil = before_pil.resize(after_pil.size, PILImage.Resampling.LANCZOS)
        else:
            self._before_pil = before_pil.copy()

        # Reset view
        self._is_fit_mode = True
        self._pan_x = 0.5
        self._pan_y = 0.5
        self._update_display()

    def _update_display(self, use_hq: bool = False):
        """Update cached pixmaps based on zoom and pan."""
        if not self._before_pil or not self._after_pil:
            return

        from PIL import Image as PILImage
        from PIL.ImageQt import ImageQt

        # Use after image dimensions (larger/upscaled)
        # Note: _before_pil is already pre-scaled to match _after_pil size in set_pil_images()
        img_w, img_h = self._after_pil.width, self._after_pil.height
        view_w, view_h = self.width() - 20, self.height() - 20
        if view_w < 100:
            view_w = 800
        if view_h < 100:
            view_h = 600

        # Calculate fit scale
        self._fit_scale = min(view_w / img_w, view_h / img_h, 1.0)

        if self._is_fit_mode:
            self._zoom_scale = self._fit_scale

        scale = self._zoom_scale

        # Resampling strategy:
        # - BOX is mathematically optimal for downscaling (averages all contributing pixels)
        # - LANCZOS is best for upscaling or small downscales (sharper)
        # - For interactive speed, use BOX (fast) then switch to LANCZOS after idle
        if use_hq:
            self._use_hq = True
        else:
            self._use_hq = False
            # Schedule high-quality re-render after 2 seconds of inactivity
            self._hq_render_timer.start()

        if self._is_fit_mode or scale <= self._fit_scale:
            # Fit mode - show entire image (significant downscale)
            display_w = int(img_w * scale)
            display_h = int(img_h * scale)

            if use_hq:
                # HQ: Use thumbnail() which is optimized for downscaling
                # It uses reducing_gap=3.0 + LANCZOS internally
                before_scaled = self._before_pil.copy()
                before_scaled.thumbnail((display_w, display_h), PILImage.Resampling.LANCZOS)
                after_scaled = self._after_pil.copy()
                after_scaled.thumbnail((display_w, display_h), PILImage.Resampling.LANCZOS)
            else:
                # Fast: BOX filter is fast and alias-free for downscaling
                before_scaled = self._before_pil.resize((display_w, display_h), PILImage.Resampling.BOX)
                after_scaled = self._after_pil.resize((display_w, display_h), PILImage.Resampling.BOX)
        else:
            # Zoomed mode - show cropped region
            view_img_w = view_w / scale
            view_img_h = view_h / scale

            center_x = self._pan_x * img_w
            center_y = self._pan_y * img_h

            left = center_x - view_img_w / 2
            top = center_y - view_img_h / 2

            # Clamp to bounds
            left = max(0, min(left, img_w - view_img_w))
            top = max(0, min(top, img_h - view_img_h))
            right = min(img_w, left + view_img_w)
            bottom = min(img_h, top + view_img_h)

            # Update pan
            if view_img_w < img_w:
                self._pan_x = (left + view_img_w / 2) / img_w
            if view_img_h < img_h:
                self._pan_y = (top + view_img_h / 2) / img_h

            # Crop region - _before_pil is already same size as _after_pil
            before_crop = self._before_pil.crop((int(left), int(top), int(right), int(bottom)))
            after_crop = self._after_pil.crop((int(left), int(top), int(right), int(bottom)))

            display_w = int(before_crop.width * scale)
            display_h = int(before_crop.height * scale)

            # For zoomed crops, scale is usually >= 1.0 (upscaling or slight downscale)
            if scale >= 1.0:
                # Upscaling - LANCZOS for HQ, BILINEAR for speed
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BILINEAR
            else:
                # Downscaling - BOX for alias-free, LANCZOS for HQ sharpening
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BOX

            before_scaled = before_crop.resize((display_w, display_h), resample)
            after_scaled = after_crop.resize((display_w, display_h), resample)

        # Convert to QPixmap
        qimg_before = ImageQt(before_scaled)
        qimg_after = ImageQt(after_scaled)
        self._before_pixmap = QPixmap.fromImage(qimg_before)
        self._after_pixmap = QPixmap.fromImage(qimg_after)
        self._display_w = display_w
        self._display_h = display_h

        self.update()

    def _update_display_hq(self):
        """Re-render with high-quality LANCZOS resampling after idle timeout."""
        if not self._use_hq:  # Only re-render if not already HQ
            self._update_display(use_hq=True)

    def paintEvent(self, event):
        """Paint the split comparison view."""
        from PySide6.QtGui import QPainter, QPen, QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._before_pixmap or not self._after_pixmap:
            painter.drawText(self.rect(), Qt.AlignCenter, "Load images to compare")
            return

        img_w = self._display_w
        img_h = self._display_h
        x_offset = (self.width() - img_w) // 2
        y_offset = (self.height() - img_h) // 2

        split_x = int(img_w * self._split_position)

        # Draw before image (left side)
        painter.drawPixmap(x_offset, y_offset, self._before_pixmap, 0, 0, split_x, img_h)

        # Draw after image (right side)
        painter.drawPixmap(x_offset + split_x, y_offset, self._after_pixmap, split_x, 0, img_w - split_x, img_h)

        # Draw split line
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x_offset + split_x, y_offset, x_offset + split_x, y_offset + img_h)

        # Draw handle
        handle_y = y_offset + img_h // 2
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(x_offset + split_x - 8, handle_y - 8, 16, 16)

        # Draw labels
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x_offset + 5, y_offset + 20, "Before")
        painter.drawText(x_offset + img_w - 45, y_offset + 20, "After")

        # Draw zoom indicator
        zoom_pct = int(self._zoom_scale * 100)
        painter.drawText(x_offset + 5, y_offset + img_h - 10, f"{zoom_pct}%")

    def wheelEvent(self, event):
        """Handle scroll for zooming."""
        if not self._after_pil:
            return

        delta = event.angleDelta().y()

        if delta > 0:
            self._zoom_scale = min(self._zoom_scale * 1.25, 8.0)
        else:
            self._zoom_scale = max(self._zoom_scale / 1.25, 0.1)

        self._is_fit_mode = False

        # Use relative comparison (within 5% of fit scale) instead of absolute
        # This handles small fit_scale values correctly (e.g., 0.12 for 4x upscaled images)
        if self._fit_scale > 0 and abs(self._zoom_scale / self._fit_scale - 1.0) < 0.05:
            self._is_fit_mode = True
            self._zoom_scale = self._fit_scale

        self._update_display()
        event.accept()

    def mousePressEvent(self, event):
        """Handle mouse press."""
        # Grab focus for wheel events
        self.setFocus()

        if not self._after_pixmap:
            return

        img_w = self._display_w
        x_offset = (self.width() - img_w) // 2

        if event.button() == Qt.LeftButton:
            # Left click - drag split slider
            if x_offset <= event.pos().x() <= x_offset + img_w:
                self._dragging_split = True
                self._update_split_from_mouse(event.pos().x())
        elif event.button() == Qt.RightButton:
            # Right click - pan
            if not self._is_fit_mode:
                self._dragging_pan = True
                self._drag_start_x = event.pos().x()
                self._drag_start_y = event.pos().y()
                self._drag_start_pan_x = self._pan_x
                self._drag_start_pan_y = self._pan_y
                self.setCursor(Qt.ClosedHandCursor)

    def enterEvent(self, event):
        """Grab focus when mouse enters for wheel events."""
        self.setFocus()
        super().enterEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self._dragging_split and self._after_pixmap:
            self._update_split_from_mouse(event.pos().x())
        elif self._dragging_pan and self._after_pil:
            dx = event.pos().x() - self._drag_start_x
            dy = event.pos().y() - self._drag_start_y

            img_w, img_h = self._after_pil.width, self._after_pil.height
            pan_dx = -dx / (img_w * self._zoom_scale)
            pan_dy = -dy / (img_h * self._zoom_scale)

            self._pan_x = max(0, min(1, self._drag_start_pan_x + pan_dx))
            self._pan_y = max(0, min(1, self._drag_start_pan_y + pan_dy))
            self._update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self._dragging_split = False
        elif event.button() == Qt.RightButton:
            self._dragging_pan = False
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset view."""
        if event.button() == Qt.LeftButton:
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5
            self._update_display()

    def _update_split_from_mouse(self, mouse_x: int):
        """Update split position based on mouse x coordinate."""
        if not self._after_pixmap:
            return

        img_w = self._display_w
        x_offset = (self.width() - img_w) // 2

        rel_x = mouse_x - x_offset
        self._split_position = max(0.0, min(1.0, rel_x / img_w))
        self.update()

    def resizeEvent(self, event):
        """Handle resize - update display."""
        super().resizeEvent(event)
        if self._after_pil:
            self._update_display()


class ComparisonDialog(QDialog):
    """
    Dialog for comparing before/after images with split view.
    Supports in-memory upscaling for quick comparison without saving to disk.
    """

    def __init__(self, before_path: str = "", after_path: str = "", parent=None, config=None, onnx_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Before/After Comparison")
        self.setMinimumSize(900, 700)

        # Enable minimize, maximize, and resize
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        self.before_path = before_path
        self.after_path = after_path
        self.config = config
        self.onnx_path = onnx_path
        self._upscale_worker = None
        self._before_pil = None  # Cache for in-memory comparison

        self._setup_ui()

        # If we have a before image and config, enable upscale button
        if before_path and config and onnx_path:
            self._load_before_only()
        elif before_path and after_path:
            self._load_images()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        self.info_label = QLabel("Scroll to zoom, Right-drag to pan, Left-drag to move slider")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.info_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Split compare widget
        self.compare_widget = SplitCompareWidget()
        layout.addWidget(self.compare_widget, 1)

        # Path selection row (wrap in widget to hide in fullscreen)
        self.path_widget = QWidget()
        path_layout = QHBoxLayout(self.path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)

        path_layout.addWidget(QLabel("Before:"))
        self.before_edit = QLineEdit(self.before_path)
        self.before_edit.setReadOnly(True)
        path_layout.addWidget(self.before_edit, 1)
        btn_before = QPushButton("Browse")
        btn_before.clicked.connect(self._browse_before)
        path_layout.addWidget(btn_before)

        path_layout.addWidget(QLabel("After:"))
        self.after_edit = QLineEdit(self.after_path if self.after_path else "(click Upscale)")
        self.after_edit.setReadOnly(True)
        path_layout.addWidget(self.after_edit, 1)
        btn_after = QPushButton("Browse")
        btn_after.clicked.connect(self._browse_after)
        path_layout.addWidget(btn_after)

        layout.addWidget(self.path_widget)

        # Buttons (wrap in widget to hide in fullscreen)
        self.btn_widget = QWidget()
        btn_layout = QHBoxLayout(self.btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        # Upscale button (only if we have config)
        self.upscale_btn = QPushButton("Upscale to Memory")
        self.upscale_btn.setToolTip("Upscale the 'before' image to RAM for quick comparison (no file saved)")
        self.upscale_btn.clicked.connect(self._upscale_to_memory)
        self.upscale_btn.setEnabled(bool(self.config and self.onnx_path and self.before_path))
        btn_layout.addWidget(self.upscale_btn)

        # Fullscreen button
        self.fullscreen_btn = QPushButton("Fullscreen (F11)")
        self.fullscreen_btn.setToolTip("Toggle fullscreen mode")
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        btn_layout.addWidget(self.fullscreen_btn)

        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addWidget(self.btn_widget)

    def _browse_before(self):
        """Browse for before image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Before Image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)"
        )
        if path:
            self.before_path = path
            self.before_edit.setText(path)
            self._load_before_only()
            self.upscale_btn.setEnabled(bool(self.config and self.onnx_path))

    def _browse_after(self):
        """Browse for after image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select After Image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)"
        )
        if path:
            self.after_path = path
            self.after_edit.setText(path)
            self._load_images()

    def _load_before_only(self):
        """Load just the before image (for preview before upscaling)."""
        from PIL import Image as PILImage
        import os

        if self.before_path and os.path.exists(self.before_path):
            try:
                self._before_pil = PILImage.open(self.before_path)
                if self._before_pil.mode not in ("RGB", "RGBA"):
                    self._before_pil = self._before_pil.convert("RGB")
                self._before_pil = self._before_pil.copy()

                # Show before image on both sides initially
                self.compare_widget.set_pil_images(self._before_pil, self._before_pil)
                self.info_label.setText("Click 'Upscale to Memory' to generate comparison")
            except Exception as e:
                print(f"Failed to load before image: {e}")

    def _load_images(self):
        """Load images into comparison widget from file paths."""
        if self.before_path and self.after_path:
            import os
            if os.path.exists(self.before_path) and os.path.exists(self.after_path):
                self.compare_widget.set_images(self.before_path, self.after_path)
                self.info_label.setText("Scroll to zoom, Right-drag to pan, Left-drag to move slider")
                QTimer.singleShot(100, self.compare_widget.setFocus)

    def _upscale_to_memory(self):
        """Upscale the before image to memory for comparison."""
        if not self.before_path or not self.config or not self.onnx_path:
            return

        import os
        if not os.path.exists(self.before_path):
            QMessageBox.warning(self, "Error", "Before image not found.")
            return

        if not os.path.exists(self.onnx_path):
            QMessageBox.warning(self, "Error", "ONNX model not found.")
            return

        # Disable button during processing
        self.upscale_btn.setEnabled(False)
        self.upscale_btn.setText("Upscaling...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Create and start worker
        from PySide6.QtCore import QThread, Signal
        import numpy as np

        class UpscaleToMemoryWorker(QThread):
            progress = Signal(int, int)
            finished = Signal(bool, str, object)  # success, message, PIL image or None

            def __init__(self, image_path, config, onnx_path):
                super().__init__()
                self.image_path = image_path
                self.config = config
                self.onnx_path = onnx_path

            def run(self):
                upscaler = None
                try:
                    import numpy as np
                    import gc
                    from PIL import Image as PILImage
                    from tensorrt_upscaler.upscaler import ImageUpscaler
                    from tensorrt_upscaler.fast_io import load_image_fast
                    from tensorrt_upscaler.resize import resize_array, compute_scaled_size
                    from tensorrt_upscaler.sharpening import cas_sharpen_array, adaptive_sharpen_array

                    cfg = self.config

                    # Load image
                    img, has_alpha = load_image_fast(self.image_path)
                    height, width = img.shape[:2]

                    # Pre-scale (before upscaling)
                    if cfg.prescale_enabled:
                        new_size = compute_scaled_size(
                            width, height,
                            cfg.prescale_mode,
                            cfg.prescale_width,
                            cfg.prescale_height,
                            scale_factor=cfg.prescale_scale_factor,
                        )
                        img = resize_array(img, new_size, cfg.prescale_kernel, has_alpha)
                        height, width = img.shape[:2]

                    # Create upscaler
                    upscaler = ImageUpscaler(
                        onnx_path=self.onnx_path,
                        tile_size=(cfg.tile_width, cfg.tile_height),
                        overlap=cfg.tile_overlap,
                        fp16=cfg.use_fp16,
                        bf16=cfg.use_bf16,
                        tf32=cfg.use_tf32,
                    )

                    # Upscale with progress
                    def on_progress(current, total):
                        self.progress.emit(current, total)

                    result = upscaler.upscale_array(img, has_alpha, on_progress)
                    height, width = result.shape[:2]

                    # Release VRAM immediately after upscale (result is in system RAM)
                    del upscaler
                    upscaler = None
                    gc.collect()

                    # Custom resolution (after upscaling)
                    if cfg.custom_res_enabled:
                        new_size = compute_scaled_size(
                            width, height,
                            cfg.custom_res_mode,
                            cfg.custom_res_width,
                            cfg.custom_res_height,
                            keep_aspect=cfg.custom_res_keep_aspect,
                            scale_factor=cfg.custom_res_scale_factor,
                        )
                        result = resize_array(result, new_size, cfg.custom_res_kernel, has_alpha)

                    # Sharpening (after custom resolution)
                    if cfg.sharpen_enabled and cfg.sharpen_value > 0:
                        sharpen_method = getattr(cfg, 'sharpen_method', 'cas')
                        if sharpen_method == 'adaptive':
                            anime_mode = getattr(cfg, 'sharpen_anime_mode', False)
                            result = adaptive_sharpen_array(
                                result, cfg.sharpen_value, has_alpha,
                                overshoot_ctrl=False, anime_mode=anime_mode
                            )
                        else:  # cas or legacy
                            result = cas_sharpen_array(result, cfg.sharpen_value, has_alpha)

                    # Convert to PIL
                    result_uint8 = (result * 255.0).clip(0, 255).astype(np.uint8)
                    if has_alpha and result.shape[2] == 4:
                        pil_result = PILImage.fromarray(result_uint8, mode="RGBA")
                    else:
                        pil_result = PILImage.fromarray(result_uint8, mode="RGB")

                    self.finished.emit(True, "Upscaling complete", pil_result)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.finished.emit(False, str(e), None)
                finally:
                    # Ensure VRAM is released even on error
                    if upscaler is not None:
                        del upscaler
                        import gc
                        gc.collect()

        self._upscale_worker = UpscaleToMemoryWorker(self.before_path, self.config, self.onnx_path)
        self._upscale_worker.progress.connect(self._on_upscale_progress)
        self._upscale_worker.finished.connect(self._on_upscale_finished)
        self._upscale_worker.start()

    def _on_upscale_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            pct = int(current / total * 1000)
            self.progress_bar.setValue(pct)

    def _on_upscale_finished(self, success: bool, message: str, pil_result):
        """Handle upscale completion."""
        self.upscale_btn.setEnabled(True)
        self.upscale_btn.setText("Upscale to Memory")
        self.progress_bar.setVisible(False)

        if success and pil_result and self._before_pil:
            self.compare_widget.set_pil_images(self._before_pil, pil_result)
            self.after_edit.setText("(in memory)")
            self.info_label.setText("Scroll to zoom, Right-drag to pan, Left-drag to move slider")
            # Ensure compare widget has focus for zoom/pan to work (delay to ensure UI is ready)
            QTimer.singleShot(100, self.compare_widget.setFocus)
        else:
            QMessageBox.warning(self, "Upscale Failed", f"Failed to upscale: {message}")

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode with hidden toolbar."""
        if self.isFullScreen():
            # Exit fullscreen - show toolbar
            self.showNormal()
            self.fullscreen_btn.setText("Fullscreen (F11)")
            self.info_label.setVisible(True)
            self.path_widget.setVisible(True)
            self.btn_widget.setVisible(True)
        else:
            # Enter fullscreen - hide toolbar for true fullscreen
            self.info_label.setVisible(False)
            self.path_widget.setVisible(False)
            self.btn_widget.setVisible(False)
            self.showFullScreen()
            self.fullscreen_btn.setText("Exit Fullscreen (F11)")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
        elif event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                # Exit fullscreen - use _toggle_fullscreen to restore UI
                self._toggle_fullscreen()
            else:
                self.accept()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """Forward wheel events to compare widget for zooming."""
        if hasattr(self, 'compare_widget'):
            self.compare_widget.wheelEvent(event)
        else:
            super().wheelEvent(event)
