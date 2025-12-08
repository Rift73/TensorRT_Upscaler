"""
Custom Qt widgets for the TensorRT Upscaler GUI.
Contains reusable widgets like DropLineEdit and ThumbnailLabel.
"""

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import QLineEdit, QLabel


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


class ThumbnailLabel(QLabel):
    """Label that displays image with pan and zoom support.

    - Scroll to zoom in/out
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
        """Handle mouse wheel for zooming."""
        if not self._pil_image:
            return

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
