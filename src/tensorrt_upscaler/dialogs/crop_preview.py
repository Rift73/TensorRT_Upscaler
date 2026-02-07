"""
Crop preview dialog.
Select a crop region and preview the upscaled result.
Includes CropSelectionWidget and ZoomableImageWidget helper widgets.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QPushButton,
    QWidget,
    QMessageBox,
    QSplitter,
)


class CropSelectionWidget(QWidget):
    """
    Widget for selecting a crop region on an image.
    Displays a scaled image with a draggable selection rectangle.
    """

    # Signal emitted when selection changes: (x, y, width, height) in original image coords
    selection_changed = None  # Will be set up in __init__

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import Signal

        self._pil_image = None
        self._display_pixmap: QPixmap = None
        self._scale_factor: float = 1.0
        self._image_offset = (0, 0)  # Where image starts in widget

        # Selection rectangle in original image coordinates
        self._selection_rect = [0, 0, 256, 256]  # x, y, w, h
        self._dragging = False
        self._resizing = False
        self._drag_start = None
        self._resize_corner = None

        # Zoom and pan state
        self._zoom_scale = 1.0  # Additional zoom (1.0 = fit)
        self._fit_scale = 1.0  # Scale needed to fit
        self._is_fit_mode = True
        self._pan_x = 0.5  # Pan position (0-1)
        self._pan_y = 0.5
        self._panning = False  # Right-click panning
        self._pan_start = None
        self._pan_start_x = 0.0
        self._pan_start_y = 0.0
        self._visible_left = 0  # Top-left of visible region in image coords
        self._visible_top = 0

        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable focus for wheel events

        # High-quality render timer (2 second delay after interaction stops)
        self._hq_render_timer = QTimer(self)
        self._hq_render_timer.setSingleShot(True)
        self._hq_render_timer.setInterval(2000)  # 2 seconds
        self._hq_render_timer.timeout.connect(self._update_display_hq)
        self._use_hq = False  # Flag for current render quality

    def load_image(self, path: str):
        """Load image for crop selection."""
        from PIL import Image as PILImage
        from PIL.ImageQt import ImageQt

        try:
            self._pil_image = PILImage.open(path)
            if self._pil_image.mode not in ("RGB", "RGBA"):
                self._pil_image = self._pil_image.convert("RGB")

            # Reset zoom/pan state
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5

            # Scale to fit widget
            self._update_display()

            # Initialize selection to center
            img_w, img_h = self._pil_image.size
            crop_size = min(256, img_w, img_h)
            self._selection_rect = [
                (img_w - crop_size) // 2,
                (img_h - crop_size) // 2,
                crop_size,
                crop_size
            ]
            self.update()
        except Exception as e:
            print(f"Failed to load image for crop: {e}")

    def _update_display(self, use_hq: bool = False):
        """Update the display pixmap based on current widget size and zoom/pan."""
        if not self._pil_image:
            return

        from PIL.ImageQt import ImageQt
        from PIL import Image as PILImage

        img_w, img_h = self._pil_image.size
        view_w = self.width() - 20
        view_h = self.height() - 20
        if view_w < 100:
            view_w = 600
        if view_h < 100:
            view_h = 400

        # Calculate fit scale
        self._fit_scale = min(view_w / img_w, view_h / img_h, 1.0)

        if self._is_fit_mode:
            self._zoom_scale = self._fit_scale

        scale = self._zoom_scale
        self._scale_factor = scale  # For selection coordinate conversion

        # Resampling strategy
        if use_hq:
            self._use_hq = True
        else:
            self._use_hq = False
            self._hq_render_timer.start()

        if self._is_fit_mode or scale <= self._fit_scale:
            # Fit mode - show entire image
            display_w = int(img_w * scale)
            display_h = int(img_h * scale)
            self._image_offset = ((self.width() - display_w) // 2, (self.height() - display_h) // 2)

            # Track visible region (full image in fit mode)
            self._visible_left = 0
            self._visible_top = 0

            if use_hq:
                scaled = self._pil_image.copy()
                scaled.thumbnail((display_w, display_h), PILImage.Resampling.LANCZOS)
            else:
                scaled = self._pil_image.resize((display_w, display_h), PILImage.Resampling.BOX)
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

            # Track visible region for coordinate conversion
            self._visible_left = int(left)
            self._visible_top = int(top)

            # Crop and scale
            cropped = self._pil_image.crop((int(left), int(top), int(right), int(bottom)))
            display_w = int(cropped.width * scale)
            display_h = int(cropped.height * scale)
            self._image_offset = ((self.width() - display_w) // 2, (self.height() - display_h) // 2)

            if scale >= 1.0:
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BILINEAR
            else:
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BOX

            scaled = cropped.resize((display_w, display_h), resample)

        qimg = ImageQt(scaled)
        self._display_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _update_display_hq(self):
        """Re-render with high-quality LANCZOS resampling after idle timeout."""
        if not self._use_hq:  # Only re-render if not already HQ
            self._update_display(use_hq=True)

    def resizeEvent(self, event):
        """Handle resize to update display."""
        super().resizeEvent(event)
        self._update_display()

    def _img_to_display(self, img_x, img_y):
        """Convert image coordinates to display coordinates."""
        disp_x = int((img_x - self._visible_left) * self._scale_factor) + self._image_offset[0]
        disp_y = int((img_y - self._visible_top) * self._scale_factor) + self._image_offset[1]
        return disp_x, disp_y

    def _display_to_img(self, disp_x, disp_y):
        """Convert display coordinates to image coordinates."""
        img_x = (disp_x - self._image_offset[0]) / self._scale_factor + self._visible_left
        img_y = (disp_y - self._image_offset[1]) / self._scale_factor + self._visible_top
        return img_x, img_y

    def paintEvent(self, event):
        """Paint the image with selection overlay."""
        from PySide6.QtGui import QPainter, QPen, QColor, QBrush

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._display_pixmap:
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        # Draw image
        painter.drawPixmap(self._image_offset[0], self._image_offset[1], self._display_pixmap)

        # Convert selection to display coordinates (accounting for zoom/pan)
        sel_x, sel_y = self._img_to_display(self._selection_rect[0], self._selection_rect[1])
        sel_w = int(self._selection_rect[2] * self._scale_factor)
        sel_h = int(self._selection_rect[3] * self._scale_factor)

        # Draw darkened overlay outside selection
        overlay_color = QColor(0, 0, 0, 128)
        painter.fillRect(
            self._image_offset[0], self._image_offset[1],
            self._display_pixmap.width(), sel_y - self._image_offset[1],
            overlay_color
        )
        painter.fillRect(
            self._image_offset[0], sel_y + sel_h,
            self._display_pixmap.width(),
            self._image_offset[1] + self._display_pixmap.height() - sel_y - sel_h,
            overlay_color
        )
        painter.fillRect(
            self._image_offset[0], sel_y,
            sel_x - self._image_offset[0], sel_h,
            overlay_color
        )
        painter.fillRect(
            sel_x + sel_w, sel_y,
            self._image_offset[0] + self._display_pixmap.width() - sel_x - sel_w, sel_h,
            overlay_color
        )

        # Draw selection rectangle
        pen = QPen(QColor(255, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(sel_x, sel_y, sel_w, sel_h)

        # Draw corner handles
        handle_size = 8
        painter.setBrush(QColor(255, 255, 0))
        corners = [
            (sel_x, sel_y),
            (sel_x + sel_w, sel_y),
            (sel_x, sel_y + sel_h),
            (sel_x + sel_w, sel_y + sel_h)
        ]
        for cx, cy in corners:
            painter.drawRect(cx - handle_size//2, cy - handle_size//2, handle_size, handle_size)

        # Draw size label
        painter.setPen(QColor(255, 255, 255))
        label = f"{self._selection_rect[2]}x{self._selection_rect[3]}"
        painter.drawText(sel_x + 5, sel_y + sel_h - 5, label)

        # Draw zoom indicator
        zoom_pct = int(self._zoom_scale * 100)
        painter.drawText(self._image_offset[0] + 5, self._image_offset[1] + self._display_pixmap.height() - 10, f"{zoom_pct}%")

    def mousePressEvent(self, event):
        """Handle mouse press to start drag, resize, or pan."""
        self.setFocus()  # Grab focus for wheel events

        if not self._pil_image:
            return

        pos = event.pos()

        # Right-click - start panning (when zoomed)
        if event.button() == Qt.RightButton:
            if not self._is_fit_mode:
                self._panning = True
                self._pan_start = pos
                self._pan_start_x = self._pan_x
                self._pan_start_y = self._pan_y
                self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        # Check if clicking on a resize handle
        corner = self._get_corner_at(pos)
        if corner is not None:
            self._resizing = True
            self._resize_corner = corner
            self._drag_start = pos
            return

        # Check if clicking inside selection (to drag)
        sel_x, sel_y = self._img_to_display(self._selection_rect[0], self._selection_rect[1])
        sel_w = int(self._selection_rect[2] * self._scale_factor)
        sel_h = int(self._selection_rect[3] * self._scale_factor)

        if sel_x <= pos.x() <= sel_x + sel_w and sel_y <= pos.y() <= sel_y + sel_h:
            self._dragging = True
            self._drag_start = pos
            return

        # Click outside - move selection center to click point
        self._move_selection_to(pos)

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging/resizing/panning."""
        if not self._pil_image:
            return

        pos = event.pos()

        # Handle panning
        if self._panning and self._pan_start:
            img_w, img_h = self._pil_image.size
            dx = pos.x() - self._pan_start.x()
            dy = pos.y() - self._pan_start.y()

            pan_dx = -dx / (img_w * self._zoom_scale)
            pan_dy = -dy / (img_h * self._zoom_scale)

            self._pan_x = max(0, min(1, self._pan_start_x + pan_dx))
            self._pan_y = max(0, min(1, self._pan_start_y + pan_dy))
            self._update_display()
            return

        if self._dragging and self._drag_start:
            # Move selection
            dx = int((pos.x() - self._drag_start.x()) / self._scale_factor)
            dy = int((pos.y() - self._drag_start.y()) / self._scale_factor)

            new_x = self._selection_rect[0] + dx
            new_y = self._selection_rect[1] + dy

            # Clamp to image bounds
            img_w, img_h = self._pil_image.size
            new_x = max(0, min(new_x, img_w - self._selection_rect[2]))
            new_y = max(0, min(new_y, img_h - self._selection_rect[3]))

            self._selection_rect[0] = new_x
            self._selection_rect[1] = new_y
            self._drag_start = pos
            self.update()

        elif self._resizing and self._drag_start:
            # Resize selection
            dx = int((pos.x() - self._drag_start.x()) / self._scale_factor)
            dy = int((pos.y() - self._drag_start.y()) / self._scale_factor)

            x, y, w, h = self._selection_rect
            img_w, img_h = self._pil_image.size

            if self._resize_corner == 0:  # Top-left
                new_x = max(0, x + dx)
                new_y = max(0, y + dy)
                new_w = w - (new_x - x)
                new_h = h - (new_y - y)
                if new_w >= 64 and new_h >= 64:
                    self._selection_rect = [new_x, new_y, new_w, new_h]
            elif self._resize_corner == 1:  # Top-right
                new_y = max(0, y + dy)
                new_w = max(64, w + dx)
                new_h = h - (new_y - y)
                if new_h >= 64 and x + new_w <= img_w:
                    self._selection_rect = [x, new_y, new_w, new_h]
            elif self._resize_corner == 2:  # Bottom-left
                new_x = max(0, x + dx)
                new_w = w - (new_x - x)
                new_h = max(64, h + dy)
                if new_w >= 64 and y + new_h <= img_h:
                    self._selection_rect = [new_x, y, new_w, new_h]
            elif self._resize_corner == 3:  # Bottom-right
                new_w = max(64, w + dx)
                new_h = max(64, h + dy)
                if x + new_w <= img_w and y + new_h <= img_h:
                    self._selection_rect = [x, y, new_w, new_h]

            self._drag_start = pos
            self.update()

        else:
            # Update cursor based on position
            corner = self._get_corner_at(pos)
            if corner is not None:
                if corner in (0, 3):
                    self.setCursor(Qt.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.SizeBDiagCursor)
            else:
                sel_x, sel_y = self._img_to_display(self._selection_rect[0], self._selection_rect[1])
                sel_w = int(self._selection_rect[2] * self._scale_factor)
                sel_h = int(self._selection_rect[3] * self._scale_factor)
                if sel_x <= pos.x() <= sel_x + sel_w and sel_y <= pos.y() <= sel_y + sel_h:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.RightButton:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CrossCursor)
        elif event.button() == Qt.LeftButton:
            self._dragging = False
            self._resizing = False
            self._drag_start = None
            self._resize_corner = None

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom."""
        if event.button() == Qt.LeftButton:
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5
            self._update_display()

    def wheelEvent(self, event):
        """Handle scroll for zooming."""
        if not self._pil_image:
            return

        delta = event.angleDelta().y()

        if delta > 0:
            self._zoom_scale = min(self._zoom_scale * 1.25, 8.0)
        else:
            self._zoom_scale = max(self._zoom_scale / 1.25, 0.1)

        self._is_fit_mode = False

        if abs(self._zoom_scale - self._fit_scale) < 0.05:
            self._is_fit_mode = True
            self._zoom_scale = self._fit_scale

        self._update_display()
        event.accept()

    def enterEvent(self, event):
        """Grab focus when mouse enters for wheel events."""
        self.setFocus()
        super().enterEvent(event)

    def _get_corner_at(self, pos):
        """Return corner index (0-3) if pos is near a corner handle, else None."""
        handle_size = 12
        sel_x, sel_y = self._img_to_display(self._selection_rect[0], self._selection_rect[1])
        sel_w = int(self._selection_rect[2] * self._scale_factor)
        sel_h = int(self._selection_rect[3] * self._scale_factor)

        corners = [
            (sel_x, sel_y),
            (sel_x + sel_w, sel_y),
            (sel_x, sel_y + sel_h),
            (sel_x + sel_w, sel_y + sel_h)
        ]

        for i, (cx, cy) in enumerate(corners):
            if abs(pos.x() - cx) <= handle_size and abs(pos.y() - cy) <= handle_size:
                return i
        return None

    def _move_selection_to(self, pos):
        """Move selection center to the given display position."""
        if not self._pil_image:
            return

        # Convert display coords to image coords (accounting for zoom/pan)
        img_x, img_y = self._display_to_img(pos.x(), pos.y())
        img_x = int(img_x)
        img_y = int(img_y)

        # Center selection at this point
        w, h = self._selection_rect[2], self._selection_rect[3]
        new_x = img_x - w // 2
        new_y = img_y - h // 2

        # Clamp to image bounds
        img_w, img_h = self._pil_image.size
        new_x = max(0, min(new_x, img_w - w))
        new_y = max(0, min(new_y, img_h - h))

        self._selection_rect[0] = new_x
        self._selection_rect[1] = new_y
        self.update()

    def get_selection(self):
        """Return selection rectangle as (x, y, width, height) in original image coords."""
        return tuple(self._selection_rect)

    def set_selection_size(self, width: int, height: int):
        """Set selection size, keeping it centered."""
        if not self._pil_image:
            return

        img_w, img_h = self._pil_image.size
        old_cx = self._selection_rect[0] + self._selection_rect[2] // 2
        old_cy = self._selection_rect[1] + self._selection_rect[3] // 2

        # Clamp size to image
        width = min(width, img_w)
        height = min(height, img_h)

        new_x = old_cx - width // 2
        new_y = old_cy - height // 2

        # Clamp position
        new_x = max(0, min(new_x, img_w - width))
        new_y = max(0, min(new_y, img_h - height))

        self._selection_rect = [new_x, new_y, width, height]
        self.update()

    def get_cropped_image(self):
        """Return the cropped PIL image based on current selection."""
        if not self._pil_image:
            return None

        x, y, w, h = self._selection_rect
        return self._pil_image.crop((x, y, x + w, y + h))


class ZoomableImageWidget(QWidget):
    """
    Widget for displaying a single image with zoom and pan support.

    Features:
    - Scroll to zoom in/out
    - Right-click drag to pan when zoomed
    - Double-click to reset to fit view
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pil_image = None
        self._display_pixmap: QPixmap = None

        # Zoom and pan state
        self._zoom_scale = 1.0  # Display scale (1.0 = fit)
        self._fit_scale = 1.0  # Scale needed to fit
        self._is_fit_mode = True
        self._pan_x = 0.5  # Pan position (0-1)
        self._pan_y = 0.5
        self._dragging_pan = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_pan_x = 0.0
        self._drag_start_pan_y = 0.0
        self._display_w = 0
        self._display_h = 0

        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        self.setFocusPolicy(Qt.StrongFocus)

        # High-quality render timer
        self._hq_render_timer = QTimer(self)
        self._hq_render_timer.setSingleShot(True)
        self._hq_render_timer.setInterval(2000)
        self._hq_render_timer.timeout.connect(self._update_display_hq)
        self._use_hq = False

        # Placeholder text when no image
        self._placeholder_text = "No image"

    def set_placeholder_text(self, text: str):
        """Set placeholder text shown when no image is loaded."""
        self._placeholder_text = text
        self.update()

    def set_pil_image(self, pil_image):
        """Set image from PIL Image object."""
        from PIL import Image as PILImage

        if pil_image is None:
            self._pil_image = None
            self._display_pixmap = None
            self.update()
            return

        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGB")

        self._pil_image = pil_image.copy()

        # Reset view
        self._is_fit_mode = True
        self._pan_x = 0.5
        self._pan_y = 0.5
        self._update_display()

    def clear(self):
        """Clear the image."""
        self._pil_image = None
        self._display_pixmap = None
        self.update()

    def _update_display(self, use_hq: bool = False):
        """Update cached pixmap based on zoom and pan."""
        if not self._pil_image:
            return

        from PIL import Image as PILImage
        from PIL.ImageQt import ImageQt

        img_w, img_h = self._pil_image.width, self._pil_image.height
        view_w, view_h = self.width() - 10, self.height() - 10
        if view_w < 50:
            view_w = 300
        if view_h < 50:
            view_h = 300

        # Calculate fit scale
        self._fit_scale = min(view_w / img_w, view_h / img_h, 1.0)

        if self._is_fit_mode:
            self._zoom_scale = self._fit_scale

        scale = self._zoom_scale

        if use_hq:
            self._use_hq = True
        else:
            self._use_hq = False
            self._hq_render_timer.start()

        if self._is_fit_mode or scale <= self._fit_scale:
            # Fit mode - show entire image
            display_w = int(img_w * scale)
            display_h = int(img_h * scale)

            if use_hq:
                scaled = self._pil_image.copy()
                scaled.thumbnail((display_w, display_h), PILImage.Resampling.LANCZOS)
            else:
                scaled = self._pil_image.resize((display_w, display_h), PILImage.Resampling.BOX)
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

            # Crop region
            crop = self._pil_image.crop((int(left), int(top), int(right), int(bottom)))

            display_w = int(crop.width * scale)
            display_h = int(crop.height * scale)

            if scale >= 1.0:
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BILINEAR
            else:
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BOX

            scaled = crop.resize((display_w, display_h), resample)

        # Convert to QPixmap
        qimg = ImageQt(scaled)
        self._display_pixmap = QPixmap.fromImage(qimg)
        self._display_w = display_w
        self._display_h = display_h

        self.update()

    def _update_display_hq(self):
        """Re-render with high-quality resampling after idle."""
        if not self._use_hq:
            self._update_display(use_hq=True)

    def paintEvent(self, event):
        """Paint the image."""
        from PySide6.QtGui import QPainter, QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._display_pixmap:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder_text)
            return

        img_w = self._display_w
        img_h = self._display_h
        x_offset = (self.width() - img_w) // 2
        y_offset = (self.height() - img_h) // 2

        painter.drawPixmap(x_offset, y_offset, self._display_pixmap)

        # Draw zoom indicator
        zoom_pct = int(self._zoom_scale * 100)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x_offset + 5, y_offset + img_h - 5, f"{zoom_pct}%")

    def wheelEvent(self, event):
        """Handle scroll for zooming."""
        if not self._pil_image:
            return

        delta = event.angleDelta().y()

        if delta > 0:
            self._zoom_scale = min(self._zoom_scale * 1.25, 8.0)
        else:
            self._zoom_scale = max(self._zoom_scale / 1.25, 0.1)

        self._is_fit_mode = False

        if abs(self._zoom_scale - self._fit_scale) < 0.05:
            self._is_fit_mode = True
            self._zoom_scale = self._fit_scale

        self._update_display()
        event.accept()

    def mousePressEvent(self, event):
        """Handle mouse press."""
        self.setFocus()

        if not self._pil_image:
            return

        if event.button() == Qt.RightButton:
            if not self._is_fit_mode:
                self._dragging_pan = True
                self._drag_start_x = event.pos().x()
                self._drag_start_y = event.pos().y()
                self._drag_start_pan_x = self._pan_x
                self._drag_start_pan_y = self._pan_y
                self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse move for panning."""
        if self._dragging_pan and self._pil_image:
            dx = event.pos().x() - self._drag_start_x
            dy = event.pos().y() - self._drag_start_y

            img_w, img_h = self._pil_image.width, self._pil_image.height
            pan_dx = -dx / (img_w * self._zoom_scale)
            pan_dy = -dy / (img_h * self._zoom_scale)

            self._pan_x = max(0, min(1, self._drag_start_pan_x + pan_dx))
            self._pan_y = max(0, min(1, self._drag_start_pan_y + pan_dy))
            self._update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.RightButton:
            self._dragging_pan = False
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset view."""
        if event.button() == Qt.LeftButton:
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5
            self._update_display()

    def enterEvent(self, event):
        """Grab focus when mouse enters."""
        self.setFocus()
        super().enterEvent(event)

    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        if self._pil_image:
            self._update_display()


class CropPreviewDialog(QDialog):
    """
    Dialog for selecting a crop region and previewing the upscaled result.
    """

    def __init__(self, image_path: str, onnx_path: str, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Crop Region")
        self.setMinimumSize(1000, 700)

        # Enable minimize, maximize, and resize
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        self.image_path = image_path
        self.onnx_path = onnx_path
        self.config = config
        self._preview_pixmap = None

        self._setup_ui()
        if image_path:
            self.crop_widget.load_image(image_path)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info
        info = QLabel(
            "Select a region to preview. Drag corners to resize, drag inside to move. "
            "Scroll to zoom, Right-drag to pan, Double-click to reset."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info)

        # Main content: crop selector on left, preview on right
        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)

        # Left: crop selection
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.crop_widget = CropSelectionWidget()
        left_layout.addWidget(self.crop_widget, 1)

        # Size controls
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Crop size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128x128", "256x256", "384x384", "512x512", "Custom"])
        self.size_combo.setCurrentText("256x256")
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        size_row.addWidget(self.size_combo)
        size_row.addStretch()

        self.btn_preview = QPushButton("Preview Upscale")
        self.btn_preview.clicked.connect(self._run_preview)
        size_row.addWidget(self.btn_preview)
        left_layout.addLayout(size_row)

        splitter.addWidget(left_widget)

        # Right: preview result (zoomable/pannable)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(QLabel("Upscaled Preview (scroll to zoom, right-drag to pan):"))
        self.preview_widget = ZoomableImageWidget()
        self.preview_widget.set_placeholder_text("Click 'Preview Upscale' to see result")
        self.preview_widget.setMinimumSize(300, 300)
        self.preview_widget.setStyleSheet("background-color: #222; border: 1px solid #444;")
        right_layout.addWidget(self.preview_widget, 1)

        self.preview_info = QLabel("")
        self.preview_info.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.preview_info)

        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter, 1)

        # Buttons (wrap in widget to hide in fullscreen)
        self.btn_widget = QWidget()
        btn_layout = QHBoxLayout(self.btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)

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

        # Store info label reference for fullscreen toggle
        self.info_label = info

    def _on_size_changed(self, text: str):
        """Handle crop size combo change."""
        if text == "Custom":
            return
        try:
            w, h = text.split("x")
            self.crop_widget.set_selection_size(int(w), int(h))
        except ValueError:
            pass

    def _run_preview(self):
        """Run upscale on the selected crop region."""
        cropped = self.crop_widget.get_cropped_image()
        if cropped is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        if not self.onnx_path:
            QMessageBox.warning(self, "No Model", "No ONNX model specified.")
            return

        import os
        if not os.path.exists(self.onnx_path):
            QMessageBox.warning(self, "Model Not Found", f"Model not found: {self.onnx_path}")
            return

        self.btn_preview.setEnabled(False)
        self.preview_widget.set_placeholder_text("Processing...")
        self.preview_widget.clear()
        self.preview_info.setText("")

        # Process in this thread (crop is small, should be fast)
        upscaler = None
        try:
            import numpy as np
            from PIL import Image as PILImage

            # Import upscaler
            from ..upscaler import ImageUpscaler

            # Create upscaler (will be released after use to free VRAM)
            upscaler = ImageUpscaler(
                onnx_path=self.onnx_path,
                tile_size=(self.config.tile_width, self.config.tile_height),
                overlap=self.config.tile_overlap,
                fp16=self.config.use_fp16,
                bf16=self.config.use_bf16,
                tf32=self.config.use_tf32,
            )

            # Convert to numpy array in [0, 1] float range
            has_alpha = cropped.mode == "RGBA"
            if has_alpha:
                arr = np.array(cropped).astype(np.float32) / 255.0
            else:
                arr = np.array(cropped.convert("RGB")).astype(np.float32) / 255.0

            # Upscale
            import time
            start = time.perf_counter()
            result = upscaler.upscale_array(arr, has_alpha=has_alpha)
            elapsed = time.perf_counter() - start

            # Release VRAM immediately after upscale (result is in system RAM)
            del upscaler
            upscaler = None
            import gc
            gc.collect()

            # Convert result to PIL (result is in [0, 1] range)
            result_uint8 = (result * 255.0).clip(0, 255).astype(np.uint8)
            if has_alpha and result.shape[2] == 4:
                result_pil = PILImage.fromarray(result_uint8, mode="RGBA")
            else:
                result_pil = PILImage.fromarray(result_uint8, mode="RGB")

            # Display in zoomable widget (full resolution, widget handles zoom/pan)
            self.preview_widget.set_pil_image(result_pil)

            # Show info
            orig_w, orig_h = cropped.size
            self.preview_info.setText(
                f"Input: {orig_w}x{orig_h} -> Output: {result.shape[1]}x{result.shape[0]} | "
                f"Time: {elapsed:.2f}s"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.preview_widget.set_placeholder_text(f"Error: {e}")
            self.preview_widget.clear()
            self.preview_info.setText("")

        finally:
            # Ensure VRAM is released even on error
            if upscaler is not None:
                del upscaler
                import gc
                gc.collect()
            self.btn_preview.setEnabled(True)

    def closeEvent(self, event):
        """Clean up on close."""
        super().closeEvent(event)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode with hidden toolbar."""
        if self.isFullScreen():
            # Exit fullscreen - show toolbar
            self.showNormal()
            self.fullscreen_btn.setText("Fullscreen (F11)")
            self.info_label.setVisible(True)
            self.btn_widget.setVisible(True)
        else:
            # Enter fullscreen - hide toolbar for true fullscreen
            self.info_label.setVisible(False)
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
