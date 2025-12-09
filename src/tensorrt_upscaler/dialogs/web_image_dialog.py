"""
Dialog for selecting images extracted from a web page.
"""

import os
import tempfile
from typing import List, Optional
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QWidget,
    QGridLayout,
    QCheckBox,
    QFrame,
    QMessageBox,
    QApplication,
)

from ..config import get_config


class ThumbnailWidget(QFrame):
    """Widget showing a thumbnail with checkbox for selection."""

    def __init__(self, image_url: str, width: int, height: int, parent=None):
        super().__init__(parent)
        self.image_url = image_url
        self.img_width = width
        self.img_height = height
        self.local_path: Optional[str] = None

        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Checkbox for selection
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        layout.addWidget(self.checkbox, alignment=Qt.AlignCenter)

        # Thumbnail label
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(150, 150)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet("background-color: #333;")
        self.thumb_label.setText("Loading...")
        layout.addWidget(self.thumb_label)

        # Size label
        self.size_label = QLabel(f"{width}x{height}")
        self.size_label.setAlignment(Qt.AlignCenter)
        self.size_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.size_label)

        # Click on widget toggles checkbox
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.checkbox.setChecked(not self.checkbox.isChecked())
        super().mousePressEvent(event)

    def set_thumbnail(self, pixmap: QPixmap):
        """Set the thumbnail image."""
        scaled = pixmap.scaled(
            150, 150,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.thumb_label.setPixmap(scaled)

    def set_error(self, message: str):
        """Show error message instead of thumbnail."""
        self.thumb_label.setText(message)
        self.thumb_label.setStyleSheet("background-color: #533; color: #faa;")

    def is_selected(self) -> bool:
        return self.checkbox.isChecked()


class ExtractWorker(QThread):
    """Worker thread for extracting images from a webpage."""

    progress = Signal(str)
    image_found = Signal(str, int, int)  # url, width, height
    finished = Signal(bool, str)  # success, message

    def __init__(self, url: str, browser: str, wait_time: float):
        super().__init__()
        self.url = url
        self.browser = browser
        self.wait_time = wait_time
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from ..web_extractor import WebImageExtractor, PLAYWRIGHT_AVAILABLE

            if not PLAYWRIGHT_AVAILABLE:
                self.finished.emit(False, "Playwright not installed. Go to Settings > Web Extract to install.")
                return

            extractor = WebImageExtractor(
                browser_cookie_source=self.browser,
                wait_time=self.wait_time,
                headless=True,
            )

            def on_progress(msg: str):
                if not self._cancelled:
                    self.progress.emit(msg)

            images = extractor.extract_images(self.url, progress_callback=on_progress)

            if self._cancelled:
                self.finished.emit(False, "Cancelled")
                return

            for img in images:
                if self._cancelled:
                    break
                self.image_found.emit(img.url, img.width, img.height)

            self.finished.emit(True, f"Found {len(images)} images")

        except Exception as e:
            self.finished.emit(False, str(e))


class DownloadWorker(QThread):
    """Worker thread for downloading selected images."""

    progress = Signal(int, int)  # current, total
    downloaded = Signal(str, str)  # url, local_path
    finished = Signal(bool, str, list)  # success, message, paths

    def __init__(self, image_urls: List[str]):
        super().__init__()
        self.image_urls = image_urls
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import urllib.request
        import hashlib
        from urllib.parse import urlparse

        downloaded_paths = []
        total = len(self.image_urls)

        for i, url in enumerate(self.image_urls):
            if self._cancelled:
                self.finished.emit(False, "Cancelled", downloaded_paths)
                return

            self.progress.emit(i + 1, total)

            try:
                # Parse URL for filename
                parsed = urlparse(url)
                filename = os.path.basename(parsed.path)

                # Generate filename if empty or invalid
                if not filename or '.' not in filename:
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"image_{url_hash}.jpg"

                # Create temp file
                ext = os.path.splitext(filename)[1] or ".jpg"
                fd, dest_path = tempfile.mkstemp(suffix=ext)
                os.close(fd)

                # Download
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                req = urllib.request.Request(url, headers=headers)

                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(dest_path, "wb") as f:
                        f.write(response.read())

                downloaded_paths.append(dest_path)
                self.downloaded.emit(url, dest_path)

            except Exception as e:
                print(f"Failed to download {url}: {e}")

        self.finished.emit(True, f"Downloaded {len(downloaded_paths)} images", downloaded_paths)


class WebImageDialog(QDialog):
    """
    Dialog for extracting and selecting images from a web page.

    Shows thumbnails of found images with checkboxes for selection.
    """

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.url = url
        self.config = get_config()
        self.selected_paths: List[str] = []

        self._extract_worker: Optional[ExtractWorker] = None
        self._download_worker: Optional[DownloadWorker] = None
        self._thumbnail_widgets: List[ThumbnailWidget] = []

        self.setWindowTitle("Extract Images from Web Page")
        self.setMinimumSize(800, 600)

        self._setup_ui()
        self._start_extraction()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # URL display
        url_label = QLabel(f"<b>URL:</b> {self.url}")
        url_label.setWordWrap(True)
        url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(url_label)

        # Progress section
        self._progress_label = QLabel("Initializing...")
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self._progress_bar)

        # Scroll area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._thumb_container = QWidget()
        self._thumb_layout = QGridLayout(self._thumb_container)
        self._thumb_layout.setSpacing(8)
        scroll.setWidget(self._thumb_container)
        layout.addWidget(scroll, stretch=1)

        # Selection buttons
        sel_layout = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all)
        sel_layout.addWidget(self._select_all_btn)

        self._select_none_btn = QPushButton("Select None")
        self._select_none_btn.clicked.connect(self._select_none)
        sel_layout.addWidget(self._select_none_btn)

        sel_layout.addStretch()

        self._count_label = QLabel("0 images selected")
        sel_layout.addWidget(self._count_label)

        layout.addLayout(sel_layout)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        self._import_btn = QPushButton("Import Selected")
        self._import_btn.clicked.connect(self._on_import)
        self._import_btn.setEnabled(False)
        btn_layout.addWidget(self._import_btn)

        layout.addLayout(btn_layout)

    def _start_extraction(self):
        """Start extracting images from the webpage."""
        self._progress_label.setText("Extracting images from webpage...")
        self._progress_bar.setRange(0, 0)

        self._extract_worker = ExtractWorker(
            url=self.url,
            browser=self.config.web_extract_browser,
            wait_time=self.config.web_extract_wait_time,
        )
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.image_found.connect(self._on_image_found)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.start()

    def _on_extract_progress(self, message: str):
        self._progress_label.setText(message)

    def _on_image_found(self, url: str, width: int, height: int):
        """Add a thumbnail widget for the found image."""
        widget = ThumbnailWidget(url, width, height)
        widget.checkbox.stateChanged.connect(self._update_count)
        self._thumbnail_widgets.append(widget)

        # Add to grid (4 columns)
        row = len(self._thumbnail_widgets) // 4
        col = len(self._thumbnail_widgets) % 4
        self._thumb_layout.addWidget(widget, row, col)

        # Start loading thumbnail
        self._load_thumbnail(widget)

        self._update_count()

    def _load_thumbnail(self, widget: ThumbnailWidget):
        """Load thumbnail for a widget (in background)."""
        import urllib.request
        import hashlib

        try:
            # Download to temp file
            url_hash = hashlib.md5(widget.image_url.encode()).hexdigest()[:8]
            fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = urllib.request.Request(widget.image_url, headers=headers)

            with urllib.request.urlopen(req, timeout=10) as response:
                with open(temp_path, "wb") as f:
                    f.write(response.read())

            widget.local_path = temp_path

            # Load as pixmap
            pixmap = QPixmap(temp_path)
            if not pixmap.isNull():
                widget.set_thumbnail(pixmap)
            else:
                widget.set_error("Invalid image")

        except Exception as e:
            widget.set_error("Load failed")

    def _on_extract_finished(self, success: bool, message: str):
        if success:
            self._progress_label.setText(message)
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(100)
            self._import_btn.setEnabled(len(self._thumbnail_widgets) > 0)
        else:
            self._progress_label.setText(f"Error: {message}")
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)

            if "not installed" in message.lower():
                QMessageBox.warning(self, "Playwright Required", message)

        self._update_count()

    def _select_all(self):
        for widget in self._thumbnail_widgets:
            widget.checkbox.setChecked(True)

    def _select_none(self):
        for widget in self._thumbnail_widgets:
            widget.checkbox.setChecked(False)

    def _update_count(self):
        count = sum(1 for w in self._thumbnail_widgets if w.is_selected())
        self._count_label.setText(f"{count} images selected")
        self._import_btn.setEnabled(count > 0)

    def _on_cancel(self):
        if self._extract_worker and self._extract_worker.isRunning():
            self._extract_worker.cancel()
            self._extract_worker.wait()

        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.cancel()
            self._download_worker.wait()

        self.reject()

    def _on_import(self):
        """Import selected images."""
        # Get selected URLs that have been downloaded
        selected = [w for w in self._thumbnail_widgets if w.is_selected() and w.local_path]

        if not selected:
            # Need to download first
            urls = [w.image_url for w in self._thumbnail_widgets if w.is_selected()]
            if not urls:
                QMessageBox.warning(self, "No Selection", "Please select at least one image.")
                return

            # Start download
            self._progress_label.setText("Downloading selected images...")
            self._progress_bar.setRange(0, len(urls))
            self._progress_bar.setValue(0)
            self._import_btn.setEnabled(False)

            self._download_worker = DownloadWorker(urls)
            self._download_worker.progress.connect(self._on_download_progress)
            self._download_worker.finished.connect(self._on_download_finished)
            self._download_worker.start()
        else:
            # Already downloaded
            self.selected_paths = [w.local_path for w in selected]
            self.accept()

    def _on_download_progress(self, current: int, total: int):
        self._progress_bar.setValue(current)
        self._progress_label.setText(f"Downloading {current}/{total}...")

    def _on_download_finished(self, success: bool, message: str, paths: List[str]):
        if success and paths:
            self.selected_paths = paths
            self.accept()
        else:
            self._progress_label.setText(f"Download failed: {message}")
            self._import_btn.setEnabled(True)

    def get_selected_paths(self) -> List[str]:
        """Get the paths of selected and downloaded images."""
        return self.selected_paths

    def closeEvent(self, event):
        self._on_cancel()
        super().closeEvent(event)
