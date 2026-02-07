"""
GUI components for TensorRT Upscaler.
Separates workers, widgets, and main window for better organization.
"""

from .workers import UpscaleWorker, ClipboardWorker
from .widgets import DropLineEdit, ThumbnailLabel
from .tray_manager import TrayManagerMixin
from .shortcuts import ShortcutsMixin
from .watch_folder import WatchFolderMixin
from .progress_tracker import ProgressTrackerMixin

__all__ = [
    "UpscaleWorker",
    "ClipboardWorker",
    "DropLineEdit",
    "ThumbnailLabel",
    "TrayManagerMixin",
    "ShortcutsMixin",
    "WatchFolderMixin",
    "ProgressTrackerMixin",
]
