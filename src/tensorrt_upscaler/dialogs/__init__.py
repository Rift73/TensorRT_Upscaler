"""
Dialog windows for advanced settings.
Re-exports all dialogs from the legacy dialogs module.

For new dialogs, create separate files in this package:
- resolution.py - CustomResolutionDialog (example)
- animated.py - AnimatedOutputDialog
- settings.py - SettingsDialog
- etc.
"""

# Re-export from resolution.py (refactored)
from .resolution import CustomResolutionDialog

# Re-export from pytorch_options.py
from .pytorch_options import PyTorchOptionsDialog

# Re-export remaining dialogs from legacy module
# These can be moved to separate files as needed
from ..dialogs_legacy import (
    AnimatedOutputDialog,
    PngOptionsDialog,
    SettingsDialog,
    NotificationsDialog,
    LogDialog,
    ModelQueueDialog,
    ComparisonDialog,
    CropPreviewDialog,
    SharpenDialog,
)

__all__ = [
    "CustomResolutionDialog",
    "PyTorchOptionsDialog",
    "AnimatedOutputDialog",
    "PngOptionsDialog",
    "SettingsDialog",
    "NotificationsDialog",
    "LogDialog",
    "ModelQueueDialog",
    "ComparisonDialog",
    "CropPreviewDialog",
    "SharpenDialog",
]
