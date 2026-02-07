"""
Dialog windows for advanced settings.
Each dialog is in its own module for maintainability.
"""

from .resolution import CustomResolutionDialog
from .pytorch_options import PyTorchOptionsDialog
from .tensorrt_options import TensorRTOptionsDialog
from .web_image_dialog import WebImageDialog
from .animated_output import AnimatedOutputDialog
from .png_options import PngOptionsDialog
from .settings import SettingsDialog
from .notifications import NotificationsDialog
from .log import LogDialog
from .model_queue import ModelQueueDialog
from .comparison import ComparisonDialog
from .crop_preview import CropPreviewDialog
from .sharpen import SharpenDialog

__all__ = [
    "CustomResolutionDialog",
    "PyTorchOptionsDialog",
    "TensorRTOptionsDialog",
    "WebImageDialog",
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
