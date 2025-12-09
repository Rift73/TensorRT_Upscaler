"""
PyTorch Options Dialog - Configure precision and optimization settings.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QGridLayout,
)
from PySide6.QtCore import Qt


class PyTorchOptionsDialog(QDialog):
    """Dialog for configuring PyTorch precision and optimization settings."""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("PyTorch Options")
        self.setMinimumWidth(350)

        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Precision group
        precision_group = QGroupBox("Precision")
        precision_layout = QGridLayout()
        precision_group.setLayout(precision_layout)

        self._fp16_check = QCheckBox("FP16 (Half Precision)")
        self._fp16_check.setToolTip(
            "Use FP16 half precision for faster inference.\n"
            "May have slight quality loss on some models."
        )

        self._bf16_check = QCheckBox("BF16 (BFloat16)")
        self._bf16_check.setToolTip(
            "Use BF16 precision (Ampere+ GPUs only).\n"
            "Better numerical stability than FP16, similar speed."
        )

        self._tf32_check = QCheckBox("TF32 (TensorFloat32)")
        self._tf32_check.setToolTip(
            "Enable TensorFloat32 for matrix operations.\n"
            "Ampere+ GPUs only. ~2-3x faster matmuls with minimal quality loss."
        )

        # Mutually exclusive: FP16 and BF16
        self._fp16_check.toggled.connect(self._on_fp16_toggled)
        self._bf16_check.toggled.connect(self._on_bf16_toggled)

        precision_layout.addWidget(self._fp16_check, 0, 0)
        precision_layout.addWidget(self._bf16_check, 0, 1)
        precision_layout.addWidget(self._tf32_check, 1, 0)

        layout.addWidget(precision_group)

        # Optimization group
        optim_group = QGroupBox("Optimizations")
        optim_layout = QGridLayout()
        optim_group.setLayout(optim_layout)

        self._channels_last_check = QCheckBox("Channels Last (NHWC)")
        self._channels_last_check.setToolTip(
            "Use NHWC memory format instead of NCHW.\n"
            "Faster for CNN models on modern GPUs with tensor cores."
        )

        optim_layout.addWidget(self._channels_last_check, 0, 0)

        layout.addWidget(optim_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._btn_defaults = QPushButton("Restore Defaults")
        self._btn_defaults.clicked.connect(self._restore_defaults)
        btn_layout.addWidget(self._btn_defaults)

        self._btn_ok = QPushButton("OK")
        self._btn_ok.setDefault(True)
        self._btn_ok.clicked.connect(self._on_ok)
        btn_layout.addWidget(self._btn_ok)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_cancel)

        layout.addLayout(btn_layout)

    def _on_fp16_toggled(self, checked):
        """When FP16 is enabled, disable BF16."""
        if checked:
            self._bf16_check.setChecked(False)

    def _on_bf16_toggled(self, checked):
        """When BF16 is enabled, disable FP16."""
        if checked:
            self._fp16_check.setChecked(False)

    def _load_from_config(self):
        """Load settings from config."""
        if self.config is None:
            return

        self._fp16_check.setChecked(self.config.pytorch_half)
        self._bf16_check.setChecked(self.config.pytorch_bf16)
        self._tf32_check.setChecked(self.config.pytorch_enable_tf32)
        self._channels_last_check.setChecked(self.config.pytorch_channels_last)

    def _save_to_config(self):
        """Save settings to config."""
        if self.config is None:
            return

        self.config.pytorch_half = self._fp16_check.isChecked()
        self.config.pytorch_bf16 = self._bf16_check.isChecked()
        self.config.pytorch_enable_tf32 = self._tf32_check.isChecked()
        self.config.pytorch_channels_last = self._channels_last_check.isChecked()

    def _restore_defaults(self):
        """Restore default settings."""
        self._fp16_check.setChecked(False)
        self._bf16_check.setChecked(True)
        self._tf32_check.setChecked(True)
        self._channels_last_check.setChecked(True)

    def _on_ok(self):
        """Save and close."""
        self._save_to_config()
        self.accept()

    def get_settings(self) -> dict:
        """Get current settings as a dictionary."""
        return {
            "pytorch_half": self._fp16_check.isChecked(),
            "pytorch_bf16": self._bf16_check.isChecked(),
            "pytorch_enable_tf32": self._tf32_check.isChecked(),
            "pytorch_channels_last": self._channels_last_check.isChecked(),
        }
