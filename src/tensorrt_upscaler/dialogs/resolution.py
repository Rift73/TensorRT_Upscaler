"""
Resolution settings dialog.
Custom resolution, secondary output, and pre-scale settings.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QLabel,
    QPushButton,
)

from ..config import get_config, save_config


class CustomResolutionDialog(QDialog):
    """
    Dialog for custom resolution, secondary output, and pre-scale settings.
    Features #20-35
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resolution Settings")
        self.setMinimumWidth(450)

        self.config = get_config()
        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Custom Resolution Group (#20-25)
        custom_group = QGroupBox("Custom Resolution (after upscaling)")
        custom_layout = QVBoxLayout(custom_group)

        self.custom_enabled = QCheckBox("Enable custom resolution")
        custom_layout.addWidget(self.custom_enabled)

        self.custom_aspect = QCheckBox("Keep aspect ratio")
        custom_layout.addWidget(self.custom_aspect)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.custom_mode = QComboBox()
        self.custom_mode.addItems(["Custom width", "Custom height", "2x downscale"])
        mode_row.addWidget(self.custom_mode)
        mode_row.addStretch()
        custom_layout.addLayout(mode_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Width:"))
        self.custom_width = QSpinBox()
        self.custom_width.setRange(1, 16384)
        self.custom_width.setValue(1920)
        size_row.addWidget(self.custom_width)
        size_row.addWidget(QLabel("Height:"))
        self.custom_height = QSpinBox()
        self.custom_height.setRange(1, 16384)
        self.custom_height.setValue(1080)
        size_row.addWidget(self.custom_height)
        custom_layout.addLayout(size_row)

        kernel_row = QHBoxLayout()
        kernel_row.addWidget(QLabel("Kernel:"))
        self.custom_kernel = QComboBox()
        self.custom_kernel.addItems(["Lanczos", "Hermite"])
        kernel_row.addWidget(self.custom_kernel)
        kernel_row.addStretch()
        custom_layout.addLayout(kernel_row)

        layout.addWidget(custom_group)

        # Secondary Output Group (#26-30)
        secondary_group = QGroupBox("Secondary Output")
        secondary_layout = QVBoxLayout(secondary_group)

        self.secondary_enabled = QCheckBox("Enable secondary output")
        secondary_layout.addWidget(self.secondary_enabled)

        mode_row2 = QHBoxLayout()
        mode_row2.addWidget(QLabel("Mode:"))
        self.secondary_mode = QComboBox()
        self.secondary_mode.addItems(["Custom width", "Custom height", "2x downscale"])
        mode_row2.addWidget(self.secondary_mode)
        mode_row2.addStretch()
        secondary_layout.addLayout(mode_row2)

        size_row2 = QHBoxLayout()
        size_row2.addWidget(QLabel("Width:"))
        self.secondary_width = QSpinBox()
        self.secondary_width.setRange(1, 16384)
        self.secondary_width.setValue(1920)
        size_row2.addWidget(self.secondary_width)
        size_row2.addWidget(QLabel("Height:"))
        self.secondary_height = QSpinBox()
        self.secondary_height.setRange(1, 16384)
        self.secondary_height.setValue(1080)
        size_row2.addWidget(self.secondary_height)
        secondary_layout.addLayout(size_row2)

        kernel_row2 = QHBoxLayout()
        kernel_row2.addWidget(QLabel("Kernel:"))
        self.secondary_kernel = QComboBox()
        self.secondary_kernel.addItems(["Lanczos", "Hermite"])
        kernel_row2.addWidget(self.secondary_kernel)
        kernel_row2.addStretch()
        secondary_layout.addLayout(kernel_row2)

        layout.addWidget(secondary_group)

        # Pre-Scale Group (#31-35)
        prescale_group = QGroupBox("Pre-Scale (before upscaling)")
        prescale_layout = QVBoxLayout(prescale_group)

        self.prescale_enabled = QCheckBox("Enable pre-scale")
        prescale_layout.addWidget(self.prescale_enabled)

        mode_row3 = QHBoxLayout()
        mode_row3.addWidget(QLabel("Mode:"))
        self.prescale_mode = QComboBox()
        self.prescale_mode.addItems(["Custom width", "Custom height", "2x downscale"])
        mode_row3.addWidget(self.prescale_mode)
        mode_row3.addStretch()
        prescale_layout.addLayout(mode_row3)

        size_row3 = QHBoxLayout()
        size_row3.addWidget(QLabel("Width:"))
        self.prescale_width = QSpinBox()
        self.prescale_width.setRange(1, 16384)
        self.prescale_width.setValue(1920)
        size_row3.addWidget(self.prescale_width)
        size_row3.addWidget(QLabel("Height:"))
        self.prescale_height = QSpinBox()
        self.prescale_height.setRange(1, 16384)
        self.prescale_height.setValue(1080)
        size_row3.addWidget(self.prescale_height)
        prescale_layout.addLayout(size_row3)

        kernel_row3 = QHBoxLayout()
        kernel_row3.addWidget(QLabel("Kernel:"))
        self.prescale_kernel = QComboBox()
        self.prescale_kernel.addItems(["Lanczos", "Hermite"])
        kernel_row3.addWidget(self.prescale_kernel)
        kernel_row3.addStretch()
        prescale_layout.addLayout(kernel_row3)

        layout.addWidget(prescale_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._save_and_close)
        btn_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _load_from_config(self):
        cfg = self.config

        # Custom resolution
        self.custom_enabled.setChecked(cfg.custom_res_enabled)
        self.custom_aspect.setChecked(cfg.custom_res_keep_aspect)
        self.custom_mode.setCurrentIndex(["width", "height", "2x"].index(cfg.custom_res_mode))
        self.custom_width.setValue(cfg.custom_res_width)
        self.custom_height.setValue(cfg.custom_res_height)
        self.custom_kernel.setCurrentIndex(0 if cfg.custom_res_kernel == "lanczos" else 1)

        # Secondary
        self.secondary_enabled.setChecked(cfg.secondary_enabled)
        self.secondary_mode.setCurrentIndex(["width", "height", "2x"].index(cfg.secondary_mode))
        self.secondary_width.setValue(cfg.secondary_width)
        self.secondary_height.setValue(cfg.secondary_height)
        self.secondary_kernel.setCurrentIndex(0 if cfg.secondary_kernel == "lanczos" else 1)

        # Pre-scale
        self.prescale_enabled.setChecked(cfg.prescale_enabled)
        self.prescale_mode.setCurrentIndex(["width", "height", "2x"].index(cfg.prescale_mode))
        self.prescale_width.setValue(cfg.prescale_width)
        self.prescale_height.setValue(cfg.prescale_height)
        self.prescale_kernel.setCurrentIndex(0 if cfg.prescale_kernel == "lanczos" else 1)

    def _save_and_close(self):
        cfg = self.config
        modes = ["width", "height", "2x"]
        kernels = ["lanczos", "hermite"]

        # Custom resolution
        cfg.custom_res_enabled = self.custom_enabled.isChecked()
        cfg.custom_res_keep_aspect = self.custom_aspect.isChecked()
        cfg.custom_res_mode = modes[self.custom_mode.currentIndex()]
        cfg.custom_res_width = self.custom_width.value()
        cfg.custom_res_height = self.custom_height.value()
        cfg.custom_res_kernel = kernels[self.custom_kernel.currentIndex()]

        # Secondary
        cfg.secondary_enabled = self.secondary_enabled.isChecked()
        cfg.secondary_mode = modes[self.secondary_mode.currentIndex()]
        cfg.secondary_width = self.secondary_width.value()
        cfg.secondary_height = self.secondary_height.value()
        cfg.secondary_kernel = kernels[self.secondary_kernel.currentIndex()]

        # Pre-scale
        cfg.prescale_enabled = self.prescale_enabled.isChecked()
        cfg.prescale_mode = modes[self.prescale_mode.currentIndex()]
        cfg.prescale_width = self.prescale_width.value()
        cfg.prescale_height = self.prescale_height.value()
        cfg.prescale_kernel = kernels[self.prescale_kernel.currentIndex()]

        self.accept()
