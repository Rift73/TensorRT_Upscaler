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
    QDoubleSpinBox,
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
        self.custom_mode.addItems(["Custom width", "Custom height", "Scale factor"])
        self.custom_mode.currentIndexChanged.connect(self._on_custom_mode_changed)
        mode_row.addWidget(self.custom_mode)
        mode_row.addStretch()
        custom_layout.addLayout(mode_row)

        size_row = QHBoxLayout()
        self.custom_width_label = QLabel("Width:")
        size_row.addWidget(self.custom_width_label)
        self.custom_width = QSpinBox()
        self.custom_width.setRange(1, 16384)
        self.custom_width.setValue(1920)
        size_row.addWidget(self.custom_width)
        self.custom_height_label = QLabel("Height:")
        size_row.addWidget(self.custom_height_label)
        self.custom_height = QSpinBox()
        self.custom_height.setRange(1, 16384)
        self.custom_height.setValue(1080)
        size_row.addWidget(self.custom_height)
        custom_layout.addLayout(size_row)

        scale_row = QHBoxLayout()
        self.custom_scale_label = QLabel("Scale factor:")
        scale_row.addWidget(self.custom_scale_label)
        self.custom_scale_factor = QDoubleSpinBox()
        self.custom_scale_factor.setRange(1.0, 16.0)
        self.custom_scale_factor.setValue(2.0)
        self.custom_scale_factor.setSingleStep(0.5)
        self.custom_scale_factor.setDecimals(1)
        scale_row.addWidget(self.custom_scale_factor)
        scale_row.addStretch()
        custom_layout.addLayout(scale_row)

        kernel_row = QHBoxLayout()
        kernel_row.addWidget(QLabel("Kernel:"))
        self.custom_kernel = QComboBox()
        self.custom_kernel.addItems(["Lanczos", "Hermite", "Catmull-Rom"])
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
        self.secondary_mode.addItems(["Custom width", "Custom height", "Scale factor"])
        self.secondary_mode.currentIndexChanged.connect(self._on_secondary_mode_changed)
        mode_row2.addWidget(self.secondary_mode)
        mode_row2.addStretch()
        secondary_layout.addLayout(mode_row2)

        size_row2 = QHBoxLayout()
        self.secondary_width_label = QLabel("Width:")
        size_row2.addWidget(self.secondary_width_label)
        self.secondary_width = QSpinBox()
        self.secondary_width.setRange(1, 16384)
        self.secondary_width.setValue(1920)
        size_row2.addWidget(self.secondary_width)
        self.secondary_height_label = QLabel("Height:")
        size_row2.addWidget(self.secondary_height_label)
        self.secondary_height = QSpinBox()
        self.secondary_height.setRange(1, 16384)
        self.secondary_height.setValue(1080)
        size_row2.addWidget(self.secondary_height)
        secondary_layout.addLayout(size_row2)

        scale_row2 = QHBoxLayout()
        self.secondary_scale_label = QLabel("Scale factor:")
        scale_row2.addWidget(self.secondary_scale_label)
        self.secondary_scale_factor = QDoubleSpinBox()
        self.secondary_scale_factor.setRange(1.0, 16.0)
        self.secondary_scale_factor.setValue(2.0)
        self.secondary_scale_factor.setSingleStep(0.5)
        self.secondary_scale_factor.setDecimals(1)
        scale_row2.addWidget(self.secondary_scale_factor)
        scale_row2.addStretch()
        secondary_layout.addLayout(scale_row2)

        kernel_row2 = QHBoxLayout()
        kernel_row2.addWidget(QLabel("Kernel:"))
        self.secondary_kernel = QComboBox()
        self.secondary_kernel.addItems(["Lanczos", "Hermite", "Catmull-Rom"])
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
        self.prescale_mode.addItems(["Custom width", "Custom height", "Scale factor"])
        self.prescale_mode.currentIndexChanged.connect(self._on_prescale_mode_changed)
        mode_row3.addWidget(self.prescale_mode)
        mode_row3.addStretch()
        prescale_layout.addLayout(mode_row3)

        size_row3 = QHBoxLayout()
        self.prescale_width_label = QLabel("Width:")
        size_row3.addWidget(self.prescale_width_label)
        self.prescale_width = QSpinBox()
        self.prescale_width.setRange(1, 16384)
        self.prescale_width.setValue(1920)
        size_row3.addWidget(self.prescale_width)
        self.prescale_height_label = QLabel("Height:")
        size_row3.addWidget(self.prescale_height_label)
        self.prescale_height = QSpinBox()
        self.prescale_height.setRange(1, 16384)
        self.prescale_height.setValue(1080)
        size_row3.addWidget(self.prescale_height)
        prescale_layout.addLayout(size_row3)

        scale_row3 = QHBoxLayout()
        self.prescale_scale_label = QLabel("Scale factor:")
        scale_row3.addWidget(self.prescale_scale_label)
        self.prescale_scale_factor = QDoubleSpinBox()
        self.prescale_scale_factor.setRange(1.0, 16.0)
        self.prescale_scale_factor.setValue(2.0)
        self.prescale_scale_factor.setSingleStep(0.5)
        self.prescale_scale_factor.setDecimals(1)
        scale_row3.addWidget(self.prescale_scale_factor)
        scale_row3.addStretch()
        prescale_layout.addLayout(scale_row3)

        kernel_row3 = QHBoxLayout()
        kernel_row3.addWidget(QLabel("Kernel:"))
        self.prescale_kernel = QComboBox()
        self.prescale_kernel.addItems(["Lanczos", "Hermite", "Catmull-Rom"])
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
        kernels = ["lanczos", "hermite", "catmull-rom"]
        modes = ["width", "height", "scale_factor"]

        # Custom resolution
        self.custom_enabled.setChecked(cfg.custom_res_enabled)
        self.custom_aspect.setChecked(cfg.custom_res_keep_aspect)
        # Handle legacy "2x" mode by converting to "scale_factor"
        custom_mode = cfg.custom_res_mode if cfg.custom_res_mode != "2x" else "scale_factor"
        mode_idx = modes.index(custom_mode) if custom_mode in modes else 0
        self.custom_mode.setCurrentIndex(mode_idx)
        self.custom_width.setValue(cfg.custom_res_width)
        self.custom_height.setValue(cfg.custom_res_height)
        self.custom_scale_factor.setValue(cfg.custom_res_scale_factor)
        kernel_idx = kernels.index(cfg.custom_res_kernel) if cfg.custom_res_kernel in kernels else 0
        self.custom_kernel.setCurrentIndex(kernel_idx)
        self._on_custom_mode_changed(mode_idx)

        # Secondary
        self.secondary_enabled.setChecked(cfg.secondary_enabled)
        secondary_mode = cfg.secondary_mode if cfg.secondary_mode != "2x" else "scale_factor"
        mode_idx = modes.index(secondary_mode) if secondary_mode in modes else 0
        self.secondary_mode.setCurrentIndex(mode_idx)
        self.secondary_width.setValue(cfg.secondary_width)
        self.secondary_height.setValue(cfg.secondary_height)
        self.secondary_scale_factor.setValue(cfg.secondary_scale_factor)
        kernel_idx = kernels.index(cfg.secondary_kernel) if cfg.secondary_kernel in kernels else 0
        self.secondary_kernel.setCurrentIndex(kernel_idx)
        self._on_secondary_mode_changed(mode_idx)

        # Pre-scale
        self.prescale_enabled.setChecked(cfg.prescale_enabled)
        prescale_mode = cfg.prescale_mode if cfg.prescale_mode != "2x" else "scale_factor"
        mode_idx = modes.index(prescale_mode) if prescale_mode in modes else 0
        self.prescale_mode.setCurrentIndex(mode_idx)
        self.prescale_width.setValue(cfg.prescale_width)
        self.prescale_height.setValue(cfg.prescale_height)
        self.prescale_scale_factor.setValue(cfg.prescale_scale_factor)
        kernel_idx = kernels.index(cfg.prescale_kernel) if cfg.prescale_kernel in kernels else 0
        self.prescale_kernel.setCurrentIndex(kernel_idx)
        self._on_prescale_mode_changed(mode_idx)

    def _save_and_close(self):
        cfg = self.config
        modes = ["width", "height", "scale_factor"]
        kernels = ["lanczos", "hermite", "catmull-rom"]

        # Custom resolution
        cfg.custom_res_enabled = self.custom_enabled.isChecked()
        cfg.custom_res_keep_aspect = self.custom_aspect.isChecked()
        cfg.custom_res_mode = modes[self.custom_mode.currentIndex()]
        cfg.custom_res_width = self.custom_width.value()
        cfg.custom_res_height = self.custom_height.value()
        cfg.custom_res_scale_factor = self.custom_scale_factor.value()
        cfg.custom_res_kernel = kernels[self.custom_kernel.currentIndex()]

        # Secondary
        cfg.secondary_enabled = self.secondary_enabled.isChecked()
        cfg.secondary_mode = modes[self.secondary_mode.currentIndex()]
        cfg.secondary_width = self.secondary_width.value()
        cfg.secondary_height = self.secondary_height.value()
        cfg.secondary_scale_factor = self.secondary_scale_factor.value()
        cfg.secondary_kernel = kernels[self.secondary_kernel.currentIndex()]

        # Pre-scale
        cfg.prescale_enabled = self.prescale_enabled.isChecked()
        cfg.prescale_mode = modes[self.prescale_mode.currentIndex()]
        cfg.prescale_width = self.prescale_width.value()
        cfg.prescale_height = self.prescale_height.value()
        cfg.prescale_scale_factor = self.prescale_scale_factor.value()
        cfg.prescale_kernel = kernels[self.prescale_kernel.currentIndex()]

        self.accept()

    def _on_custom_mode_changed(self, index):
        """Show/hide scale factor vs width/height based on mode selection."""
        is_scale_mode = index == 2  # "Scale factor"
        self.custom_width_label.setVisible(not is_scale_mode)
        self.custom_width.setVisible(not is_scale_mode)
        self.custom_height_label.setVisible(not is_scale_mode)
        self.custom_height.setVisible(not is_scale_mode)
        self.custom_scale_label.setVisible(is_scale_mode)
        self.custom_scale_factor.setVisible(is_scale_mode)

    def _on_secondary_mode_changed(self, index):
        """Show/hide scale factor vs width/height based on mode selection."""
        is_scale_mode = index == 2  # "Scale factor"
        self.secondary_width_label.setVisible(not is_scale_mode)
        self.secondary_width.setVisible(not is_scale_mode)
        self.secondary_height_label.setVisible(not is_scale_mode)
        self.secondary_height.setVisible(not is_scale_mode)
        self.secondary_scale_label.setVisible(is_scale_mode)
        self.secondary_scale_factor.setVisible(is_scale_mode)

    def _on_prescale_mode_changed(self, index):
        """Show/hide scale factor vs width/height based on mode selection."""
        is_scale_mode = index == 2  # "Scale factor"
        self.prescale_width_label.setVisible(not is_scale_mode)
        self.prescale_width.setVisible(not is_scale_mode)
        self.prescale_height_label.setVisible(not is_scale_mode)
        self.prescale_height.setVisible(not is_scale_mode)
        self.prescale_scale_label.setVisible(is_scale_mode)
        self.prescale_scale_factor.setVisible(is_scale_mode)
