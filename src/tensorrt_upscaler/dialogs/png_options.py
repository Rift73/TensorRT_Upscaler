"""
PNG optimization settings dialog.
Settings for pngquant quantization and pingo optimization.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QSpinBox,
    QSlider,
    QLabel,
    QPushButton,
)

from ..config import get_config, save_config


class PngOptionsDialog(QDialog):
    """
    Dialog for PNG optimization settings.
    Features #47-49
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PNG Options")
        self.setMinimumWidth(350)

        self.config = get_config()
        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info about fpng
        info_label = QLabel("PNG encoding uses fpng (12-19x faster than libpng)")
        info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info_label)

        # Quantize group
        quantize_group = QGroupBox("Quantize (pngquant)")
        quantize_layout = QVBoxLayout(quantize_group)

        self.quantize_enabled = QCheckBox("Enable quantization (lossy)")
        quantize_layout.addWidget(self.quantize_enabled)

        colors_row = QHBoxLayout()
        colors_row.addWidget(QLabel("Colors:"))
        self.colors_slider = QSlider(Qt.Horizontal)
        self.colors_slider.setRange(1, 256)
        self.colors_slider.setValue(256)
        self.colors_slider.valueChanged.connect(
            lambda v: self.colors_spin.setValue(v)
        )
        colors_row.addWidget(self.colors_slider)
        self.colors_spin = QSpinBox()
        self.colors_spin.setRange(1, 256)
        self.colors_spin.setValue(256)
        self.colors_spin.valueChanged.connect(
            lambda v: self.colors_slider.setValue(v)
        )
        colors_row.addWidget(self.colors_spin)
        quantize_layout.addLayout(colors_row)

        layout.addWidget(quantize_group)

        # Optimize group
        optimize_group = QGroupBox("Optimize (pingo)")
        optimize_layout = QVBoxLayout(optimize_group)

        self.optimize_enabled = QCheckBox("Enable lossless optimization")
        optimize_layout.addWidget(self.optimize_enabled)

        layout.addWidget(optimize_group)

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
        self.quantize_enabled.setChecked(cfg.png_quantize_enabled)
        self.colors_spin.setValue(cfg.png_quantize_colors)
        self.optimize_enabled.setChecked(cfg.png_optimize_enabled)

    def _save_and_close(self):
        cfg = self.config
        cfg.png_quantize_enabled = self.quantize_enabled.isChecked()
        cfg.png_quantize_colors = self.colors_spin.value()
        cfg.png_optimize_enabled = self.optimize_enabled.isChecked()
        self.accept()
