"""
Animated output format settings dialog.
Settings for GIF, WebP, AVIF, and APNG encoding.
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
    QSlider,
    QLabel,
    QPushButton,
)

from ..config import get_config, save_config


class AnimatedOutputDialog(QDialog):
    """
    Dialog for animated output format settings.
    Features #36-46
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Animated Output Settings")
        self.setMinimumWidth(400)

        self.config = get_config()
        self._setup_ui()
        self._load_from_config()
        self._on_format_changed()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Format selection
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["GIF", "WebP", "AVIF", "APNG"])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_row.addWidget(self.format_combo)
        format_row.addStretch()
        layout.addLayout(format_row)

        # GIF settings (gifski)
        self.gif_group = QGroupBox("GIF Settings (gifski)")
        gif_layout = QVBoxLayout(self.gif_group)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.gif_quality_slider = QSlider(Qt.Horizontal)
        self.gif_quality_slider.setRange(1, 100)
        self.gif_quality_slider.setValue(90)
        self.gif_quality_slider.valueChanged.connect(
            lambda v: self.gif_quality_spin.setValue(v)
        )
        quality_row.addWidget(self.gif_quality_slider)
        self.gif_quality_spin = QSpinBox()
        self.gif_quality_spin.setRange(1, 100)
        self.gif_quality_spin.setValue(90)
        self.gif_quality_spin.valueChanged.connect(
            lambda v: self.gif_quality_slider.setValue(v)
        )
        quality_row.addWidget(self.gif_quality_spin)
        gif_layout.addLayout(quality_row)

        self.gif_fast = QCheckBox("Fast mode (50% faster, 10% worse quality)")
        gif_layout.addWidget(self.gif_fast)

        layout.addWidget(self.gif_group)

        # WebP settings
        self.webp_group = QGroupBox("WebP Settings (FFmpeg)")
        webp_layout = QVBoxLayout(self.webp_group)

        self.webp_lossless = QCheckBox("Lossless")
        self.webp_lossless.stateChanged.connect(self._on_webp_lossless_changed)
        webp_layout.addWidget(self.webp_lossless)

        quality_row2 = QHBoxLayout()
        quality_row2.addWidget(QLabel("Quality:"))
        self.webp_quality_slider = QSlider(Qt.Horizontal)
        self.webp_quality_slider.setRange(0, 100)
        self.webp_quality_slider.setValue(90)
        self.webp_quality_slider.valueChanged.connect(
            lambda v: self.webp_quality_spin.setValue(v)
        )
        quality_row2.addWidget(self.webp_quality_slider)
        self.webp_quality_spin = QSpinBox()
        self.webp_quality_spin.setRange(0, 100)
        self.webp_quality_spin.setValue(90)
        self.webp_quality_spin.valueChanged.connect(
            lambda v: self.webp_quality_slider.setValue(v)
        )
        quality_row2.addWidget(self.webp_quality_spin)
        webp_layout.addLayout(quality_row2)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.webp_preset = QComboBox()
        self.webp_preset.addItems(["none", "default", "picture", "photo", "drawing", "icon", "text"])
        preset_row.addWidget(self.webp_preset)
        preset_row.addStretch()
        webp_layout.addLayout(preset_row)

        layout.addWidget(self.webp_group)

        # AVIF settings
        self.avif_group = QGroupBox("AVIF Settings (avifenc)")
        avif_layout = QVBoxLayout(self.avif_group)

        self.avif_lossless = QCheckBox("Lossless")
        self.avif_lossless.stateChanged.connect(self._on_avif_lossless_changed)
        avif_layout.addWidget(self.avif_lossless)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color Quality:"))
        self.avif_color_slider = QSlider(Qt.Horizontal)
        self.avif_color_slider.setRange(0, 100)
        self.avif_color_slider.setValue(80)
        self.avif_color_slider.valueChanged.connect(
            lambda v: self.avif_color_spin.setValue(v)
        )
        color_row.addWidget(self.avif_color_slider)
        self.avif_color_spin = QSpinBox()
        self.avif_color_spin.setRange(0, 100)
        self.avif_color_spin.setValue(80)
        self.avif_color_spin.valueChanged.connect(
            lambda v: self.avif_color_slider.setValue(v)
        )
        color_row.addWidget(self.avif_color_spin)
        avif_layout.addLayout(color_row)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Alpha Quality:"))
        self.avif_alpha_slider = QSlider(Qt.Horizontal)
        self.avif_alpha_slider.setRange(0, 100)
        self.avif_alpha_slider.setValue(90)
        self.avif_alpha_slider.valueChanged.connect(
            lambda v: self.avif_alpha_spin.setValue(v)
        )
        alpha_row.addWidget(self.avif_alpha_slider)
        self.avif_alpha_spin = QSpinBox()
        self.avif_alpha_spin.setRange(0, 100)
        self.avif_alpha_spin.setValue(90)
        self.avif_alpha_spin.valueChanged.connect(
            lambda v: self.avif_alpha_slider.setValue(v)
        )
        alpha_row.addWidget(self.avif_alpha_spin)
        avif_layout.addLayout(alpha_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed (0=slow/best, 10=fast):"))
        self.avif_speed_slider = QSlider(Qt.Horizontal)
        self.avif_speed_slider.setRange(0, 10)
        self.avif_speed_slider.setValue(6)
        self.avif_speed_slider.valueChanged.connect(
            lambda v: self.avif_speed_spin.setValue(v)
        )
        speed_row.addWidget(self.avif_speed_slider)
        self.avif_speed_spin = QSpinBox()
        self.avif_speed_spin.setRange(0, 10)
        self.avif_speed_spin.setValue(6)
        self.avif_speed_spin.valueChanged.connect(
            lambda v: self.avif_speed_slider.setValue(v)
        )
        speed_row.addWidget(self.avif_speed_spin)
        avif_layout.addLayout(speed_row)

        layout.addWidget(self.avif_group)

        # APNG settings
        self.apng_group = QGroupBox("APNG Settings (FFmpeg)")
        apng_layout = QVBoxLayout(self.apng_group)

        pred_row = QHBoxLayout()
        pred_row.addWidget(QLabel("Prediction:"))
        self.apng_prediction = QComboBox()
        self.apng_prediction.addItems(["none", "sub", "up", "avg", "paeth", "mixed"])
        self.apng_prediction.setCurrentText("mixed")
        pred_row.addWidget(self.apng_prediction)
        pred_row.addStretch()
        apng_layout.addLayout(pred_row)

        layout.addWidget(self.apng_group)

        # Buttons
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._save_and_close)
        btn_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _on_format_changed(self):
        fmt = self.format_combo.currentText().lower()
        self.gif_group.setVisible(fmt == "gif")
        self.webp_group.setVisible(fmt == "webp")
        self.avif_group.setVisible(fmt == "avif")
        self.apng_group.setVisible(fmt == "apng")
        self.adjustSize()

    def _on_webp_lossless_changed(self):
        lossless = self.webp_lossless.isChecked()
        self.webp_quality_slider.setEnabled(not lossless)
        self.webp_quality_spin.setEnabled(not lossless)

    def _on_avif_lossless_changed(self):
        lossless = self.avif_lossless.isChecked()
        self.avif_color_slider.setEnabled(not lossless)
        self.avif_color_spin.setEnabled(not lossless)
        self.avif_alpha_slider.setEnabled(not lossless)
        self.avif_alpha_spin.setEnabled(not lossless)

    def _load_from_config(self):
        cfg = self.config

        formats = ["gif", "webp", "avif", "apng"]
        self.format_combo.setCurrentIndex(formats.index(cfg.animated_format))

        self.gif_quality_spin.setValue(cfg.gif_quality)
        self.gif_fast.setChecked(cfg.gif_fast_mode)

        self.webp_lossless.setChecked(cfg.webp_lossless)
        self.webp_quality_spin.setValue(cfg.webp_quality)
        presets = ["none", "default", "picture", "photo", "drawing", "icon", "text"]
        self.webp_preset.setCurrentIndex(presets.index(cfg.webp_preset))

        self.avif_lossless.setChecked(cfg.avif_lossless)
        self.avif_color_spin.setValue(cfg.avif_color_quality)
        self.avif_alpha_spin.setValue(cfg.avif_alpha_quality)
        self.avif_speed_spin.setValue(cfg.avif_speed)

        preds = ["none", "sub", "up", "avg", "paeth", "mixed"]
        self.apng_prediction.setCurrentIndex(preds.index(cfg.apng_prediction))

    def _reset_defaults(self):
        self.format_combo.setCurrentIndex(0)
        self.gif_quality_spin.setValue(90)
        self.gif_fast.setChecked(False)
        self.webp_lossless.setChecked(False)
        self.webp_quality_spin.setValue(90)
        self.webp_preset.setCurrentIndex(0)
        self.avif_lossless.setChecked(False)
        self.avif_color_spin.setValue(80)
        self.avif_alpha_spin.setValue(90)
        self.avif_speed_spin.setValue(6)
        self.apng_prediction.setCurrentText("mixed")

    def _save_and_close(self):
        cfg = self.config
        formats = ["gif", "webp", "avif", "apng"]
        presets = ["none", "default", "picture", "photo", "drawing", "icon", "text"]
        preds = ["none", "sub", "up", "avg", "paeth", "mixed"]

        cfg.animated_format = formats[self.format_combo.currentIndex()]
        cfg.gif_quality = self.gif_quality_spin.value()
        cfg.gif_fast_mode = self.gif_fast.isChecked()
        cfg.webp_lossless = self.webp_lossless.isChecked()
        cfg.webp_quality = self.webp_quality_spin.value()
        cfg.webp_preset = presets[self.webp_preset.currentIndex()]
        cfg.avif_lossless = self.avif_lossless.isChecked()
        cfg.avif_color_quality = self.avif_color_spin.value()
        cfg.avif_alpha_quality = self.avif_alpha_spin.value()
        cfg.avif_speed = self.avif_speed_spin.value()
        cfg.apng_prediction = preds[self.apng_prediction.currentIndex()]

        self.accept()
