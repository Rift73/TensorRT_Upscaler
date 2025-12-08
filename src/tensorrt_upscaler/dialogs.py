"""
Dialog windows for advanced settings.
Features #20-49 + QoL features
"""

import json
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QLabel,
    QPushButton,
    QWidget,
    QLineEdit,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QTabWidget,
    QFormLayout,
    QProgressBar,
)

from .config import get_config, save_config


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


class SettingsDialog(QDialog):
    """
    Settings dialog for QoL features:
    - Skip existing files
    - Conditional processing (size filters)
    - Aspect ratio filter
    - Presets management
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)

        self.config = get_config()
        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Use tabs for organization
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Processing
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)

        # Skip existing group
        skip_group = QGroupBox("Output Options")
        skip_layout = QVBoxLayout(skip_group)
        self.skip_existing = QCheckBox("Skip files that already have output")
        self.skip_existing.setToolTip("Don't process images if output file already exists")
        skip_layout.addWidget(self.skip_existing)
        self.preserve_metadata = QCheckBox("Preserve image metadata")
        self.preserve_metadata.setToolTip("Preserve ICC color profile and EXIF data from source images")
        skip_layout.addWidget(self.preserve_metadata)
        processing_layout.addWidget(skip_group)

        # Conditional processing group
        cond_group = QGroupBox("Conditional Processing")
        cond_layout = QVBoxLayout(cond_group)

        self.conditional_enabled = QCheckBox("Enable size-based filtering")
        self.conditional_enabled.setToolTip("Only process images within specified dimensions")
        cond_layout.addWidget(self.conditional_enabled)

        size_grid = QHBoxLayout()

        # Min dimensions
        min_box = QGroupBox("Minimum Size")
        min_layout = QFormLayout(min_box)
        self.cond_min_width = QSpinBox()
        self.cond_min_width.setRange(0, 65535)
        self.cond_min_width.setSpecialValueText("Any")
        min_layout.addRow("Width:", self.cond_min_width)
        self.cond_min_height = QSpinBox()
        self.cond_min_height.setRange(0, 65535)
        self.cond_min_height.setSpecialValueText("Any")
        min_layout.addRow("Height:", self.cond_min_height)
        size_grid.addWidget(min_box)

        # Max dimensions
        max_box = QGroupBox("Maximum Size")
        max_layout = QFormLayout(max_box)
        self.cond_max_width = QSpinBox()
        self.cond_max_width.setRange(0, 65535)
        self.cond_max_width.setSpecialValueText("Any")
        max_layout.addRow("Width:", self.cond_max_width)
        self.cond_max_height = QSpinBox()
        self.cond_max_height.setRange(0, 65535)
        self.cond_max_height.setSpecialValueText("Any")
        max_layout.addRow("Height:", self.cond_max_height)
        size_grid.addWidget(max_box)

        cond_layout.addLayout(size_grid)
        processing_layout.addWidget(cond_group)

        # Aspect ratio filter group
        aspect_group = QGroupBox("Aspect Ratio Filter")
        aspect_layout = QVBoxLayout(aspect_group)

        self.aspect_enabled = QCheckBox("Enable aspect ratio filtering")
        aspect_layout.addWidget(self.aspect_enabled)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.aspect_mode = QComboBox()
        self.aspect_mode.addItems(["Any", "Landscape only", "Portrait only", "Square only", "Custom range"])
        self.aspect_mode.currentIndexChanged.connect(self._on_aspect_mode_changed)
        mode_row.addWidget(self.aspect_mode)
        mode_row.addStretch()
        aspect_layout.addLayout(mode_row)

        # Custom range (only shown when "Custom range" selected)
        self.aspect_range_widget = QWidget()
        range_layout = QHBoxLayout(self.aspect_range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(QLabel("Min ratio:"))
        self.aspect_min = QDoubleSpinBox()
        self.aspect_min.setRange(0.0, 10.0)
        self.aspect_min.setDecimals(2)
        self.aspect_min.setSingleStep(0.1)
        range_layout.addWidget(self.aspect_min)
        range_layout.addWidget(QLabel("Max ratio:"))
        self.aspect_max = QDoubleSpinBox()
        self.aspect_max.setRange(0.0, 10.0)
        self.aspect_max.setDecimals(2)
        self.aspect_max.setSingleStep(0.1)
        range_layout.addWidget(self.aspect_max)
        range_layout.addStretch()
        aspect_layout.addWidget(self.aspect_range_widget)
        self.aspect_range_widget.hide()

        processing_layout.addWidget(aspect_group)
        processing_layout.addStretch()

        tabs.addTab(processing_tab, "Processing")

        # Tab 2: Presets
        presets_tab = QWidget()
        presets_layout = QVBoxLayout(presets_tab)

        presets_info = QLabel(
            "Presets save your current settings (tile size, precision, resolution, etc.) "
            "for quick switching between different workflows."
        )
        presets_info.setWordWrap(True)
        presets_info.setStyleSheet("color: #888; font-style: italic;")
        presets_layout.addWidget(presets_info)

        self.presets_list = QListWidget()
        presets_layout.addWidget(self.presets_list)

        preset_btn_row = QHBoxLayout()
        self.btn_save_preset = QPushButton("Save Current as Preset")
        self.btn_save_preset.clicked.connect(self._save_preset)
        preset_btn_row.addWidget(self.btn_save_preset)
        self.btn_load_preset = QPushButton("Load Selected")
        self.btn_load_preset.clicked.connect(self._load_preset)
        preset_btn_row.addWidget(self.btn_load_preset)
        self.btn_delete_preset = QPushButton("Delete")
        self.btn_delete_preset.clicked.connect(self._delete_preset)
        preset_btn_row.addWidget(self.btn_delete_preset)
        presets_layout.addLayout(preset_btn_row)

        tabs.addTab(presets_tab, "Presets")

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

    def _on_aspect_mode_changed(self, index: int):
        """Show/hide custom range based on mode selection."""
        # Index 4 = "Custom range"
        self.aspect_range_widget.setVisible(index == 4)

    def _load_from_config(self):
        cfg = self.config

        # Skip existing / metadata
        self.skip_existing.setChecked(cfg.skip_existing)
        self.preserve_metadata.setChecked(cfg.preserve_metadata)

        # Conditional processing
        self.conditional_enabled.setChecked(cfg.conditional_enabled)
        self.cond_min_width.setValue(cfg.conditional_min_width)
        self.cond_min_height.setValue(cfg.conditional_min_height)
        self.cond_max_width.setValue(cfg.conditional_max_width)
        self.cond_max_height.setValue(cfg.conditional_max_height)

        # Aspect ratio filter
        self.aspect_enabled.setChecked(cfg.aspect_filter_enabled)
        mode_map = {"any": 0, "landscape": 1, "portrait": 2, "square": 3, "custom": 4}
        self.aspect_mode.setCurrentIndex(mode_map.get(cfg.aspect_filter_mode, 0))
        self.aspect_min.setValue(cfg.aspect_filter_min_ratio)
        self.aspect_max.setValue(cfg.aspect_filter_max_ratio)
        self._on_aspect_mode_changed(self.aspect_mode.currentIndex())

        # Presets
        self._refresh_presets_list()

    def _refresh_presets_list(self):
        """Refresh the presets list from config."""
        self.presets_list.clear()
        try:
            presets = json.loads(self.config.presets)
            for name in sorted(presets.keys()):
                self.presets_list.addItem(name)
        except json.JSONDecodeError:
            pass

    def _save_preset(self):
        """Save current settings as a new preset."""
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return

        name = name.strip()

        # Collect current config values to save
        cfg = self.config
        preset_data = {
            "tile_width": cfg.tile_width,
            "tile_height": cfg.tile_height,
            "tile_overlap": cfg.tile_overlap,
            "use_fp16": cfg.use_fp16,
            "use_bf16": cfg.use_bf16,
            "sharpen_enabled": cfg.sharpen_enabled,
            "sharpen_value": cfg.sharpen_value,
            "custom_res_enabled": cfg.custom_res_enabled,
            "custom_res_mode": cfg.custom_res_mode,
            "custom_res_width": cfg.custom_res_width,
            "custom_res_height": cfg.custom_res_height,
            "custom_res_kernel": cfg.custom_res_kernel,
            "prescale_enabled": cfg.prescale_enabled,
            "prescale_mode": cfg.prescale_mode,
            "prescale_width": cfg.prescale_width,
            "prescale_height": cfg.prescale_height,
        }

        try:
            presets = json.loads(cfg.presets)
        except json.JSONDecodeError:
            presets = {}

        presets[name] = preset_data
        cfg.presets = json.dumps(presets)
        save_config()

        self._refresh_presets_list()
        QMessageBox.information(self, "Saved", f"Preset '{name}' saved successfully.")

    def _load_preset(self):
        """Load selected preset."""
        item = self.presets_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Please select a preset to load.")
            return

        name = item.text()
        try:
            presets = json.loads(self.config.presets)
            if name not in presets:
                return

            preset_data = presets[name]
            cfg = self.config

            # Apply preset values
            for key, value in preset_data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

            cfg.last_preset = name
            save_config()

            QMessageBox.information(
                self, "Loaded",
                f"Preset '{name}' loaded.\n\nNote: UI will reflect changes after closing this dialog."
            )
        except json.JSONDecodeError:
            pass

    def _delete_preset(self):
        """Delete selected preset."""
        item = self.presets_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Please select a preset to delete.")
            return

        name = item.text()
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete preset '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            presets = json.loads(self.config.presets)
            if name in presets:
                del presets[name]
                self.config.presets = json.dumps(presets)
                save_config()
                self._refresh_presets_list()
        except json.JSONDecodeError:
            pass

    def _save_and_close(self):
        cfg = self.config

        # Skip existing / metadata
        cfg.skip_existing = self.skip_existing.isChecked()
        cfg.preserve_metadata = self.preserve_metadata.isChecked()

        # Conditional processing
        cfg.conditional_enabled = self.conditional_enabled.isChecked()
        cfg.conditional_min_width = self.cond_min_width.value()
        cfg.conditional_min_height = self.cond_min_height.value()
        cfg.conditional_max_width = self.cond_max_width.value()
        cfg.conditional_max_height = self.cond_max_height.value()

        # Aspect ratio filter
        cfg.aspect_filter_enabled = self.aspect_enabled.isChecked()
        mode_map = {0: "any", 1: "landscape", 2: "portrait", 3: "square", 4: "custom"}
        cfg.aspect_filter_mode = mode_map.get(self.aspect_mode.currentIndex(), "any")
        cfg.aspect_filter_min_ratio = self.aspect_min.value()
        cfg.aspect_filter_max_ratio = self.aspect_max.value()

        save_config()
        self.accept()


class NotificationsDialog(QDialog):
    """
    Notifications and window behavior settings:
    - System tray notifications
    - Sound on completion
    - Always on top
    - Minimize to tray
    - Auto-shutdown options
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Notifications & Behavior")
        self.setMinimumWidth(450)

        self.config = get_config()
        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Notifications group
        notify_group = QGroupBox("Notifications")
        notify_layout = QVBoxLayout(notify_group)

        self.notify_complete = QCheckBox("Show notification when batch completes")
        notify_layout.addWidget(self.notify_complete)

        self.sound_complete = QCheckBox("Play sound when batch completes")
        notify_layout.addWidget(self.sound_complete)

        sound_row = QHBoxLayout()
        sound_row.addWidget(QLabel("Sound file:"))
        self.sound_path = QLineEdit()
        self.sound_path.setPlaceholderText("Leave empty for system default")
        sound_row.addWidget(self.sound_path)
        self.btn_browse_sound = QPushButton("Browse")
        self.btn_browse_sound.clicked.connect(self._browse_sound)
        sound_row.addWidget(self.btn_browse_sound)
        notify_layout.addLayout(sound_row)

        layout.addWidget(notify_group)

        # Window behavior group
        window_group = QGroupBox("Window Behavior")
        window_layout = QVBoxLayout(window_group)

        self.always_on_top = QCheckBox("Always on top")
        self.always_on_top.setToolTip("Keep window above other windows")
        window_layout.addWidget(self.always_on_top)

        self.minimize_to_tray = QCheckBox("Minimize to system tray")
        self.minimize_to_tray.setToolTip("Hide to tray instead of taskbar when minimized")
        window_layout.addWidget(self.minimize_to_tray)

        self.open_output = QCheckBox("Open output folder when complete")
        self.open_output.setToolTip("Automatically open Explorer to output folder")
        window_layout.addWidget(self.open_output)

        layout.addWidget(window_group)

        # Auto-shutdown group
        shutdown_group = QGroupBox("Auto-Shutdown (after batch completes)")
        shutdown_layout = QVBoxLayout(shutdown_group)

        self.auto_shutdown = QCheckBox("Enable auto-shutdown")
        self.auto_shutdown.setToolTip("Automatically sleep/hibernate/shutdown when done")
        shutdown_layout.addWidget(self.auto_shutdown)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Action:"))
        self.shutdown_mode = QComboBox()
        self.shutdown_mode.addItems(["Sleep", "Hibernate", "Shutdown"])
        mode_row.addWidget(self.shutdown_mode)
        mode_row.addStretch()
        shutdown_layout.addLayout(mode_row)

        warning = QLabel("Warning: Ensure you save your work before enabling.")
        warning.setStyleSheet("color: #c44; font-style: italic;")
        shutdown_layout.addWidget(warning)

        layout.addWidget(shutdown_group)

        layout.addStretch()

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

    def _browse_sound(self):
        """Browse for sound file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sound File",
            "",
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*.*)"
        )
        if path:
            self.sound_path.setText(path)

    def _load_from_config(self):
        cfg = self.config

        self.notify_complete.setChecked(cfg.notify_on_complete)
        self.sound_complete.setChecked(cfg.sound_on_complete)
        self.sound_path.setText(cfg.sound_file_path)

        self.always_on_top.setChecked(cfg.always_on_top)
        self.minimize_to_tray.setChecked(cfg.minimize_to_tray)
        self.open_output.setChecked(cfg.open_output_on_complete)

        self.auto_shutdown.setChecked(cfg.auto_shutdown_enabled)
        mode_map = {"sleep": 0, "hibernate": 1, "shutdown": 2}
        self.shutdown_mode.setCurrentIndex(mode_map.get(cfg.auto_shutdown_mode, 0))

    def _save_and_close(self):
        cfg = self.config

        cfg.notify_on_complete = self.notify_complete.isChecked()
        cfg.sound_on_complete = self.sound_complete.isChecked()
        cfg.sound_file_path = self.sound_path.text()

        cfg.always_on_top = self.always_on_top.isChecked()
        cfg.minimize_to_tray = self.minimize_to_tray.isChecked()
        cfg.open_output_on_complete = self.open_output.isChecked()

        cfg.auto_shutdown_enabled = self.auto_shutdown.isChecked()
        mode_map = {0: "sleep", 1: "hibernate", 2: "shutdown"}
        cfg.auto_shutdown_mode = mode_map.get(self.shutdown_mode.currentIndex(), "sleep")

        save_config()
        self.accept()


class LogDialog(QDialog):
    """
    Processing log/history dialog.
    Shows batch processing results and allows exporting.
    """

    def __init__(self, log_entries: list = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Log")
        self.setMinimumSize(600, 400)

        self.log_entries = log_entries or []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Log list
        self.log_list = QListWidget()
        self.log_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.log_list)

        # Populate with entries
        for entry in self.log_entries:
            item = QListWidgetItem(entry)
            self.log_list.addItem(item)

        # Stats
        stats_label = QLabel(f"Total entries: {len(self.log_entries)}")
        layout.addWidget(stats_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.clicked.connect(self._export_csv)
        btn_layout.addWidget(self.btn_export)
        self.btn_clear = QPushButton("Clear Log")
        self.btn_clear.clicked.connect(self._clear_log)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def add_entry(self, entry: str):
        """Add a new log entry."""
        self.log_entries.append(entry)
        self.log_list.addItem(QListWidgetItem(entry))

    def _export_csv(self):
        """Export log to CSV file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log",
            "processing_log.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("Entry\n")
                    for entry in self.log_entries:
                        # Escape quotes and wrap in quotes
                        escaped = entry.replace('"', '""')
                        f.write(f'"{escaped}"\n')
                QMessageBox.information(self, "Exported", f"Log exported to:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export: {e}")

    def _clear_log(self):
        """Clear the log."""
        reply = QMessageBox.question(
            self, "Confirm Clear",
            "Clear all log entries?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.log_entries.clear()
            self.log_list.clear()


class ModelQueueDialog(QDialog):
    """
    Dialog for managing a queue of ONNX models.
    Processes the same files with multiple models sequentially.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Queue")
        self.setMinimumSize(500, 400)

        self.config = get_config()
        self.model_list: list = []
        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "Add multiple ONNX models to process the same images with each model.\n"
            "Output files will have the model name appended to distinguish results."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(info)

        # Model list
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.list_widget)

        # Buttons row
        btn_row = QHBoxLayout()

        self.btn_add = QPushButton("Add Model")
        self.btn_add.clicked.connect(self._add_model)
        btn_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Enable checkbox
        self.enable_check = QCheckBox("Enable model queue (process files with all models)")
        layout.addWidget(self.enable_check)

        # Dialog buttons
        dialog_btn_layout = QHBoxLayout()
        dialog_btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._save_and_close)
        dialog_btn_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        dialog_btn_layout.addWidget(cancel_btn)
        layout.addLayout(dialog_btn_layout)

    def _load_from_config(self):
        """Load model queue from config."""
        try:
            self.model_list = json.loads(self.config.model_queue)
        except json.JSONDecodeError:
            self.model_list = []

        self._refresh_list()
        self.enable_check.setChecked(self.config.model_queue_enabled)

    def _refresh_list(self):
        """Refresh the list widget from model_list."""
        self.list_widget.clear()
        import os
        for path in self.model_list:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.list_widget.addItem(item)

    def _add_model(self):
        """Add a model to the queue."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select ONNX Models",
            "",
            "ONNX Models (*.onnx);;All Files (*.*)"
        )
        for path in paths:
            if path not in self.model_list:
                self.model_list.append(path)
        self._refresh_list()

    def _remove_selected(self):
        """Remove selected models from queue."""
        selected = self.list_widget.selectedItems()
        for item in selected:
            path = item.data(Qt.UserRole)
            if path in self.model_list:
                self.model_list.remove(path)
        self._refresh_list()

    def _clear_all(self):
        """Clear all models from queue."""
        self.model_list.clear()
        self._refresh_list()

    def _save_and_close(self):
        """Save queue to config and close."""
        # Update list from widget (in case of drag-drop reorder)
        self.model_list = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            self.model_list.append(item.data(Qt.UserRole))

        self.config.model_queue = json.dumps(self.model_list)
        self.config.model_queue_enabled = self.enable_check.isChecked()
        save_config()
        self.accept()

    def get_model_queue(self) -> list:
        """Return the current model queue."""
        return self.model_list.copy()

    def is_enabled(self) -> bool:
        """Return whether model queue is enabled."""
        return self.enable_check.isChecked()


class SplitCompareWidget(QWidget):
    """
    Widget for comparing two images with a draggable split slider.
    Shows 'before' on left and 'after' on right of the slider.

    Features:
    - Scroll to zoom in/out
    - Right-click drag to pan when zoomed
    - Left-click drag to move split slider
    - Double-click to reset to fit view
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_pil = None  # Full resolution PIL images
        self._after_pil = None
        self._split_position: float = 0.5  # 0.0 to 1.0
        self._dragging_split: bool = False
        self._dragging_pan: bool = False
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable focus for wheel events

        # Zoom and pan state
        self._zoom_scale = 1.0  # Display scale (1.0 = fit)
        self._fit_scale = 1.0  # Scale needed to fit
        self._is_fit_mode = True
        self._pan_x = 0.5  # Pan position (0-1)
        self._pan_y = 0.5
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_pan_x = 0.0
        self._drag_start_pan_y = 0.0

        # Cached display pixmaps
        self._before_pixmap: QPixmap = None
        self._after_pixmap: QPixmap = None
        self._display_w = 0
        self._display_h = 0

        # High-quality render timer (2 second delay after interaction stops)
        self._hq_render_timer = QTimer(self)
        self._hq_render_timer.setSingleShot(True)
        self._hq_render_timer.setInterval(2000)  # 2 seconds
        self._hq_render_timer.timeout.connect(self._update_display_hq)
        self._use_hq = False  # Flag for current render quality

    def set_images(self, before_path: str, after_path: str):
        """Load before and after images from file paths."""
        from PIL import Image as PILImage

        try:
            # Load before image
            pil_before = PILImage.open(before_path)
            if pil_before.mode not in ("RGB", "RGBA"):
                pil_before = pil_before.convert("RGB")

            # Load after image
            pil_after = PILImage.open(after_path)
            if pil_after.mode not in ("RGB", "RGBA"):
                pil_after = pil_after.convert("RGB")

            self.set_pil_images(pil_before, pil_after)

        except Exception as e:
            print(f"Failed to load comparison images: {e}")

    def set_pil_images(self, before_pil, after_pil):
        """Set before and after images from PIL Image objects (in-memory)."""
        from PIL import Image as PILImage

        # Ensure correct mode
        if before_pil.mode not in ("RGB", "RGBA"):
            before_pil = before_pil.convert("RGB")
        if after_pil.mode not in ("RGB", "RGBA"):
            after_pil = after_pil.convert("RGB")

        # Cache full resolution - scale before to match after size for 1:1 comparison
        self._after_pil = after_pil.copy()
        # Pre-scale before image to after size (do this once, not on every update)
        if before_pil.size != after_pil.size:
            self._before_pil = before_pil.resize(after_pil.size, PILImage.Resampling.LANCZOS)
        else:
            self._before_pil = before_pil.copy()

        # Reset view
        self._is_fit_mode = True
        self._pan_x = 0.5
        self._pan_y = 0.5
        self._update_display()

    def _update_display(self, use_hq: bool = False):
        """Update cached pixmaps based on zoom and pan."""
        if not self._before_pil or not self._after_pil:
            return

        from PIL import Image as PILImage
        from PIL.ImageQt import ImageQt

        # Use after image dimensions (larger/upscaled)
        # Note: _before_pil is already pre-scaled to match _after_pil size in set_pil_images()
        img_w, img_h = self._after_pil.width, self._after_pil.height
        view_w, view_h = self.width() - 20, self.height() - 20
        if view_w < 100:
            view_w = 800
        if view_h < 100:
            view_h = 600

        # Calculate fit scale
        self._fit_scale = min(view_w / img_w, view_h / img_h, 1.0)

        if self._is_fit_mode:
            self._zoom_scale = self._fit_scale

        scale = self._zoom_scale

        # Resampling strategy:
        # - BOX is mathematically optimal for downscaling (averages all contributing pixels)
        # - LANCZOS is best for upscaling or small downscales (sharper)
        # - For interactive speed, use BOX (fast) then switch to LANCZOS after idle
        if use_hq:
            self._use_hq = True
        else:
            self._use_hq = False
            # Schedule high-quality re-render after 2 seconds of inactivity
            self._hq_render_timer.start()

        if self._is_fit_mode or scale <= self._fit_scale:
            # Fit mode - show entire image (significant downscale)
            display_w = int(img_w * scale)
            display_h = int(img_h * scale)

            if use_hq:
                # HQ: Use thumbnail() which is optimized for downscaling
                # It uses reducing_gap=3.0 + LANCZOS internally
                before_scaled = self._before_pil.copy()
                before_scaled.thumbnail((display_w, display_h), PILImage.Resampling.LANCZOS)
                after_scaled = self._after_pil.copy()
                after_scaled.thumbnail((display_w, display_h), PILImage.Resampling.LANCZOS)
            else:
                # Fast: BOX filter is fast and alias-free for downscaling
                before_scaled = self._before_pil.resize((display_w, display_h), PILImage.Resampling.BOX)
                after_scaled = self._after_pil.resize((display_w, display_h), PILImage.Resampling.BOX)
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

            # Crop region - _before_pil is already same size as _after_pil
            before_crop = self._before_pil.crop((int(left), int(top), int(right), int(bottom)))
            after_crop = self._after_pil.crop((int(left), int(top), int(right), int(bottom)))

            display_w = int(before_crop.width * scale)
            display_h = int(before_crop.height * scale)

            # For zoomed crops, scale is usually >= 1.0 (upscaling or slight downscale)
            if scale >= 1.0:
                # Upscaling - LANCZOS for HQ, BILINEAR for speed
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BILINEAR
            else:
                # Downscaling - BOX for alias-free, LANCZOS for HQ sharpening
                resample = PILImage.Resampling.LANCZOS if use_hq else PILImage.Resampling.BOX

            before_scaled = before_crop.resize((display_w, display_h), resample)
            after_scaled = after_crop.resize((display_w, display_h), resample)

        # Convert to QPixmap
        qimg_before = ImageQt(before_scaled)
        qimg_after = ImageQt(after_scaled)
        self._before_pixmap = QPixmap.fromImage(qimg_before)
        self._after_pixmap = QPixmap.fromImage(qimg_after)
        self._display_w = display_w
        self._display_h = display_h

        self.update()

    def _update_display_hq(self):
        """Re-render with high-quality LANCZOS resampling after idle timeout."""
        if not self._use_hq:  # Only re-render if not already HQ
            self._update_display(use_hq=True)

    def paintEvent(self, event):
        """Paint the split comparison view."""
        from PySide6.QtGui import QPainter, QPen, QColor

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._before_pixmap or not self._after_pixmap:
            painter.drawText(self.rect(), Qt.AlignCenter, "Load images to compare")
            return

        img_w = self._display_w
        img_h = self._display_h
        x_offset = (self.width() - img_w) // 2
        y_offset = (self.height() - img_h) // 2

        split_x = int(img_w * self._split_position)

        # Draw before image (left side)
        painter.drawPixmap(x_offset, y_offset, self._before_pixmap, 0, 0, split_x, img_h)

        # Draw after image (right side)
        painter.drawPixmap(x_offset + split_x, y_offset, self._after_pixmap, split_x, 0, img_w - split_x, img_h)

        # Draw split line
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x_offset + split_x, y_offset, x_offset + split_x, y_offset + img_h)

        # Draw handle
        handle_y = y_offset + img_h // 2
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(x_offset + split_x - 8, handle_y - 8, 16, 16)

        # Draw labels
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x_offset + 5, y_offset + 20, "Before")
        painter.drawText(x_offset + img_w - 45, y_offset + 20, "After")

        # Draw zoom indicator
        zoom_pct = int(self._zoom_scale * 100)
        painter.drawText(x_offset + 5, y_offset + img_h - 10, f"{zoom_pct}%")

    def wheelEvent(self, event):
        """Handle scroll for zooming."""
        if not self._after_pil:
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
        # Grab focus for wheel events
        self.setFocus()

        if not self._after_pixmap:
            return

        img_w = self._display_w
        x_offset = (self.width() - img_w) // 2

        if event.button() == Qt.LeftButton:
            # Left click - drag split slider
            if x_offset <= event.pos().x() <= x_offset + img_w:
                self._dragging_split = True
                self._update_split_from_mouse(event.pos().x())
        elif event.button() == Qt.RightButton:
            # Right click - pan
            if not self._is_fit_mode:
                self._dragging_pan = True
                self._drag_start_x = event.pos().x()
                self._drag_start_y = event.pos().y()
                self._drag_start_pan_x = self._pan_x
                self._drag_start_pan_y = self._pan_y
                self.setCursor(Qt.ClosedHandCursor)

    def enterEvent(self, event):
        """Grab focus when mouse enters for wheel events."""
        self.setFocus()
        super().enterEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self._dragging_split and self._after_pixmap:
            self._update_split_from_mouse(event.pos().x())
        elif self._dragging_pan and self._after_pil:
            dx = event.pos().x() - self._drag_start_x
            dy = event.pos().y() - self._drag_start_y

            img_w, img_h = self._after_pil.width, self._after_pil.height
            pan_dx = -dx / (img_w * self._zoom_scale)
            pan_dy = -dy / (img_h * self._zoom_scale)

            self._pan_x = max(0, min(1, self._drag_start_pan_x + pan_dx))
            self._pan_y = max(0, min(1, self._drag_start_pan_y + pan_dy))
            self._update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self._dragging_split = False
        elif event.button() == Qt.RightButton:
            self._dragging_pan = False
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset view."""
        if event.button() == Qt.LeftButton:
            self._is_fit_mode = True
            self._pan_x = 0.5
            self._pan_y = 0.5
            self._update_display()

    def _update_split_from_mouse(self, mouse_x: int):
        """Update split position based on mouse x coordinate."""
        if not self._after_pixmap:
            return

        img_w = self._display_w
        x_offset = (self.width() - img_w) // 2

        rel_x = mouse_x - x_offset
        self._split_position = max(0.0, min(1.0, rel_x / img_w))
        self.update()

    def resizeEvent(self, event):
        """Handle resize - update display."""
        super().resizeEvent(event)
        if self._after_pil:
            self._update_display()


class ComparisonDialog(QDialog):
    """
    Dialog for comparing before/after images with split view.
    Supports in-memory upscaling for quick comparison without saving to disk.
    """

    def __init__(self, before_path: str = "", after_path: str = "", parent=None, config=None, onnx_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Before/After Comparison")
        self.setMinimumSize(900, 700)

        # Enable minimize, maximize, and resize
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        self.before_path = before_path
        self.after_path = after_path
        self.config = config
        self.onnx_path = onnx_path
        self._upscale_worker = None
        self._before_pil = None  # Cache for in-memory comparison

        self._setup_ui()

        # If we have a before image and config, enable upscale button
        if before_path and config and onnx_path:
            self._load_before_only()
        elif before_path and after_path:
            self._load_images()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        self.info_label = QLabel("Scroll to zoom, Right-drag to pan, Left-drag to move slider")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.info_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Split compare widget
        self.compare_widget = SplitCompareWidget()
        layout.addWidget(self.compare_widget, 1)

        # Path selection row (wrap in widget to hide in fullscreen)
        self.path_widget = QWidget()
        path_layout = QHBoxLayout(self.path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)

        path_layout.addWidget(QLabel("Before:"))
        self.before_edit = QLineEdit(self.before_path)
        self.before_edit.setReadOnly(True)
        path_layout.addWidget(self.before_edit, 1)
        btn_before = QPushButton("Browse")
        btn_before.clicked.connect(self._browse_before)
        path_layout.addWidget(btn_before)

        path_layout.addWidget(QLabel("After:"))
        self.after_edit = QLineEdit(self.after_path if self.after_path else "(click Upscale)")
        self.after_edit.setReadOnly(True)
        path_layout.addWidget(self.after_edit, 1)
        btn_after = QPushButton("Browse")
        btn_after.clicked.connect(self._browse_after)
        path_layout.addWidget(btn_after)

        layout.addWidget(self.path_widget)

        # Buttons (wrap in widget to hide in fullscreen)
        self.btn_widget = QWidget()
        btn_layout = QHBoxLayout(self.btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        # Upscale button (only if we have config)
        self.upscale_btn = QPushButton("Upscale to Memory")
        self.upscale_btn.setToolTip("Upscale the 'before' image to RAM for quick comparison (no file saved)")
        self.upscale_btn.clicked.connect(self._upscale_to_memory)
        self.upscale_btn.setEnabled(bool(self.config and self.onnx_path and self.before_path))
        btn_layout.addWidget(self.upscale_btn)

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

    def _browse_before(self):
        """Browse for before image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Before Image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)"
        )
        if path:
            self.before_path = path
            self.before_edit.setText(path)
            self._load_before_only()
            self.upscale_btn.setEnabled(bool(self.config and self.onnx_path))

    def _browse_after(self):
        """Browse for after image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select After Image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)"
        )
        if path:
            self.after_path = path
            self.after_edit.setText(path)
            self._load_images()

    def _load_before_only(self):
        """Load just the before image (for preview before upscaling)."""
        from PIL import Image as PILImage
        import os

        if self.before_path and os.path.exists(self.before_path):
            try:
                self._before_pil = PILImage.open(self.before_path)
                if self._before_pil.mode not in ("RGB", "RGBA"):
                    self._before_pil = self._before_pil.convert("RGB")
                self._before_pil = self._before_pil.copy()

                # Show before image on both sides initially
                self.compare_widget.set_pil_images(self._before_pil, self._before_pil)
                self.info_label.setText("Click 'Upscale to Memory' to generate comparison")
            except Exception as e:
                print(f"Failed to load before image: {e}")

    def _load_images(self):
        """Load images into comparison widget from file paths."""
        if self.before_path and self.after_path:
            import os
            if os.path.exists(self.before_path) and os.path.exists(self.after_path):
                self.compare_widget.set_images(self.before_path, self.after_path)
                self.info_label.setText("Scroll to zoom, Right-drag to pan, Left-drag to move slider")

    def _upscale_to_memory(self):
        """Upscale the before image to memory for comparison."""
        if not self.before_path or not self.config or not self.onnx_path:
            return

        import os
        if not os.path.exists(self.before_path):
            QMessageBox.warning(self, "Error", "Before image not found.")
            return

        if not os.path.exists(self.onnx_path):
            QMessageBox.warning(self, "Error", "ONNX model not found.")
            return

        # Disable button during processing
        self.upscale_btn.setEnabled(False)
        self.upscale_btn.setText("Upscaling...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Create and start worker
        from PySide6.QtCore import QThread, Signal
        import numpy as np

        class UpscaleToMemoryWorker(QThread):
            progress = Signal(int, int)
            finished = Signal(bool, str, object)  # success, message, PIL image or None

            def __init__(self, image_path, config, onnx_path):
                super().__init__()
                self.image_path = image_path
                self.config = config
                self.onnx_path = onnx_path

            def run(self):
                upscaler = None
                try:
                    import numpy as np
                    import gc
                    from PIL import Image as PILImage
                    from tensorrt_upscaler.upscaler import ImageUpscaler
                    from tensorrt_upscaler.fast_io import load_image_fast

                    cfg = self.config

                    # Load image
                    img, has_alpha = load_image_fast(self.image_path)

                    # Create upscaler
                    upscaler = ImageUpscaler(
                        onnx_path=self.onnx_path,
                        tile_size=(cfg.tile_width, cfg.tile_height),
                        overlap=cfg.tile_overlap,
                        fp16=cfg.use_fp16,
                        bf16=cfg.use_bf16,
                    )

                    # Upscale with progress
                    def on_progress(current, total):
                        self.progress.emit(current, total)

                    result = upscaler.upscale_array(img, has_alpha, on_progress)

                    # Release VRAM immediately after upscale (result is in system RAM)
                    del upscaler
                    upscaler = None
                    gc.collect()

                    # Convert to PIL
                    result_uint8 = (result * 255.0).clip(0, 255).astype(np.uint8)
                    if has_alpha and result.shape[2] == 4:
                        pil_result = PILImage.fromarray(result_uint8, mode="RGBA")
                    else:
                        pil_result = PILImage.fromarray(result_uint8, mode="RGB")

                    self.finished.emit(True, "Upscaling complete", pil_result)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.finished.emit(False, str(e), None)
                finally:
                    # Ensure VRAM is released even on error
                    if upscaler is not None:
                        del upscaler
                        import gc
                        gc.collect()

        self._upscale_worker = UpscaleToMemoryWorker(self.before_path, self.config, self.onnx_path)
        self._upscale_worker.progress.connect(self._on_upscale_progress)
        self._upscale_worker.finished.connect(self._on_upscale_finished)
        self._upscale_worker.start()

    def _on_upscale_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            pct = int(current / total * 1000)
            self.progress_bar.setValue(pct)

    def _on_upscale_finished(self, success: bool, message: str, pil_result):
        """Handle upscale completion."""
        self.upscale_btn.setEnabled(True)
        self.upscale_btn.setText("Upscale to Memory")
        self.progress_bar.setVisible(False)

        if success and pil_result and self._before_pil:
            self.compare_widget.set_pil_images(self._before_pil, pil_result)
            self.after_edit.setText("(in memory)")
            self.info_label.setText("Ctrl+Scroll to zoom, Right-drag to pan, Left-drag to move slider")
        else:
            QMessageBox.warning(self, "Upscale Failed", f"Failed to upscale: {message}")

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode with hidden toolbar."""
        if self.isFullScreen():
            # Exit fullscreen - show toolbar
            self.showNormal()
            self.fullscreen_btn.setText("Fullscreen (F11)")
            self.info_label.setVisible(True)
            self.path_widget.setVisible(True)
            self.btn_widget.setVisible(True)
        else:
            # Enter fullscreen - hide toolbar for true fullscreen
            self.info_label.setVisible(False)
            self.path_widget.setVisible(False)
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
            from .upscaler import ImageUpscaler

            # Create upscaler (will be released after use to free VRAM)
            upscaler = ImageUpscaler(
                onnx_path=self.onnx_path,
                tile_size=(self.config.tile_width, self.config.tile_height),
                overlap=self.config.tile_overlap,
                fp16=self.config.use_fp16,
                bf16=self.config.use_bf16,
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
                f"Input: {orig_w}x{orig_h} → Output: {result.shape[1]}x{result.shape[0]} | "
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
