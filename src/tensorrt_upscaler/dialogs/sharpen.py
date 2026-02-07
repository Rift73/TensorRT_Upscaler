"""
Sharpen settings dialog.
CAS and Adaptive Sharpen algorithm selection and configuration.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
)

from ..config import get_config, save_config


class SharpenDialog(QDialog):
    """
    Dialog for sharpening settings.
    Allows selection between CAS and Adaptive Sharpen algorithms.
    """

    # Sharpen method options
    METHODS = ["None", "CAS (Contrast Adaptive)", "Adaptive Sharpen"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sharpen Settings")
        self.setMinimumWidth(320)

        self.config = get_config()
        self._setup_ui()
        self._load_from_config()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Method selection
        method_group = QGroupBox("Sharpen Method")
        method_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(self.METHODS)
        method_layout.addWidget(self.method_combo)

        # Method descriptions
        self.method_desc = QLabel("")
        self.method_desc.setWordWrap(True)
        self.method_desc.setStyleSheet("color: gray; font-size: 11px;")
        method_layout.addWidget(self.method_desc)

        layout.addWidget(method_group)

        # Strength settings
        strength_group = QGroupBox("Strength")
        strength_layout = QHBoxLayout(strength_group)

        strength_layout.addWidget(QLabel("Amount:"))
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.01, 2.00)
        self.strength_spin.setSingleStep(0.01)
        self.strength_spin.setValue(0.50)
        self.strength_spin.setFixedWidth(70)
        strength_layout.addWidget(self.strength_spin)
        strength_layout.addStretch()

        layout.addWidget(strength_group)

        # Anime mode checkbox (only for Adaptive Sharpen)
        self.anime_mode_check = QCheckBox("Anime Mode (darken edges only)")
        self.anime_mode_check.setToolTip(
            "Only darken edges, no brightening. Better for anime/cartoon content."
        )
        self.anime_mode_check.setEnabled(False)  # Enabled only for Adaptive Sharpen
        layout.addWidget(self.anime_mode_check)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self._save_and_close)
        btn_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def _connect_signals(self):
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

    def _on_method_changed(self, index: int):
        """Update description and enable/disable strength based on method."""
        descriptions = [
            "No sharpening applied.",
            "AMD FidelityFX Contrast Adaptive Sharpening. Fast, good for general use. Strength 0.0-1.0.",
            "Advanced edge-aware sharpening with anti-ringing. Better quality, slightly slower. Strength 0.0-2.0.",
        ]
        self.method_desc.setText(descriptions[index])

        # Enable strength only if method is not None
        self.strength_spin.setEnabled(index > 0)

        # Anime mode only available for Adaptive Sharpen
        self.anime_mode_check.setEnabled(index == 2)

        # Adjust range based on method
        if index == 1:  # CAS
            self.strength_spin.setRange(0.01, 1.00)
            if self.strength_spin.value() > 1.0:
                self.strength_spin.setValue(1.0)
        elif index == 2:  # Adaptive Sharpen
            self.strength_spin.setRange(0.01, 2.00)

    def _load_from_config(self):
        """Load settings from config."""
        cfg = self.config

        # Load method (default to None if not set)
        method = getattr(cfg, 'sharpen_method', 'none')
        if method == 'cas':
            self.method_combo.setCurrentIndex(1)
        elif method == 'adaptive':
            self.method_combo.setCurrentIndex(2)
        else:
            self.method_combo.setCurrentIndex(0)

        # Load strength
        self.strength_spin.setValue(cfg.sharpen_value)

        # Load anime mode
        self.anime_mode_check.setChecked(getattr(cfg, 'sharpen_anime_mode', False))

        # Trigger description update
        self._on_method_changed(self.method_combo.currentIndex())

    def _save_and_close(self):
        """Save settings to config and close."""
        cfg = self.config

        # Save method
        index = self.method_combo.currentIndex()
        if index == 0:
            cfg.sharpen_method = 'none'
            cfg.sharpen_enabled = False
        elif index == 1:
            cfg.sharpen_method = 'cas'
            cfg.sharpen_enabled = True
        else:
            cfg.sharpen_method = 'adaptive'
            cfg.sharpen_enabled = True

        # Save strength
        cfg.sharpen_value = self.strength_spin.value()

        # Save anime mode
        cfg.sharpen_anime_mode = self.anime_mode_check.isChecked()

        save_config()
        self.accept()

    def get_summary(self) -> str:
        """Return a short summary of current settings for button text."""
        index = self.method_combo.currentIndex()
        if index == 0:
            return "None"
        elif index == 1:
            return f"CAS {self.strength_spin.value():.2f}"
        else:
            return f"Adaptive {self.strength_spin.value():.2f}"
