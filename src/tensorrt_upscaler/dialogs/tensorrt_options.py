"""
TensorRT Options Dialog - Configure precision, optimization, and CUDA graph settings.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QGridLayout,
    QSpinBox,
)
from PySide6.QtCore import Qt


class TensorRTOptionsDialog(QDialog):
    """Dialog for configuring TensorRT precision and optimization settings."""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("TensorRT Options")
        self.setMinimumWidth(450)

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
            "Ampere+ GPUs only. Faster matmuls with minimal quality loss."
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

        # Builder optimization level
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Builder Optimization Level:"))

        self._opt_level_spin = QSpinBox()
        self._opt_level_spin.setRange(0, 5)
        self._opt_level_spin.setValue(5)
        self._opt_level_spin.setToolTip(
            "TensorRT builder optimization level (0-5).\n\n"
            "Higher levels explore more tactics for better runtime performance,\n"
            "but take longer to build the engine.\n\n"
            "0 = Fastest build, basic optimization\n"
            "3 = Balanced build time and performance\n"
            "5 = Slowest build, maximum optimization (default)"
        )
        opt_row.addWidget(self._opt_level_spin)

        self._opt_level_label = QLabel("(Maximum)")
        opt_row.addWidget(self._opt_level_label)
        opt_row.addStretch()

        self._opt_level_spin.valueChanged.connect(self._on_opt_level_changed)

        optim_layout.addLayout(opt_row, 0, 0, 1, 2)

        # CUDA graphs checkbox
        self._cuda_graphs_check = QCheckBox("Enable CUDA Graphs")
        self._cuda_graphs_check.setToolTip(
            "Use CUDA graphs to reduce kernel launch overhead.\n\n"
            "Benefits:\n"
            "- Reduces CPU overhead from kernel launches\n"
            "- Can provide 5-20% speedup on small tile sizes\n"
            "- Most effective for repetitive workloads\n\n"
            "Caveats:\n"
            "- Requires TensorRT 8.6+\n"
            "- May not work with all models\n"
            "- Requires engine rebuild when enabled/disabled\n\n"
            "Leave disabled if you encounter issues."
        )
        optim_layout.addWidget(self._cuda_graphs_check, 1, 0, 1, 2)

        layout.addWidget(optim_group)

        # Info label
        info_label = QLabel(
            "Note: Changing these settings will require rebuilding the TensorRT engine.\n"
            "Engine rebuild can take several minutes depending on tile size and optimization level."
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

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

    def _on_opt_level_changed(self, value):
        """Update label when optimization level changes."""
        labels = {
            0: "(Fastest build)",
            1: "(Fast build)",
            2: "(Normal)",
            3: "(Good)",
            4: "(Better)",
            5: "(Maximum)",
        }
        self._opt_level_label.setText(labels.get(value, ""))

    def _load_from_config(self):
        """Load settings from config."""
        if self.config is None:
            return

        self._fp16_check.setChecked(getattr(self.config, 'use_fp16', False))
        self._bf16_check.setChecked(getattr(self.config, 'use_bf16', True))
        self._tf32_check.setChecked(getattr(self.config, 'use_tf32', False))
        self._opt_level_spin.setValue(getattr(self.config, 'trt_builder_optimization', 5))
        self._cuda_graphs_check.setChecked(getattr(self.config, 'trt_cuda_graphs', False))

    def _save_to_config(self):
        """Save settings to config."""
        if self.config is None:
            return

        self.config.use_fp16 = self._fp16_check.isChecked()
        self.config.use_bf16 = self._bf16_check.isChecked()
        self.config.use_tf32 = self._tf32_check.isChecked()
        self.config.trt_builder_optimization = self._opt_level_spin.value()
        self.config.trt_cuda_graphs = self._cuda_graphs_check.isChecked()

    def _restore_defaults(self):
        """Restore default settings."""
        self._fp16_check.setChecked(False)
        self._bf16_check.setChecked(True)
        self._tf32_check.setChecked(False)
        self._opt_level_spin.setValue(5)
        self._cuda_graphs_check.setChecked(False)

    def _on_ok(self):
        """Save and close."""
        self._save_to_config()
        self.accept()

    def get_settings(self) -> dict:
        """Get current settings as a dictionary."""
        return {
            "use_fp16": self._fp16_check.isChecked(),
            "use_bf16": self._bf16_check.isChecked(),
            "use_tf32": self._tf32_check.isChecked(),
            "trt_builder_optimization": self._opt_level_spin.value(),
            "trt_cuda_graphs": self._cuda_graphs_check.isChecked(),
        }
