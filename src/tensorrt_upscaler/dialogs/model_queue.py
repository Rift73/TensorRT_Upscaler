"""
Model queue dialog.
Manages a queue of ONNX models for sequential batch processing.
"""

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
)

from ..config import get_config, save_config


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
