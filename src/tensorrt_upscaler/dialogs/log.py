"""
Processing log dialog.
Shows batch processing results and allows exporting to CSV.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
)


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
