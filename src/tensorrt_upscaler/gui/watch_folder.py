"""
Watch folder mode mixin for MainWindow.
"""

import os
import time
from pathlib import Path

from PySide6.QtCore import QTimer, QFileSystemWatcher
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..utils import IMAGE_EXTENSIONS, natural_sort_key


class WatchFolderMixin:
    """Mixin providing watch folder mode functionality."""

    def _toggle_watch_folder(self):
        """Toggle watch folder mode."""
        if self._watch_folder:
            self._stop_watch_folder()
        else:
            self._start_watch_folder()

    def _start_watch_folder(self):
        """Start watching a folder for new images."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Watch",
            self._input_edit.text() or ""
        )
        if not folder:
            self._watch_button.setChecked(False)
            return

        # Validate ONNX model is set
        onnx_path = self.onnx_edit.text()
        if self._upscale_check.isChecked() and (not onnx_path or not os.path.exists(onnx_path)):
            QMessageBox.warning(self, "No Model", "Please select a valid ONNX model before starting watch mode.")
            self._watch_button.setChecked(False)
            return

        # Setup file watcher
        self._watch_folder = folder
        self._file_watcher = QFileSystemWatcher([folder], self)
        self._file_watcher.directoryChanged.connect(self._on_watch_folder_changed)

        # Setup delay timer for debouncing file changes
        self._watch_delay_timer = QTimer(self)
        self._watch_delay_timer.setSingleShot(True)
        self._watch_delay_timer.timeout.connect(self._process_watch_pending)

        # Update UI
        self._watch_button.setChecked(True)
        self._watch_button.setText("Stop Watch")
        self._progress_label.setText(f"Watching: {folder}")

        # Get initial file list to track new additions
        self._watch_existing_files = set(
            str(p) for p in Path(folder).glob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        # Add to log
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_entries.append(f"[{timestamp}] Started watching: {folder}")

    def _stop_watch_folder(self):
        """Stop watching folder."""
        if self._file_watcher:
            self._file_watcher.deleteLater()
            self._file_watcher = None

        if self._watch_delay_timer:
            self._watch_delay_timer.stop()
            self._watch_delay_timer = None

        self._watch_folder = None
        self._watch_pending_files.clear()
        self._watch_processing = False

        # Update UI
        self._watch_button.setChecked(False)
        self._watch_button.setText("Watch Folder")
        self._progress_label.setText("Watch mode stopped")

        # Add to log
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_entries.append(f"[{timestamp}] Stopped watch mode")

    def _on_watch_folder_changed(self, path: str):
        """Handle folder content change."""
        if not self._watch_folder:
            return

        # Find new files
        current_files = set(
            str(p) for p in Path(self._watch_folder).glob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        new_files = current_files - self._watch_existing_files
        self._watch_existing_files = current_files

        if new_files:
            for f in sorted(new_files, key=natural_sort_key):
                if f not in self._watch_pending_files:
                    self._watch_pending_files.append(f)

            # Start/restart delay timer (wait for file to finish writing)
            if self._watch_delay_timer:
                self._watch_delay_timer.start(1000)

    def _process_watch_pending(self):
        """Process pending files from watch folder."""
        if not self._watch_pending_files or self._watch_processing:
            return

        if self.worker and self.worker.isRunning():
            return

        # Get files that are ready (fully written)
        ready_files = []
        still_pending = []

        for file_path in self._watch_pending_files:
            if self._is_file_ready(file_path):
                ready_files.append(file_path)
            else:
                still_pending.append(file_path)

        self._watch_pending_files = still_pending

        if ready_files:
            self._watch_processing = True
            self.files = ready_files
            self._update_file_list_ui()
            self._input_edit.setText(f"{len(ready_files)} new files")
            self._start_upscaling()

    def _is_file_ready(self, file_path: str) -> bool:
        """Check if file is ready (not still being written)."""
        try:
            if not os.path.exists(file_path):
                return False
            size1 = os.path.getsize(file_path)
            time.sleep(0.1)
            size2 = os.path.getsize(file_path)
            return size1 == size2 and size1 > 0
        except (OSError, PermissionError):
            return False
