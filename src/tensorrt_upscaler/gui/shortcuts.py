"""
Keyboard shortcut management mixin for MainWindow.
"""

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

from ..config import save_config


class ShortcutsMixin:
    """Mixin providing keyboard shortcut setup and handlers."""

    def _setup_keyboard_shortcuts(self):
        """Setup global keyboard shortcuts."""
        if not self.config.shortcuts_enabled:
            return

        # Start/Stop processing - Enter/Escape
        start_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        start_shortcut.activated.connect(self._shortcut_start)

        cancel_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        cancel_shortcut.activated.connect(self._shortcut_cancel)

        # File operations
        open_file_shortcut = QShortcut(QKeySequence("Ctrl+I"), self)
        open_file_shortcut.activated.connect(self._browse_input_file)

        open_folder_shortcut = QShortcut(QKeySequence("Ctrl+Shift+I"), self)
        open_folder_shortcut.activated.connect(self._browse_input_folder)

        open_output_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        open_output_shortcut.activated.connect(self._open_output_folder)

        copy_path_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_path_shortcut.activated.connect(self._copy_output_path)

        log_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        log_shortcut.activated.connect(self._open_log_dialog)

        settings_shortcut = QShortcut(QKeySequence("Ctrl+,"), self)
        settings_shortcut.activated.connect(self._open_settings_dialog)

        watch_shortcut = QShortcut(QKeySequence("Ctrl+W"), self)
        watch_shortcut.activated.connect(self._toggle_watch_folder)

        paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        paste_shortcut.activated.connect(self._handle_clipboard_paste)

        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        delete_shortcut.activated.connect(self._remove_selected_files)

        clear_shortcut = QShortcut(QKeySequence("Ctrl+Delete"), self)
        clear_shortcut.activated.connect(self._clear_file_list)

        refresh_shortcut = QShortcut(QKeySequence(Qt.Key_F5), self)
        refresh_shortcut.activated.connect(self._refresh_input)

        always_on_top_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        always_on_top_shortcut.activated.connect(self._toggle_always_on_top)

        zoom_shortcut = QShortcut(QKeySequence(Qt.Key_Z), self)
        zoom_shortcut.activated.connect(self._toggle_zoom)

    def _shortcut_start(self):
        """Handle Enter shortcut - start if not running."""
        if self._start_button.isEnabled():
            self._start_upscaling()

    def _shortcut_cancel(self):
        """Handle Escape shortcut - cancel if running."""
        if self._cancel_button.isEnabled():
            self._cancel()

    def _refresh_input(self):
        """Refresh input - re-collect files from current path."""
        input_text = self._input_edit.text()
        if not input_text or "files" in input_text:
            return
        if os.path.exists(input_text):
            self._set_inputs_from_paths([Path(input_text)])

    def _toggle_always_on_top(self):
        """Toggle always on top setting."""
        self.config.always_on_top = not self.config.always_on_top
        save_config()
        self._apply_window_flags()
        status = "ON" if self.config.always_on_top else "OFF"
        self._progress_label.setText(f"Always on top: {status}")
