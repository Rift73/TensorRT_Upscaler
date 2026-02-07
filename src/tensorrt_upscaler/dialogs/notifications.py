"""
Notifications and window behavior settings dialog.
System tray, sound, always-on-top, minimize-to-tray, auto-shutdown options.
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
    QLineEdit,
    QFileDialog,
)

from ..config import get_config, save_config


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
