"""
System tray integration mixin for MainWindow.
"""

from PySide6.QtCore import QTimer, QEvent
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMenu, QStyle, QSystemTrayIcon,
)


class TrayManagerMixin:
    """Mixin providing system tray icon functionality."""

    def _setup_tray_icon(self):
        """Setup system tray icon for minimize to tray feature."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray_icon = QSystemTrayIcon(self)
        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self._tray_icon.setIcon(icon)
        self._tray_icon.setToolTip("TensorRT Upscaler v2")

        tray_menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self._show_from_tray)
        tray_menu.addAction(show_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)

        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)

    def _show_from_tray(self):
        """Show window from tray."""
        self.showNormal()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_from_tray()

    def changeEvent(self, event):
        """Handle window state changes for minimize to tray."""
        if event.type() == QEvent.WindowStateChange:
            if self.isMinimized() and self.config.minimize_to_tray and self._tray_icon:
                QTimer.singleShot(0, self._hide_to_tray)
        super().changeEvent(event)

    def _hide_to_tray(self):
        """Hide window to system tray."""
        self.hide()
        if self._tray_icon:
            self._tray_icon.show()
            self._tray_icon.showMessage(
                "TensorRT Upscaler",
                "Minimized to tray. Double-click to restore.",
                QSystemTrayIcon.Information,
                2000
            )
