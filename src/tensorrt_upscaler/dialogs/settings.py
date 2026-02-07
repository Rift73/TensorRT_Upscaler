"""
Settings dialog.
QoL features: skip existing, conditional processing, aspect ratio filter, presets, web extraction.
"""

import json
import subprocess
import sys

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QWidget,
    QListWidget,
    QMessageBox,
    QTabWidget,
    QFormLayout,
    QInputDialog,
)

from ..config import get_config, save_config


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

        # Tile settings group
        tile_group = QGroupBox("Tile Settings")
        tile_layout = QVBoxLayout(tile_group)
        self.disable_tile_limit = QCheckBox("Disable tile alignment limit")
        self.disable_tile_limit.setToolTip(
            "When enabled, allows any tile size (not just multiples of 64).\n"
            "Also disables automatic padding. Use with caution - some models\n"
            "may require input dimensions to be multiples of 64."
        )
        tile_layout.addWidget(self.disable_tile_limit)
        processing_layout.addWidget(tile_group)

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

        # Tab 3: Web Extraction
        web_tab = QWidget()
        web_layout = QVBoxLayout(web_tab)

        web_info = QLabel(
            "Extract images from web pages by pasting a URL. Uses Playwright for "
            "JavaScript rendering and can use browser cookies for authenticated pages."
        )
        web_info.setWordWrap(True)
        web_info.setStyleSheet("color: #888; font-style: italic;")
        web_layout.addWidget(web_info)

        # Browser cookie source
        cookie_group = QGroupBox("Browser Cookies")
        cookie_layout = QFormLayout(cookie_group)

        self.web_browser_combo = QComboBox()
        self.web_browser_combo.addItems([
            "None (no cookies)",
            "Chrome",
            "Firefox",
            "Edge",
            "Chromium",
            "Brave"
        ])
        self.web_browser_combo.setToolTip(
            "Select browser to copy cookies from for authenticated pages.\n"
            "The browser must be closed when extracting cookies."
        )
        cookie_layout.addRow("Cookie source:", self.web_browser_combo)
        web_layout.addWidget(cookie_group)

        # Wait time
        timing_group = QGroupBox("Page Loading")
        timing_layout = QFormLayout(timing_group)

        self.web_wait_spin = QDoubleSpinBox()
        self.web_wait_spin.setRange(0.5, 30.0)
        self.web_wait_spin.setSingleStep(0.5)
        self.web_wait_spin.setDecimals(1)
        self.web_wait_spin.setSuffix(" seconds")
        self.web_wait_spin.setToolTip(
            "Time to wait for JavaScript to load images.\n"
            "Increase this for slow-loading pages."
        )
        timing_layout.addRow("Wait time:", self.web_wait_spin)
        web_layout.addWidget(timing_group)

        # Install button
        install_group = QGroupBox("Dependencies")
        install_layout = QVBoxLayout(install_group)
        install_info = QLabel(
            "Web extraction requires Playwright browser automation.\n"
            "Click below to install if not already installed."
        )
        install_info.setWordWrap(True)
        install_layout.addWidget(install_info)

        self.btn_install_playwright = QPushButton("Install Playwright")
        self.btn_install_playwright.clicked.connect(self._install_playwright)
        install_layout.addWidget(self.btn_install_playwright)
        web_layout.addWidget(install_group)

        web_layout.addStretch()
        tabs.addTab(web_tab, "Web Extract")

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

        # Tile settings
        self.disable_tile_limit.setChecked(cfg.disable_tile_limit)

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

        # Web extraction
        browser_map = {"none": 0, "chrome": 1, "firefox": 2, "edge": 3, "chromium": 4, "brave": 5}
        self.web_browser_combo.setCurrentIndex(browser_map.get(cfg.web_extract_browser, 0))
        self.web_wait_spin.setValue(cfg.web_extract_wait_time)

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

        # Tile settings
        cfg.disable_tile_limit = self.disable_tile_limit.isChecked()

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

        # Web extraction
        browser_map = {0: "none", 1: "chrome", 2: "firefox", 3: "edge", 4: "chromium", 5: "brave"}
        cfg.web_extract_browser = browser_map.get(self.web_browser_combo.currentIndex(), "none")
        cfg.web_extract_wait_time = self.web_wait_spin.value()

        save_config()
        self.accept()

    def _install_playwright(self):
        """Install Playwright and browser."""
        reply = QMessageBox.question(
            self,
            "Install Playwright",
            "This will install Playwright and download Chromium browser (~150MB).\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self.btn_install_playwright.setEnabled(False)
            self.btn_install_playwright.setText("Installing...")
            QApplication.processEvents()

            # Install playwright package
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "playwright", "browser_cookie3"],
                check=True
            )

            # Install chromium browser
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=True
            )

            QMessageBox.information(self, "Success", "Playwright installed successfully!")
            self.btn_install_playwright.setText("Installed \u2713")

        except subprocess.CalledProcessError as e:
            QMessageBox.warning(self, "Error", f"Failed to install Playwright:\n{e}")
            self.btn_install_playwright.setEnabled(True)
            self.btn_install_playwright.setText("Install Playwright")
