"""
Dependencies installation window for TensorRT Image Upscaler.

Handles installation of:
- Python packages via pip (PySide6, numpy, Pillow, opencv-python, numba, fpng-py)
- TensorRT and CUDA Python packages
- External tools (ffmpeg, gifski, avifenc, pngquant, pingo) with PATH setup
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QMessageBox,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget as QWidgetType


# === Configuration ===

# Python packages to install via pip
PIP_PACKAGES = [
    "PySide6>=6.5.0",
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
    "opencv-python>=4.8.0",
    "numba>=0.58.0",
    "fpng-py>=0.0.4",
]

# CUDA/TensorRT packages (separate due to special handling)
CUDA_PACKAGES = [
    "cuda-python",
    "pycuda",
    "tensorrt>=10.0.0",
]

# DirectML packages (alternative backend for AMD/Intel GPUs)
DIRECTML_PACKAGES = [
    "onnxruntime-directml>=1.16.0",
]

# PyTorch packages (alternative backend supporting .pth/.safetensors models)
# Note: torch itself should be installed separately for CUDA version choice
PYTORCH_PACKAGES = [
    "spandrel>=0.3.0",
    "spandrel_extra_arches>=0.1.0",
    "safetensors>=0.4.0",
]

# Optional: HuggingFace Accelerate for better layer offloading in Low VRAM mode
# RamTorch for layer-by-layer offloading
PYTORCH_OPTIONAL_PACKAGES = [
    "accelerate>=0.25.0",
    "ramtorch",
]

# Web extraction packages (for extracting images from web pages)
WEB_EXTRACT_PACKAGES = [
    "playwright>=1.40.0",
    "browser_cookie3>=0.19.0",
]

# External tools to download
# Format: (name, url, extract_type, path_subdir)
# path_subdir: subdirectory within TOOLS_DIR to add to PATH
EXTERNAL_TOOLS = [
    (
        "ffmpeg",
        "https://github.com/nekotrix/FFmpeg-Builds-SVT-AV1-Essential/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "zip",
        "ffmpeg-master-latest-win64-gpl/bin",
    ),
    (
        "gifski",
        "https://gif.ski/gifski-1.32.0.zip",
        "zip",
        "win",
    ),
    (
        "avifenc",
        "https://github.com/AOMediaCodec/libavif/releases/download/v1.3.0/windows-artifacts.zip",
        "zip",
        "",
    ),
    (
        "pngquant",
        "https://github.com/jibsen/pngquant-winbuild/releases/download/v2.17.0/pngquant-2.17.0-win-x64.zip",
        "zip",
        "",
    ),
    (
        "pingo",
        "https://css-ig.net/bin/pingo-win64.zip",
        "zip",
        "",
    ),
]

# Installation directories
APPDATA = Path(os.environ.get("APPDATA", ""))
TOOLS_DIR = APPDATA / "tensorrt-upscaler-GUI"


def _is_package_installed(package: str) -> bool:
    """Check if a Python package is already installed."""
    # Extract package name without version specifier
    pkg_name = package.split(">=")[0].split("==")[0].split("<")[0].strip()
    # Handle package names with hyphens vs underscores (pip normalizes them)
    pkg_name_normalized = pkg_name.replace("-", "_").lower()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def _is_tool_installed(tool_name: str) -> bool:
    """Check if an external tool is available in PATH or TOOLS_DIR."""
    # Check if tool exists in PATH
    if shutil.which(tool_name):
        return True
    if shutil.which(f"{tool_name}.exe"):
        return True

    # Check in TOOLS_DIR and subdirectories
    if TOOLS_DIR.exists():
        for exe in TOOLS_DIR.rglob(f"{tool_name}.exe"):
            return True
        for exe in TOOLS_DIR.rglob(tool_name):
            if exe.is_file():
                return True

    return False


class InstallWorker(QThread):
    """Worker thread for installing dependencies."""

    progress_signal = Signal(str)  # Log message
    status_signal = Signal(str, int, int)  # (status, current, total)
    finished_signal = Signal(bool, str)  # (success, message)

    def __init__(
        self,
        install_pip: bool = True,
        install_tensorrt: bool = True,
        install_directml: bool = True,
        install_pytorch: bool = False,
        install_pytorch_optional: bool = False,
        install_tools: bool = True,
    ):
        super().__init__()
        self._install_pip = install_pip
        self._install_tensorrt = install_tensorrt
        self._install_directml = install_directml
        self._install_pytorch = install_pytorch
        self._install_pytorch_optional = install_pytorch_optional
        self._install_tools = install_tools
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    def run(self) -> None:
        """Run the installation process."""
        try:
            # First, check what's already installed
            self.progress_signal.emit("Checking installed packages...\n")

            pip_to_install = []
            cuda_to_install = []
            dml_to_install = []
            pytorch_to_install = []
            tools_to_install = []

            if self._install_pip:
                for pkg in PIP_PACKAGES:
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    if _is_package_installed(pkg):
                        self.progress_signal.emit(f"  {pkg_name}: already installed")
                    else:
                        pip_to_install.append(pkg)
                        self.progress_signal.emit(f"  {pkg_name}: needs installation")

            if self._install_tensorrt:
                for pkg in CUDA_PACKAGES:
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    if _is_package_installed(pkg):
                        self.progress_signal.emit(f"  {pkg_name}: already installed")
                    else:
                        cuda_to_install.append(pkg)
                        self.progress_signal.emit(f"  {pkg_name}: needs installation")

            if self._install_directml:
                for pkg in DIRECTML_PACKAGES:
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    if _is_package_installed(pkg):
                        self.progress_signal.emit(f"  {pkg_name}: already installed")
                    else:
                        dml_to_install.append(pkg)
                        self.progress_signal.emit(f"  {pkg_name}: needs installation")

            if self._install_pytorch:
                for pkg in PYTORCH_PACKAGES:
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    if _is_package_installed(pkg):
                        self.progress_signal.emit(f"  {pkg_name}: already installed")
                    else:
                        pytorch_to_install.append(pkg)
                        self.progress_signal.emit(f"  {pkg_name}: needs installation")

            if self._install_pytorch_optional:
                for pkg in PYTORCH_OPTIONAL_PACKAGES:
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    if _is_package_installed(pkg):
                        self.progress_signal.emit(f"  {pkg_name}: already installed")
                    else:
                        pytorch_to_install.append(pkg)
                        self.progress_signal.emit(f"  {pkg_name}: needs installation (optional)")

            if self._install_tools:
                for name, url, extract_type, path_subdir in EXTERNAL_TOOLS:
                    if _is_tool_installed(name):
                        self.progress_signal.emit(f"  {name}: already installed")
                    else:
                        tools_to_install.append((name, url, extract_type, path_subdir))
                        self.progress_signal.emit(f"  {name}: needs installation")

            # Calculate total steps (only packages that need installation)
            total_steps = len(pip_to_install) + len(cuda_to_install) + len(dml_to_install) + len(pytorch_to_install) + len(tools_to_install)

            if total_steps == 0:
                self.progress_signal.emit("\n=== All dependencies already installed! ===\n")
                self.finished_signal.emit(True, "All dependencies already installed!")
                return

            self.progress_signal.emit(f"\n{total_steps} package(s) to install...\n")
            current_step = 0

            # Install pip packages
            if pip_to_install:
                self.progress_signal.emit("\n=== Installing Python packages ===\n")
                for pkg in pip_to_install:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    self.status_signal.emit(f"Installing {pkg_name}...", current_step, total_steps)
                    self.progress_signal.emit(f"Installing {pkg}...")
                    success, msg = self._install_pip_package(pkg)
                    self.progress_signal.emit(msg)
                    if not success:
                        self.progress_signal.emit(f"Warning: Failed to install {pkg}")

            # Install CUDA/TensorRT packages
            if cuda_to_install:
                self.progress_signal.emit("\n=== Installing CUDA/TensorRT packages ===\n")
                self.progress_signal.emit("Note: TensorRT requires NVIDIA GPU and CUDA toolkit installed.\n")
                for pkg in cuda_to_install:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    self.status_signal.emit(f"Installing {pkg_name}...", current_step, total_steps)
                    self.progress_signal.emit(f"Installing {pkg}...")
                    success, msg = self._install_pip_package(pkg)
                    self.progress_signal.emit(msg)
                    if not success:
                        self.progress_signal.emit(f"Warning: Failed to install {pkg}")
                        self.progress_signal.emit("  You may need to install TensorRT manually from NVIDIA.")

            # Install DirectML packages
            if dml_to_install:
                self.progress_signal.emit("\n=== Installing DirectML packages ===\n")
                self.progress_signal.emit("Note: DirectML works with any DirectX 12 GPU (AMD, Intel, NVIDIA).\n")
                for pkg in dml_to_install:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    self.status_signal.emit(f"Installing {pkg_name}...", current_step, total_steps)
                    self.progress_signal.emit(f"Installing {pkg}...")
                    success, msg = self._install_pip_package(pkg)
                    self.progress_signal.emit(msg)
                    if not success:
                        self.progress_signal.emit(f"Warning: Failed to install {pkg}")

            # Install PyTorch packages (spandrel, safetensors, etc.)
            if pytorch_to_install:
                self.progress_signal.emit("\n=== Installing PyTorch support packages ===\n")
                self.progress_signal.emit("Note: PyTorch itself must be installed separately with your CUDA version.\n")
                self.progress_signal.emit("      Visit https://pytorch.org/get-started/locally/ for instructions.\n")
                for pkg in pytorch_to_install:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    pkg_name = pkg.split(">=")[0].split("==")[0]
                    self.status_signal.emit(f"Installing {pkg_name}...", current_step, total_steps)
                    self.progress_signal.emit(f"Installing {pkg}...")
                    success, msg = self._install_pip_package(pkg)
                    self.progress_signal.emit(msg)
                    if not success:
                        self.progress_signal.emit(f"Warning: Failed to install {pkg}")

            # Install external tools
            if tools_to_install:
                self.progress_signal.emit("\n=== Installing external tools ===\n")
                TOOLS_DIR.mkdir(parents=True, exist_ok=True)

                paths_to_add: list[Path] = []

                for name, url, extract_type, path_subdir in tools_to_install:
                    if self._cancelled:
                        self.finished_signal.emit(False, "Installation cancelled.")
                        return
                    current_step += 1
                    dest_path = TOOLS_DIR / path_subdir if path_subdir else TOOLS_DIR
                    self.status_signal.emit(f"Downloading {name}...", current_step, total_steps)
                    self.progress_signal.emit(f"Downloading {name}...")
                    self.progress_signal.emit(f"  URL: {url}")
                    self.progress_signal.emit(f"  Destination: {dest_path}")
                    success = self._download_and_extract_tool(name, url, extract_type)
                    if success:
                        self.progress_signal.emit(f"  {name} installed successfully")
                        # Track the path to add
                        if path_subdir:
                            paths_to_add.append(TOOLS_DIR / path_subdir)
                        else:
                            paths_to_add.append(TOOLS_DIR)
                    else:
                        self.progress_signal.emit(f"  Warning: Failed to install {name}")

                # Add all tool directories to PATH
                if paths_to_add:
                    self.progress_signal.emit("\nAdding tools directories to PATH...")
                    for path_dir in paths_to_add:
                        if path_dir.exists():
                            success = self._add_to_path(path_dir)
                            if success:
                                self.progress_signal.emit(f"  Added {path_dir} to user PATH")
                            else:
                                self.progress_signal.emit(f"  Warning: Could not add {path_dir} to PATH")
                        else:
                            self.progress_signal.emit(f"  Warning: Path does not exist: {path_dir}")

            self.progress_signal.emit("\n=== Installation complete ===\n")
            self.finished_signal.emit(True, "All dependencies installed successfully!")

        except Exception as e:
            self.finished_signal.emit(False, f"Installation failed: {e}")

    def _install_pip_package(self, package: str) -> tuple[bool, str]:
        """Install a Python package via pip."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return True, f"  {package} installed successfully"
            else:
                return False, f"  Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"  Timeout installing {package}"
        except Exception as e:
            return False, f"  Error: {e}"

    def _download_file(self, url: str, name: str) -> Path | None:
        """Download a file to temp directory."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = Request(url, headers=headers)

            # Get filename from URL
            filename = url.split("/")[-1]
            temp_path = Path(tempfile.gettempdir()) / f"trt_dep_{filename}"

            with urlopen(req, timeout=120) as response:
                with open(temp_path, "wb") as f:
                    # Download in chunks
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

            return temp_path
        except Exception as e:
            self.progress_signal.emit(f"  Download error: {e}")
            return None

    def _download_and_extract_tool(
        self, name: str, url: str, extract_type: str
    ) -> bool:
        """Download and extract an external tool to TOOLS_DIR."""
        temp_file = self._download_file(url, name)
        if not temp_file:
            return False

        try:
            with zipfile.ZipFile(temp_file, "r") as zf:
                # Extract directly to TOOLS_DIR ("extract here" style)
                zf.extractall(TOOLS_DIR)
            return True
        except Exception as e:
            self.progress_signal.emit(f"  Extract error: {e}")
            return False
        finally:
            try:
                temp_file.unlink()
            except Exception:
                pass

    def _add_to_path(self, directory: Path) -> bool:
        """Add a directory to the user's PATH environment variable."""
        try:
            import winreg

            # Open the Environment key
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
                0,
                winreg.KEY_READ | winreg.KEY_WRITE,
            ) as key:
                try:
                    current_path, _ = winreg.QueryValueEx(key, "Path")
                except FileNotFoundError:
                    current_path = ""

                dir_str = str(directory)
                if dir_str.lower() not in current_path.lower():
                    if current_path:
                        new_path = f"{current_path};{dir_str}"
                    else:
                        new_path = dir_str
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)

                    # Broadcast environment change
                    try:
                        import ctypes
                        HWND_BROADCAST = 0xFFFF
                        WM_SETTINGCHANGE = 0x001A
                        SMTO_ABORTIFHUNG = 0x0002
                        ctypes.windll.user32.SendMessageTimeoutW(
                            HWND_BROADCAST,
                            WM_SETTINGCHANGE,
                            0,
                            "Environment",
                            SMTO_ABORTIFHUNG,
                            5000,
                            None,
                        )
                    except Exception:
                        pass

                    return True
                else:
                    self.progress_signal.emit(f"  {dir_str} already in PATH")
                    return True
        except Exception as e:
            self.progress_signal.emit(f"  PATH error: {e}")
            return False


class DependenciesWindow(QDialog):
    """
    Dependencies installation window.

    Provides UI for installing all required dependencies:
    - Python packages via pip
    - TensorRT and CUDA packages
    - External tools (ffmpeg, gifski, avifenc, pngquant, pingo)
    """

    def __init__(self, parent: QWidgetType | None = None):
        super().__init__(parent)
        self.setWindowTitle("Dependencies")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)

        self._worker: InstallWorker | None = None
        self._animation_timer: QTimer | None = None
        self._animation_dots = 0

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Info section
        info_group = QGroupBox("Required Dependencies")
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)

        info_text = QLabel(
            "This will install all dependencies required for TensorRT Image Upscaler:\n\n"
            "<b>Python packages:</b> PySide6, numpy, Pillow, opencv-python, numba, fpng-py\n\n"
            "<b>CUDA/TensorRT:</b> cuda-python, pycuda, tensorrt (requires NVIDIA GPU + CUDA toolkit)\n\n"
            "<b>DirectML:</b> onnxruntime-directml (alternative backend for AMD/Intel GPUs)\n\n"
            "<b>PyTorch:</b> spandrel, safetensors (for .pth/.safetensors model support)\n\n"
            "<b>External tools:</b> ffmpeg, gifski, avifenc (animated output), "
            "pngquant, pingo (PNG optimization)\n\n"
            "<b>Prerequisites:</b>\n"
            "- TensorRT: NVIDIA GPU with CUDA Toolkit 12.x\n"
            "- DirectML: Any DirectX 12 compatible GPU\n"
            "- PyTorch: Install torch separately from https://pytorch.org/"
        )
        info_text.setTextFormat(Qt.RichText)
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)

        # Progress section
        progress_group = QGroupBox("Installation Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self._status_label = QLabel("Ready to install")
        progress_layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        # Log output
        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMinimumHeight(200)
        self._log_output.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        progress_layout.addWidget(self._log_output)

        layout.addWidget(progress_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self._install_button = QPushButton("Install All Dependencies")
        self._install_button.setMinimumHeight(40)
        self._install_button.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self._install_button, 2)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        btn_layout.addWidget(self._cancel_button, 1)

        self._close_button = QPushButton("Close")
        btn_layout.addWidget(self._close_button, 1)

        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._install_button.clicked.connect(self._on_install_clicked)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        self._close_button.clicked.connect(self.close)

    def _on_install_clicked(self) -> None:
        """Start the installation process."""
        self._install_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._close_button.setEnabled(False)
        self._log_output.clear()
        self._progress_bar.setValue(0)

        # Start animation
        self._animation_dots = 0
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._update_animation)
        self._animation_timer.start(500)

        # Start worker
        self._worker = InstallWorker(
            install_pip=True,
            install_tensorrt=True,
            install_directml=True,
            install_pytorch=True,
            install_pytorch_optional=True,
            install_tools=True,
        )
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.status_signal.connect(self._on_status)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        """Cancel the installation."""
        if self._worker:
            self._worker.cancel()
            self._status_label.setText("Cancelling...")

    def _update_animation(self) -> None:
        """Update the loading animation."""
        self._animation_dots = (self._animation_dots + 1) % 4
        dots = "." * self._animation_dots
        current = self._status_label.text().rstrip(".")
        if current:
            base = current.split("...")[0].split("..")[0].split(".")[0]
            self._status_label.setText(f"{base}{dots}")

    def _on_progress(self, message: str) -> None:
        """Handle progress log message."""
        self._log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self._log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_status(self, status: str, current: int, total: int) -> None:
        """Handle status update."""
        self._status_label.setText(status)
        if total > 0:
            percent = int(100 * current / total)
            self._progress_bar.setValue(percent)

    def _on_finished(self, success: bool, message: str) -> None:
        """Handle installation completion."""
        if self._animation_timer:
            self._animation_timer.stop()
            self._animation_timer = None

        self._install_button.setEnabled(True)
        self._cancel_button.setEnabled(False)
        self._close_button.setEnabled(True)
        self._progress_bar.setValue(100 if success else 0)
        self._status_label.setText(message)

        if success:
            QMessageBox.information(
                self,
                "Installation Complete",
                "All dependencies have been installed successfully!\n\n"
                "Note: You may need to restart the application for changes to take effect.",
            )
        else:
            QMessageBox.warning(
                self,
                "Installation Issue",
                f"{message}\n\nCheck the log output for details.",
            )

        self._worker = None

    def closeEvent(self, event) -> None:
        """Handle window close."""
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Installation in Progress",
                "Installation is still in progress. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(5000)

        if self._animation_timer:
            self._animation_timer.stop()

        super().closeEvent(event)
