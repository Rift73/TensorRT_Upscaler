#!/usr/bin/env python
"""
Quick launcher for TensorRT Upscaler GUI.
Handles missing dependencies gracefully.
"""

import sys
from pathlib import Path

# Add src directory to path for running without installation
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import PIL
    except ImportError:
        missing.append("Pillow")
    
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        missing.append("PySide6")
    
    return missing


def show_error_dialog(message: str):
    """Show error dialog using available GUI toolkit."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "TensorRT Upscaler - Error", message)
        sys.exit(1)
    except ImportError:
        # Fallback to console
        print(f"ERROR: {message}")
        input("Press Enter to exit...")
        sys.exit(1)


def main():
    # Check for missing core dependencies
    missing = check_dependencies()
    if missing:
        msg = (
            f"Missing required dependencies:\n\n"
            f"  {', '.join(missing)}\n\n"
            f"Please run install.bat or setup_venv.bat first,\n"
            f"or install dependencies manually:\n\n"
            f"  pip install {' '.join(missing)}"
        )
        show_error_dialog(msg)
        return
    
    # Try to import and run GUI
    try:
        from tensorrt_upscaler.main_window import main as gui_main
        gui_main()
    except ImportError as e:
        show_error_dialog(f"Failed to import GUI module:\n\n{e}")
    except Exception as e:
        show_error_dialog(f"Error starting application:\n\n{e}")


if __name__ == "__main__":
    main()
