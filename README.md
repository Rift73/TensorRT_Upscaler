# TensorRT Upscaler

GPU-accelerated image upscaler with direct TensorRT inference.

This is a direct upgrade over the prototype.

![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)
![TensorRT 10.0+](https://img.shields.io/badge/TensorRT-10.0+-green.svg)
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

**Core**
- Direct TensorRT inference
- Asynchronous image processing for maximum parallel process, throughput and minimum overhead
- Alpha channel preservation
- Animated format support (GIF, WebP, APNG)

**GUI**
- Drag & drop files or folders
- Before/after split comparison with draggable slider
- Crop preview for testing upscale quality
- Watch folder mode for automated processing*
- Multi-model queue for batch processing with different models*
- Resume interrupted batches*
- Presets, themes, keyboard shortcuts
- System tray integration

* not yet tested

## Requirements

- Windows 10/11
- NVIDIA GPU (RTX 20-series or newer recommended)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
- [TensorRT 10.x](https://developer.nvidia.com/tensorrt)
- Python 3.12+

## Installation

```bash
# Clone and setup
git clone https://github.com/your-username/tensorrt-upscaler-v2.git
cd tensorrt-upscaler-v2

# Create venv and install
python -m venv venv
venv\Scripts\activate
pip install -e .

# Optional: performance extras (opencv, numba, fpng)
pip install -e ".[full]"
```

Or use the provided batch files:
- `setup_venv.bat` - Create venv and install dependencies
- `run_gui.bat` - Launch GUI
- `run_cli.bat` - Launch CLI

## Usage

### GUI
```bash
python run.py
# or
tensorrt-upscaler-gui
```

### CLI
```bash
# Single image
tensorrt-upscaler input.png -m model.onnx

# Batch with options
tensorrt-upscaler ./images -m model.onnx -o ./output --recursive --bf16

# All options
tensorrt-upscaler input.png -m model.onnx \
    --tile-width 512 --tile-height 512 \
    --overlap 16 --bf16 --suffix "_4x"
```

### Python API
```python
from tensorrt_upscaler import ImageUpscaler
from PIL import Image

upscaler = ImageUpscaler("model.onnx", tile_size=(512, 512), bf16=True)
result = upscaler.upscale_image(Image.open("input.png"))
result.save("output.png")
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter | Start processing |
| Escape | Cancel |
| Ctrl+I | Open files |
| Ctrl+Shift+I | Open folder |
| Ctrl+E | Open output folder |
| Ctrl+W | Toggle watch mode |
| Ctrl+L | Show log |
| Z | Toggle 100% zoom |
| Delete | Remove selected |
| F5 | Refresh list |

## Project Structure

```
src/tensorrt_upscaler/
├── main_window.py        # PySide6 main GUI window
├── cli.py                # Command-line interface
├── engine.py             # TensorRT engine building and inference
├── engine_base.py        # InferenceEngine Protocol (shared interface)
├── dml_engine.py         # DirectML/ONNX Runtime backend
├── pytorch_engine.py     # PyTorch/Spandrel backend
├── upscaler.py           # Tiled upscaling with blending
├── animated.py           # GIF/WebP/APNG processing
├── config.py             # Configuration (Windows Registry)
├── fast_io.py            # Fast image I/O (fpng, OpenCV)
├── resize.py             # Resize kernels (Hermite, Lanczos, Catmull-Rom)
├── sharpening.py         # CAS/Adaptive sharpening
├── theme.py              # Dark/Holo themes
├── utils.py              # Utility functions
├── web_extractor.py      # Web image extraction (Playwright)
├── dependencies_window.py # Dependencies installer GUI
├── dialogs/              # Dialog windows (modular)
│   ├── resolution.py     # Custom resolution settings
│   ├── pytorch_options.py # PyTorch options
│   ├── tensorrt_options.py # TensorRT options
│   ├── web_image_dialog.py # Web image extraction
│   ├── animated_output.py # Animated format settings
│   ├── png_options.py    # PNG optimization settings
│   ├── settings.py       # General settings & presets
│   ├── notifications.py  # Notifications & window behavior
│   ├── log.py            # Processing log
│   ├── model_queue.py    # Multi-model queue
│   ├── comparison.py     # Before/after split comparison
│   ├── crop_preview.py   # Crop region preview
│   └── sharpen.py        # Sharpen settings
└── gui/                  # GUI components
    ├── widgets.py        # DropLineEdit, ThumbnailLabel
    ├── workers.py        # UpscaleWorker, ClipboardWorker
    ├── tray_manager.py   # System tray mixin
    ├── shortcuts.py      # Keyboard shortcuts mixin
    ├── watch_folder.py   # Watch folder mode mixin
    └── progress_tracker.py # Progress bar & timing mixin
```

## License

MIT
