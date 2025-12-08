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
├── engine.py      # TensorRT engine building and inference
├── upscaler.py    # Tiling, blending, alpha handling
├── animated.py    # GIF/WebP/APNG processing
├── gui.py         # PySide6 interface
├── cli.py         # Command-line interface
├── dialogs.py     # Settings, comparison, crop dialogs
└── theme.py       # Dark/light themes
```

## License

MIT
