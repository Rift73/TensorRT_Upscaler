"""
Configuration management using Windows Registry.
Saves and loads all user settings.
"""

import sys
from dataclasses import dataclass, field, fields
from typing import Optional

if sys.platform == "win32":
    import winreg

REGISTRY_KEY = r"Software\TensorRTUpscalerV2"


@dataclass
class Config:
    """Application configuration stored in Windows Registry."""

    # Model settings
    onnx_path: str = ""
    tile_width: int = 512
    tile_height: int = 512
    tile_overlap: int = 8
    model_scale: int = 4
    use_fp16: bool = False
    use_bf16: bool = True
    use_tf32: bool = False

    # Output filename options (#9-13)
    save_next_to_input: bool = False
    same_dir_suffix: str = "_upscaled"
    manga_folder_mode: bool = False
    append_model_suffix: bool = False
    overwrite_existing: bool = True

    # Processing control (#14)
    upscale_enabled: bool = True

    # Sharpening (#18-19)
    sharpen_enabled: bool = False
    sharpen_value: float = 0.5

    # Custom resolution (#20-25)
    custom_res_enabled: bool = False
    custom_res_keep_aspect: bool = True
    custom_res_mode: str = "width"  # width, height, 2x
    custom_res_width: int = 1920
    custom_res_height: int = 1080
    custom_res_kernel: str = "lanczos"  # lanczos, hermite

    # Secondary output (#26-30)
    secondary_enabled: bool = False
    secondary_mode: str = "width"
    secondary_width: int = 1920
    secondary_height: int = 1080
    secondary_kernel: str = "lanczos"

    # Pre-scale (#31-35)
    prescale_enabled: bool = False
    prescale_mode: str = "width"
    prescale_width: int = 1920
    prescale_height: int = 1080
    prescale_kernel: str = "lanczos"

    # Animated output (#36-46)
    animated_format: str = "gif"  # gif, webp, avif, apng
    gif_quality: int = 90
    gif_fast_mode: bool = False
    webp_lossless: bool = False
    webp_quality: int = 90
    webp_preset: str = "none"
    avif_lossless: bool = False
    avif_color_quality: int = 80
    avif_alpha_quality: int = 90
    avif_speed: int = 6
    apng_prediction: str = "mixed"

    # PNG optimization (#47-49)
    png_quantize_enabled: bool = False
    png_quantize_colors: int = 256
    png_optimize_enabled: bool = False

    # UI settings (#73, 77-78)
    theme: str = "dark"
    last_input_path: str = ""
    last_onnx_directory: str = ""
    last_output_path: str = ""

    # QoL: Skip existing files
    skip_existing: bool = False

    # QoL: Recent folders (JSON list of paths)
    recent_input_folders: str = "[]"
    recent_output_folders: str = "[]"
    recent_onnx_paths: str = "[]"
    max_recent_items: int = 10

    # QoL: Processing presets (JSON dict of preset_name -> config values)
    presets: str = "{}"
    last_preset: str = ""

    # QoL: Conditional processing
    conditional_enabled: bool = False
    conditional_min_width: int = 0
    conditional_min_height: int = 0
    conditional_max_width: int = 0
    conditional_max_height: int = 0

    # QoL: Aspect ratio filter
    aspect_filter_enabled: bool = False
    aspect_filter_mode: str = "any"  # any, landscape, portrait, square
    aspect_filter_min_ratio: float = 0.0
    aspect_filter_max_ratio: float = 0.0

    # QoL: Notifications
    notify_on_complete: bool = True
    sound_on_complete: bool = False
    sound_file_path: str = ""

    # QoL: Window behavior
    always_on_top: bool = False
    minimize_to_tray: bool = False
    start_minimized: bool = False

    # QoL: Auto-shutdown
    auto_shutdown_enabled: bool = False
    auto_shutdown_mode: str = "sleep"  # sleep, hibernate, shutdown

    # QoL: Logging
    log_processing: bool = False
    log_directory: str = ""

    # QoL: Open output folder on completion
    open_output_on_complete: bool = False

    # QoL: Keyboard shortcuts enabled
    shortcuts_enabled: bool = True

    # QoL: Metadata preservation
    preserve_metadata: bool = True

    # QoL: Resume batch (checkpoint file path)
    checkpoint_file: str = ""
    auto_checkpoint: bool = True

    # QoL: Multi-model queue (JSON list of ONNX paths)
    model_queue: str = "[]"
    model_queue_enabled: bool = False

    # QoL: A/B model comparison
    comparison_model_a: str = ""
    comparison_model_b: str = ""

    def save(self) -> None:
        """Save configuration to Windows Registry."""
        if sys.platform != "win32":
            return

        try:
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY)

            for f in fields(self):
                value = getattr(self, f.name)
                if isinstance(value, bool):
                    winreg.SetValueEx(key, f.name, 0, winreg.REG_DWORD, int(value))
                elif isinstance(value, int):
                    winreg.SetValueEx(key, f.name, 0, winreg.REG_DWORD, value)
                elif isinstance(value, float):
                    winreg.SetValueEx(key, f.name, 0, winreg.REG_SZ, str(value))
                else:
                    winreg.SetValueEx(key, f.name, 0, winreg.REG_SZ, str(value))

            winreg.CloseKey(key)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def load(self) -> None:
        """Load configuration from Windows Registry."""
        if sys.platform != "win32":
            return

        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY)

            for f in fields(self):
                try:
                    value, reg_type = winreg.QueryValueEx(key, f.name)

                    if f.type == bool:
                        setattr(self, f.name, bool(value))
                    elif f.type == int:
                        setattr(self, f.name, int(value))
                    elif f.type == float:
                        setattr(self, f.name, float(value))
                    else:
                        setattr(self, f.name, str(value))
                except FileNotFoundError:
                    pass  # Use default value

            winreg.CloseKey(key)
        except FileNotFoundError:
            pass  # No config yet, use defaults
        except Exception as e:
            print(f"Failed to load config: {e}")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.load()
    return _config


def save_config() -> None:
    """Save the global configuration."""
    if _config is not None:
        _config.save()
