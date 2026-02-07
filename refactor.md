# TensorRT Upscaler v2 - Refactoring Plan

## Overview
Refactoring the codebase to reduce God classes, eliminate code duplication, and add missing abstractions.
Base path: `C:\Users\PC\Desktop\Repository\tensorrt_upscaler_v2\src\tensorrt_upscaler\`

---

## Task 1: Remove Duplicated Code
**Status: DONE**
**Priority: HIGH**

### 1a: `IMAGE_EXTENSIONS` duplicated in `cli.py` and `utils.py`
- Removed duplicate from `cli.py`, now imports from `utils.py`
- Also changed `cli.py` to import `is_animated` from `utils` instead of `animated`
- Status: DONE

### 1b: `is_animated()` duplicated in `animated.py` and `utils.py`
- Removed duplicate from `animated.py`, now imports from `utils.py` and re-exports
- Canonical version lives in `utils.py`
- Status: DONE

### 1c: `get_frame_count()` duplicated in `animated.py` and `utils.py`
- Removed duplicate from `animated.py`, now imports from `utils.py` and re-exports
- Canonical version lives in `utils.py`
- Status: DONE

### 1d: `_detect_model_scale()` duplicated in `engine.py` and `dml_engine.py`
- Extracted to `utils.py` as `detect_model_scale(onnx_path, default=4) -> int`
- Both `engine.py` and `dml_engine.py` now import and call the shared function
- Status: DONE

### Files modified:
- `utils.py` — added `detect_model_scale()` function
- `cli.py` — removed `IMAGE_EXTENSIONS`, imports from `utils`; changed `is_animated` import source
- `animated.py` — removed `is_animated()` and `get_frame_count()`, imports from `utils`
- `engine.py` — replaced `_detect_model_scale()` body with call to `detect_model_scale()`
- `dml_engine.py` — replaced `_detect_model_scale()` body with call to `detect_model_scale()`

---

## Task 2: Create Engine Protocol/ABC
**Status: DONE**
**Priority: HIGH**

- Created `engine_base.py` with `InferenceEngine` Protocol (runtime_checkable)
- Interface: `model_scale: int`, `infer(np.ndarray) -> np.ndarray`, `infer_nchw(np.ndarray) -> np.ndarray`
- Added type annotation `self.engine: InferenceEngine` to `upscaler.py`
- Not enforcing inheritance on engine classes (Protocol is structural typing — no changes needed to engine classes)

### Files created:
- `engine_base.py`

### Files modified:
- `upscaler.py` — imports `InferenceEngine`, type-annotates `self.engine`

---

## Task 3: Split `main_window.py` (2418 -> 1927 lines)
**Status: DONE**
**Priority: HIGH**

Used **mixin pattern** — MainWindow now inherits from 4 mixin classes that provide extracted functionality.
Mixins operate on `self` (the MainWindow instance), preserving direct attribute access.

```python
class MainWindow(
    TrayManagerMixin,
    ShortcutsMixin,
    WatchFolderMixin,
    ProgressTrackerMixin,
    QMainWindow,
):
```

### 3a: `gui/watch_folder.py` — WatchFolderMixin (~130 lines) — DONE
- Extracted 6 methods: `_toggle_watch_folder`, `_start_watch_folder`, `_stop_watch_folder`, `_on_watch_folder_changed`, `_process_watch_pending`, `_is_file_ready`

### 3b: `gui/tray_manager.py` — TrayManagerMixin (~65 lines) — DONE
- Extracted 5 methods: `_setup_tray_icon`, `_show_from_tray`, `_on_tray_activated`, `changeEvent`, `_hide_to_tray`

### 3c: `gui/shortcuts.py` — ShortcutsMixin (~95 lines) — DONE
- Extracted 5 methods: `_setup_keyboard_shortcuts`, `_shortcut_start`, `_shortcut_cancel`, `_refresh_input`, `_toggle_always_on_top`

### 3d: `gui/progress_tracker.py` — ProgressTrackerMixin (~165 lines) — DONE
- Extracted 7 methods: `_update_time_display`, `_on_progress`, `_update_smooth_progress`, `_on_file_progress`, `_on_file_done`, `_on_file_skipped`, `_on_checkpoint_updated`

### 3e: `create_upscaler()` factory in `gui/workers.py` — DONE
- Extracted `create_upscaler(cfg, onnx_path)` factory function
- Both `UpscaleWorker` and `ClipboardWorker` now call the shared factory instead of duplicating 24 lines of `ImageUpscaler(...)` initialization

### Files created:
- `gui/watch_folder.py`
- `gui/tray_manager.py`
- `gui/shortcuts.py`
- `gui/progress_tracker.py`

### Files modified:
- `main_window.py` — imports mixins, inherits from them, removed extracted methods (2418 -> 1927 lines)
- `gui/__init__.py` — exports new mixin classes
- `gui/workers.py` — added `create_upscaler()` factory, both workers use it

---

## Task 4: Medium Priority Cleanup
**Status: CLOSED**
**Priority: MEDIUM**

### 4a: `config.py` — Group Config fields into nested dataclasses
- Status: DEFERRED — high risk of breaking many callers across 10+ files

### 4b: `upscaler.py` — Create `UpscaleOptions` dataclass for `upscale_file`
- Status: SKIPPED — `upscale_file()` is dead code (defined but never called/exported). A dataclass wrapper would just duplicate `ImageUpscaler.__init__`'s signature with no real benefit.

### Additional cleanup done:
- Updated `README.md` project structure to include `engine_base.py` and new gui mixin files

---

## Task 5: Additional Medium Priority Items
**Status: DONE**
**Priority: MEDIUM**

### 5a: `animated.py` — Extract encoding functions into `animated_encoders.py`
- Created `animated_encoders.py` with: `encode_gif`, `encode_gif_gifski`, `encode_webp`, `encode_apng`, `encode_avif`, `_rgb_to_yuv444`, `_build_y4m_stream`, `deduplicate_frames`
- `animated.py` now imports and re-exports from `animated_encoders.py` for backwards compat
- `animated.py`: 664 -> 285 lines; `animated_encoders.py`: ~380 lines
- Status: DONE

### 5b: `animated.py` — Break down `upscale_animated` method (~150 lines)
- Extracted `_process_frame()` — single frame pipeline: pre-scale, upscale, custom res, sharpen, convert to PIL
- Extracted `_encode_frames()` — format detection and encoding dispatch (static method)
- `upscale_animated()` now focuses on frame iteration, deduplication, and orchestration
- Status: DONE

### 5c: `fast_io.py` — Share save logic between `PipelinedProcessor` and `AsyncImageSaver`
- Both classes already delegate to `save_image_fast()` — the shared function IS the shared logic
- The only difference: `AsyncImageSaver` passes metadata, `PipelinedProcessor` doesn't
- Additional abstraction would be over-engineering for 2-line worker methods
- Status: SKIPPED (not needed — already properly factored)

### 5d: `gui/workers.py` — Break down `UpscaleWorker.run()` (~250 lines)
- Extracted `_process_animated_image()` — handles animated image format detection and AnimatedUpscaler delegation
- Extracted `_process_static_image()` — handles pre-scale, upscale, custom res, and sharpening pipeline
- `run()` now focuses on file iteration, skip checks, I/O orchestration, and save/optimization
- Status: DONE

### 5e: `sharpening.py` — Extract common alpha-handling logic
- Added 4 helpers: `_split_alpha_pil`, `_merge_alpha_pil`, `_split_alpha_array`, `_merge_alpha_array`
- Simplified `cas_sharpen_pil`, `cas_sharpen_array`, `adaptive_sharpen_pil`, `adaptive_sharpen_array` to use helpers
- Removed ~40 lines of duplicated alpha split/merge code across 4 functions
- Status: DONE

### 5f: `dependencies_window.py` — Break down `InstallWorker.run()`
- Extracted `_check_packages()` — shared package availability check (replaces 5 near-identical loops)
- Extracted `_install_package_group()` — shared pip install loop with header/notes/fail_note
- Extracted `_install_external_tools()` — tool download/extract/PATH setup
- `run()` now reads as a simple pipeline of check + install calls
- Status: DONE

### 5g: `web_extractor.py` — Break down `_extract_images_from_page` (~150 lines)
- Extracted `_extract_img_elements()` — `<img>` elements with lazy-load attribute handling
- Extracted `_extract_source_elements()` — `<picture>/<source>` elements
- Extracted `_extract_background_images()` — CSS background-image extraction
- Extracted `_get_element_dimensions()` — multi-fallback dimension detection (static method)
- Moved `_LAZY_ATTRS` to class-level constant
- `_extract_images_from_page()` now a clean 6-line orchestrator
- Status: DONE

---

## Task 6: Low Priority Items
**Status: PENDING**
**Priority: LOW**

### 6a: `theme.py` — Move stylesheet definitions to external `.qss` files
- Large stylesheet strings embedded in methods
- Could use external QSS files loaded at runtime
- Mostly cosmetic, current approach works fine
- Status: PENDING

### 6b: `theme.py` — Use theme dictionary pattern instead of method-per-theme
- Current approach has separate method for each theme
- Could use a data-driven dictionary mapping theme names to stylesheet parameters
- Status: PENDING

### 6c: `resize.py` — Kernel factory for similar resize kernels
- Hermite/Lanczos/Catmull-Rom kernels have repetitive patterns
- Could use a factory to generate similar kernels
- Low priority — Numba-optimized code is best left as-is for performance clarity
- Status: PENDING

### 6d: `upscaler.py` — Extract `TileManager` class
- Tile computation, blending weights, and overlap logic could be a separate class
- Would reduce `ImageUpscaler` complexity
- Status: PENDING

### 6e: `upscaler.py` — Remove dead code `upscale_file()`
- Convenience function defined but never called or exported
- Can be safely deleted
- Status: PENDING

### 6f: `pytorch_engine.py` — Consolidate GPU capability checks
- `_check_bf16_support` and `_check_tf32_support` have similar patterns
- Could be consolidated into a single capability checker
- Status: PENDING

---

## Completion Log
- Task 1a DONE — Removed `IMAGE_EXTENSIONS` from `cli.py`, imports from `utils.py`
- Task 1b DONE — Removed `is_animated()` from `animated.py`, imports from `utils.py`
- Task 1c DONE — Removed `get_frame_count()` from `animated.py`, imports from `utils.py`
- Task 1d DONE — Extracted `detect_model_scale()` to `utils.py`, updated `engine.py` and `dml_engine.py`
- Task 2 DONE — Created `engine_base.py` with `InferenceEngine` Protocol, type-annotated `upscaler.py`
- Task 3a DONE — Created `gui/watch_folder.py` with `WatchFolderMixin`
- Task 3b DONE — Created `gui/tray_manager.py` with `TrayManagerMixin`
- Task 3c DONE — Created `gui/shortcuts.py` with `ShortcutsMixin`
- Task 3d DONE — Created `gui/progress_tracker.py` with `ProgressTrackerMixin`
- Task 3e DONE — Created `create_upscaler()` factory in `gui/workers.py`
- Task 3 final DONE — Updated `main_window.py` to inherit mixins, removed extracted methods, cleaned imports
- Task 5a DONE — Created `animated_encoders.py`, extracted 8 encoding functions from `animated.py` (664->285 lines)
- Task 5b DONE — Extracted `_process_frame()` and `_encode_frames()` from `upscale_animated()`
- Task 5c SKIPPED — Save logic already shared via `save_image_fast()`
- Task 5d DONE — Extracted `_process_animated_image()` and `_process_static_image()` from `UpscaleWorker.run()`
- Task 5e DONE — Added 4 alpha split/merge helpers to `sharpening.py`, simplified 4 functions
- Task 5f DONE — Extracted `_check_packages()`, `_install_package_group()`, `_install_external_tools()` from `InstallWorker.run()`
- Task 5g DONE — Extracted 4 methods from `_extract_images_from_page()` in `web_extractor.py`

## Summary of All Changes

### Files created (7):
1. `engine_base.py` — InferenceEngine Protocol
2. `gui/watch_folder.py` — WatchFolderMixin
3. `gui/tray_manager.py` — TrayManagerMixin
4. `gui/shortcuts.py` — ShortcutsMixin
5. `gui/progress_tracker.py` — ProgressTrackerMixin
6. `animated_encoders.py` — Encoding functions for GIF/WebP/APNG/AVIF

### Files modified (13):
1. `utils.py` — added `detect_model_scale()`
2. `cli.py` — imports from `utils` instead of duplicating
3. `animated.py` — imports from `utils` and `animated_encoders`, extracted methods (664->285 lines)
4. `engine.py` — uses shared `detect_model_scale()`
5. `dml_engine.py` — uses shared `detect_model_scale()`
6. `upscaler.py` — type-annotated with `InferenceEngine` Protocol
7. `main_window.py` — uses mixin pattern (2418 -> 1927 lines)
8. `gui/__init__.py` — exports new mixins
9. `gui/workers.py` — uses `create_upscaler()` factory, extracted `_process_static_image()` and `_process_animated_image()`
10. `sharpening.py` — added alpha split/merge helpers, simplified 4 public functions
11. `dependencies_window.py` — extracted `_check_packages()`, `_install_package_group()`, `_install_external_tools()`
12. `web_extractor.py` — extracted 4 methods from `_extract_images_from_page()`
