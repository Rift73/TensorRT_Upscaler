"""
Fast image I/O using fpng, OpenCV, and threading.
Provides significant speedup over PIL for loading and saving images.

Optimizations:
- fpng for 12-19x faster PNG encoding than alternatives
- OpenCV for 2-3x faster image loading than PIL
- Thread pool with reusable workers
- Prefetching for overlapped I/O
- ICC profile and metadata preservation
"""

import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Callable, Dict, Any
from pathlib import Path

import numpy as np
import fpng_py as fpng

# Try to import OpenCV (much faster than PIL for loading)
CV2_AVAILABLE = False
cv2 = None
try:
    import cv2 as _cv2
    cv2 = _cv2
    CV2_AVAILABLE = True
    # Set OpenCV to use all threads for parallel operations
    cv2.setNumThreads(0)  # 0 = use all available cores
except ImportError:
    pass

# Fallback to PIL for loading
from PIL import Image
from PIL import PngImagePlugin


def load_image_fast(path: str) -> Tuple[np.ndarray, bool]:
    """
    Load image as float32 numpy array (H, W, C) in RGB order.

    Args:
        path: Path to image file

    Returns:
        (array, has_alpha): Image array normalized to [0, 1] and alpha flag
    """
    if CV2_AVAILABLE:
        # OpenCV is 2-3x faster than PIL
        # Use imdecode with np.fromfile to handle Unicode paths on Windows
        # cv2.imread() doesn't handle non-ASCII paths properly on Windows
        try:
            # Read file as bytes (handles Unicode paths)
            img_bytes = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None

        if img is None:
            # Fallback to PIL for problematic files
            return _load_image_pil(path)

        has_alpha = img.ndim == 3 and img.shape[2] == 4

        if has_alpha:
            # BGRA -> RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.ndim == 3:
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            # Grayscale -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            has_alpha = False

        # Convert to float32 [0, 1]
        img = img.astype(np.float32) / 255.0

        return img, has_alpha
    else:
        return _load_image_pil(path)


def _load_image_pil(path: str) -> Tuple[np.ndarray, bool]:
    """PIL fallback for loading images."""
    pil_img = Image.open(path)
    has_alpha = pil_img.mode == 'RGBA'

    if has_alpha:
        pil_img = pil_img.convert('RGBA')
    else:
        pil_img = pil_img.convert('RGB')

    img = np.array(pil_img).astype(np.float32) / 255.0
    return img, has_alpha


def extract_image_metadata(path: str) -> Dict[str, Any]:
    """
    Extract ICC profile and other metadata from source image.

    Returns dict with:
        - icc_profile: bytes or None
        - exif: bytes or None
        - pnginfo: PngInfo or None (for PNG text chunks)
    """
    metadata = {
        'icc_profile': None,
        'exif': None,
        'pnginfo': None,
    }

    try:
        with Image.open(path) as img:
            # ICC Profile (color profile)
            metadata['icc_profile'] = img.info.get('icc_profile')

            # EXIF data
            metadata['exif'] = img.info.get('exif')

            # PNG text chunks (includes things like software, description, etc.)
            if img.format == 'PNG':
                pnginfo = PngImagePlugin.PngInfo()
                for key, value in img.info.items():
                    if isinstance(key, str) and isinstance(value, str):
                        # Skip binary data keys
                        if key not in ('icc_profile', 'exif', 'transparency'):
                            try:
                                pnginfo.add_text(key, value)
                            except Exception:
                                pass
                metadata['pnginfo'] = pnginfo

    except Exception:
        pass

    return metadata


def save_image_fast(
    img: np.ndarray,
    path: str,
    has_alpha: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save numpy array as PNG image.

    Uses fpng for speed when no metadata, PIL when metadata needs to be preserved.

    Args:
        img: Image array in [0, 1] range, shape (H, W, C)
        path: Output path
        has_alpha: Whether image has alpha channel
        metadata: Optional dict with 'icc_profile', 'exif', 'pnginfo' from extract_image_metadata()
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    # Convert to uint8 (handle NaN/infinity from inference)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)

    # Check if we have metadata to preserve
    has_metadata = metadata and (metadata.get('icc_profile') or metadata.get('exif'))

    if has_metadata:
        # Use PIL to preserve ICC profile and EXIF (slower but preserves color)
        if has_alpha and img.shape[2] == 4:
            pil_img = Image.fromarray(img, mode='RGBA')
        else:
            pil_img = Image.fromarray(img, mode='RGB')

        save_kwargs = {}

        # ICC profile (color profile - critical for color accuracy)
        if metadata.get('icc_profile'):
            save_kwargs['icc_profile'] = metadata['icc_profile']

        # EXIF data
        if metadata.get('exif'):
            save_kwargs['exif'] = metadata['exif']

        # PNG text chunks
        if metadata.get('pnginfo'):
            save_kwargs['pnginfo'] = metadata['pnginfo']

        pil_img.save(path, **save_kwargs)
    else:
        # Use fpng for maximum speed (no metadata to preserve)
        # Ensure contiguous array for fpng
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        height, width = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1

        # fpng requires RGB (3) or RGBA (4) - our arrays are already in RGB order
        img_bytes = img.tobytes()
        # flags=0 for fastest encoding
        # Use encode_to_memory + manual file write to handle Unicode paths on Windows
        # fpng_encode_image_to_file() doesn't handle non-ASCII paths properly
        png_data = fpng.fpng_encode_image_to_memory(img_bytes, width, height, channels, 0)
        if png_data is None:
            raise RuntimeError("fpng encoding failed")

        # Write using Path which handles Unicode properly
        Path(path).write_bytes(png_data)


def pil_to_array(pil_img: Image.Image) -> Tuple[np.ndarray, bool]:
    """Convert PIL Image to numpy array."""
    has_alpha = pil_img.mode == 'RGBA'
    if has_alpha:
        pil_img = pil_img.convert('RGBA')
    else:
        pil_img = pil_img.convert('RGB')

    img = np.array(pil_img).astype(np.float32) / 255.0
    return img, has_alpha


def array_to_pil(img: np.ndarray, has_alpha: bool = False) -> Image.Image:
    """Convert numpy array to PIL Image."""
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)

    if has_alpha and img.shape[2] == 4:
        return Image.fromarray(img, mode='RGBA')
    else:
        return Image.fromarray(img, mode='RGB')


class PipelinedProcessor:
    """
    Pipelined image processor that overlaps I/O with processing.

    Pipeline stages:
    1. Load next image (background thread)
    2. Process current image (main thread / GPU)
    3. Save previous result (background thread)

    This hides I/O latency and can improve throughput by 20-50%.
    """

    def __init__(
        self,
        process_fn: Callable[[np.ndarray], np.ndarray],
        max_pending_saves: int = 2,
    ):
        """
        Initialize pipelined processor.

        Args:
            process_fn: Function that processes image array (e.g., upscaling)
            max_pending_saves: Max number of saves queued
        """
        self.process_fn = process_fn

        # Thread pool for I/O
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Queue for pending saves
        self.save_queue = queue.Queue(maxsize=max_pending_saves)
        self.save_thread = None
        self.save_error = None

        # Prefetch state
        self.prefetch_future = None
        self.prefetch_path = None

    def start(self):
        """Start the save worker thread."""
        self.save_error = None
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

    def stop(self):
        """Stop the save worker and wait for pending saves."""
        if self.save_thread:
            self.save_queue.put(None)  # Sentinel to stop
            self.save_thread.join(timeout=30)
            self.save_thread = None

        if self.prefetch_future:
            self.prefetch_future.cancel()
            self.prefetch_future = None

    def _save_worker(self):
        """Background thread that saves images."""
        while True:
            item = self.save_queue.get()
            if item is None:
                break

            img, path, has_alpha = item
            try:
                save_image_fast(img, path, has_alpha)
            except Exception as e:
                self.save_error = e
                print(f"Save error: {e}")

    def prefetch(self, path: str):
        """Start loading an image in the background."""
        self.prefetch_path = path
        self.prefetch_future = self.executor.submit(load_image_fast, path)

    def get_prefetched(self) -> Tuple[np.ndarray, bool]:
        """Get the prefetched image, blocking if needed."""
        if self.prefetch_future:
            result = self.prefetch_future.result()
            self.prefetch_future = None
            return result
        raise RuntimeError("No prefetch pending")

    def queue_save(self, img: np.ndarray, path: str, has_alpha: bool):
        """Queue an image to be saved in the background."""
        self.save_queue.put((img.copy(), path, has_alpha))

    def process_batch(
        self,
        input_paths: list,
        output_paths: list,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Process a batch of images with pipelined I/O.

        Args:
            input_paths: List of input file paths
            output_paths: List of output file paths
            progress_callback: Called with (current, total) after each file

        Returns:
            True if all successful
        """
        if len(input_paths) != len(output_paths):
            raise ValueError("Input and output path counts must match")

        if not input_paths:
            return True

        self.start()

        try:
            total = len(input_paths)

            # Start prefetch for first image
            self.prefetch(input_paths[0])

            for i in range(total):
                # Start prefetch for next image
                if i + 1 < total:
                    next_path = input_paths[i + 1]
                    next_future = self.executor.submit(load_image_fast, next_path)
                else:
                    next_future = None

                # Get current image (from prefetch or load directly)
                if self.prefetch_future:
                    img, has_alpha = self.prefetch_future.result()
                    self.prefetch_future = None
                else:
                    img, has_alpha = load_image_fast(input_paths[i])

                # Process (GPU inference happens here)
                result = self.process_fn(img)

                # Queue save (background)
                self.queue_save(result, output_paths[i], has_alpha)

                # Setup prefetch for next iteration
                self.prefetch_future = next_future

                if progress_callback:
                    progress_callback(i + 1, total)

                # Check for save errors
                if self.save_error:
                    raise self.save_error

            return True

        finally:
            self.stop()


def encode_png_to_bytes(img: np.ndarray, has_alpha: bool = False) -> bytes:
    """
    Encode numpy array to PNG bytes using fpng (fast).

    Args:
        img: Image array in [0, 1] range, shape (H, W, C)
        has_alpha: Whether image has alpha channel

    Returns:
        PNG-encoded bytes
    """
    # Convert to uint8 (handle NaN/infinity from inference)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)

    # Ensure contiguous array for fpng
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)

    height, width = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    img_bytes = img.tobytes()
    png_data = fpng.fpng_encode_image_to_memory(img_bytes, width, height, channels, 0)

    if png_data is None:
        raise RuntimeError("fpng encoding failed")

    return png_data


class AsyncImageSaver:
    """
    Async image saver that queues images for background saving.

    Copies the array and queues it for background PNG encoding and I/O.
    """

    def __init__(self, max_queue: int = 4):
        self.queue = queue.Queue(maxsize=max_queue)
        self.thread = None
        self.error = None
        self._stop = False

    def start(self):
        """Start the saver thread."""
        self._stop = False
        self.error = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self, wait: bool = True):
        """Stop the saver thread."""
        self._stop = True
        self.queue.put(None)
        if wait and self.thread:
            self.thread.join(timeout=60)
        self.thread = None

    def _worker(self):
        while not self._stop:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self.queue.task_done()
                break

            img, path, has_alpha, metadata = item
            try:
                save_image_fast(img, path, has_alpha, metadata)
            except Exception as e:
                self.error = e
                print(f"Async save error: {e}")
            finally:
                self.queue.task_done()

    def save(self, img: np.ndarray, path: str, has_alpha: bool = False, metadata: Optional[Dict[str, Any]] = None):
        """Queue image for background saving (copies array)."""
        self.queue.put((img.copy(), path, has_alpha, metadata))

    def wait_pending(self):
        """Wait for all pending saves to complete."""
        self.queue.join()


class AsyncImageLoader:
    """
    Prefetches the next image while the current one is being processed.
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.future = None
        self.path = None

    def prefetch(self, path: str):
        """Start loading an image in the background."""
        self.path = path
        self.future = self.executor.submit(load_image_fast, path)

    def get(self, path: Optional[str] = None) -> Tuple[np.ndarray, bool]:
        """
        Get an image. Uses prefetch if path matches, otherwise loads directly.
        """
        if path is None:
            path = self.path

        if self.future and self.path == path:
            result = self.future.result()
            self.future = None
            self.path = None
            return result
        else:
            return load_image_fast(path)

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False)
