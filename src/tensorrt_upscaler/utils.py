"""
Utility functions for the upscaler.
"""

import os
import re
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.request import urlretrieve, Request, urlopen
from urllib.parse import urlparse

from PIL import Image


# Supported image formats
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
ANIMATED_EXTENSIONS = {".gif", ".webp", ".png"}  # PNG can be APNG


def natural_sort_key(s):
    """
    Key function for natural sorting (file2 before file10).

    Feature #72
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def format_time_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    seconds = max(0, int(seconds + 0.5))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def has_alpha(image: Image.Image) -> bool:
    """
    Check if an image has actual transparency (not just an alpha channel).

    Feature #64-65
    """
    if image.mode == 'RGBA':
        # Check if any pixel has alpha < 255
        alpha = image.getchannel('A')
        return alpha.getextrema()[0] < 255

    elif image.mode == 'LA':
        alpha = image.getchannel('A')
        return alpha.getextrema()[0] < 255

    elif image.mode == 'P':
        # Palette mode - check for transparency
        if 'transparency' in image.info:
            return True
        # Check if palette has alpha
        if image.palette and hasattr(image.palette, 'mode') and 'A' in image.palette.mode:
            return True

    elif image.mode == 'PA':
        return True

    return False


def is_animated(image_path: str) -> bool:
    """Check if an image file is animated (GIF, APNG, animated WebP)."""
    try:
        with Image.open(image_path) as img:
            try:
                img.seek(1)
                return True
            except EOFError:
                return False
    except Exception:
        return False


def get_frame_count(image_path: str) -> int:
    """Get number of frames in an animated image."""
    try:
        with Image.open(image_path) as img:
            count = 0
            try:
                while True:
                    count += 1
                    img.seek(count)
            except EOFError:
                pass
            return count
    except Exception:
        return 1


def generate_output_path(
    input_path: str,
    output_dir: str,
    suffix: str = "_upscaled",
    save_next_to_input: bool = False,
    manga_folder_mode: bool = False,
    append_model_suffix: bool = False,
    model_name: str = "",
    overwrite: bool = True,
    input_root: Optional[str] = None,
    output_format: str = "png",
) -> str:
    """
    Generate output file path based on settings.

    Output path logic:
    1. save_next_to_input=True: Save next to input with suffix in filename
    2. manga_folder_mode=True (folder dropped):
       - input_root="C:/Folder", image="C:/Folder/sub/img.png"
       - output="C:/Folder_upscaled/sub/img.png" (suffix applied to root folder)
    3. Otherwise (single file, save_next_to_input=False):
       - output="<input_parent>/upscaled/img_upscaled.png"

    Features #9-13
    """
    input_path = Path(input_path)
    stem = input_path.stem
    ext = f".{output_format}" if output_format else input_path.suffix

    # Determine output directory and filename
    if save_next_to_input:
        # Save next to input with suffix in filename
        out_dir = input_path.parent
        filename = stem + suffix
        if append_model_suffix and model_name:
            filename += f"_{model_name}"
        filename += ext
    elif manga_folder_mode and input_root:
        # Manga folder mode: RootFolder_upscaled/relative/path/filename.ext
        # Suffix goes on the root folder, NOT the filename
        input_root_path = Path(input_root)
        if input_root_path.is_file():
            input_root_path = input_root_path.parent

        try:
            rel_path = input_path.parent.relative_to(input_root_path)
        except ValueError:
            rel_path = Path("")

        # Add suffix to the root folder name (e.g., "Executables" -> "Executables_upscaled")
        root_parent = input_root_path.parent
        root_name = input_root_path.name + suffix
        out_dir = root_parent / root_name / rel_path

        # Filename without suffix (since folder has suffix)
        filename = stem
        if append_model_suffix and model_name:
            filename += f"_{model_name}"
        filename += ext
    else:
        # Single file mode: save to "upscaled" subfolder
        out_dir = input_path.parent / "upscaled"
        filename = stem + suffix
        if append_model_suffix and model_name:
            filename += f"_{model_name}"
        filename += ext

    # Create directory
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / filename

    # Handle overwrite
    if not overwrite and output_path.exists():
        counter = 2
        base_stem = stem + suffix if not manga_folder_mode else stem
        while True:
            new_filename = f"{base_stem}_{counter:03d}{ext}"
            output_path = out_dir / new_filename
            if not output_path.exists():
                break
            counter += 1

    return str(output_path)


def download_url(url: str, dest_dir: Optional[str] = None) -> Optional[str]:
    """
    Download an image from URL to a temporary file.

    Features #54-55
    """
    try:
        # Parse URL
        parsed = urlparse(url)

        # Extract filename from URL
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            filename = "downloaded_image.png"

        # Determine extension
        ext = Path(filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            ext = ".png"
            filename = Path(filename).stem + ext

        # Create temp file
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)
        else:
            fd, dest_path = tempfile.mkstemp(suffix=ext)
            os.close(fd)

        # Download
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = Request(url, headers=headers)

        with urlopen(req, timeout=30) as response:
            with open(dest_path, 'wb') as f:
                f.write(response.read())

        return dest_path

    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def extract_url_from_text(text: str) -> Optional[str]:
    """
    Extract image URL from text (supports Discord CDN, direct URLs).

    Feature #55
    """
    # Direct URL pattern
    url_pattern = r'https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff?)(?:\?[^\s<>"\']*)?'

    match = re.search(url_pattern, text, re.IGNORECASE)
    if match:
        return match.group(0)

    # Discord CDN pattern (may have query params)
    discord_pattern = r'https?://(?:cdn|media)\.discord(?:app)?\.com/[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s<>"\']*)?'
    match = re.search(discord_pattern, text, re.IGNORECASE)
    if match:
        return match.group(0)

    return None


def run_pngquant(
    input_path: str,
    colors: int = 256,
    output_path: Optional[str] = None,
) -> bool:
    """
    Run pngquant for PNG quantization.

    Feature #47-48
    """
    try:
        if output_path is None:
            output_path = input_path

        cmd = [
            "pngquant",
            "--force",
            "--output", output_path,
            str(colors),
            input_path,
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0

    except FileNotFoundError:
        print("pngquant not found in PATH")
        return False
    except Exception as e:
        print(f"pngquant failed: {e}")
        return False


def run_pingo(input_path: str) -> bool:
    """
    Run pingo for lossless PNG optimization.

    Feature #49

    Note: pingo doesn't handle Unicode paths well, so we copy to a temp file
    with an ASCII name, optimize it, then copy back.
    """
    import tempfile
    import shutil
    from pathlib import Path

    try:
        input_file = Path(input_path)

        # Check if path contains non-ASCII characters
        try:
            input_path.encode('ascii')
            has_unicode = False
        except UnicodeEncodeError:
            has_unicode = True

        if has_unicode:
            # Use temp file workaround for Unicode paths
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / "temp_pingo.png"
                shutil.copy2(input_file, temp_file)

                cmd = ["pingo", "-lossless", "-s4", str(temp_file)]
                result = subprocess.run(cmd, capture_output=True, timeout=120)

                if result.returncode == 0:
                    # Copy optimized file back
                    shutil.copy2(temp_file, input_file)
                    return True
                return False
        else:
            # Direct path for ASCII-only paths
            cmd = ["pingo", "-lossless", "-s4", input_path]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            return result.returncode == 0

    except FileNotFoundError:
        print("pingo not found in PATH")
        return False
    except Exception as e:
        print(f"pingo failed: {e}")
        return False


def optimize_png(
    input_path: str,
    quantize: bool = False,
    quantize_colors: int = 256,
    optimize: bool = False,
) -> bool:
    """
    Apply PNG optimization pipeline.

    Features #47-49
    """
    success = True

    if quantize:
        success = run_pngquant(input_path, quantize_colors) and success

    if optimize:
        success = run_pingo(input_path) and success

    return success


def collect_files(
    paths: List[str],
    recursive: bool = True,
) -> List[str]:
    """
    Collect all image files from given paths (files or directories).
    Returns naturally sorted list.
    """
    files = []

    for path in paths:
        path = Path(path)

        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(str(path))

        elif path.is_dir():
            if recursive:
                for root, _, filenames in os.walk(path):
                    for f in filenames:
                        if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                            files.append(os.path.join(root, f))
            else:
                for f in path.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        files.append(str(f))

    # Natural sort
    files.sort(key=natural_sort_key)

    return files


def should_skip_image(
    image_path: str,
    output_path: str,
    skip_existing: bool = False,
    conditional_enabled: bool = False,
    min_width: int = 0,
    min_height: int = 0,
    max_width: int = 0,
    max_height: int = 0,
    aspect_filter_enabled: bool = False,
    aspect_mode: str = "any",
    aspect_min_ratio: float = 0.0,
    aspect_max_ratio: float = 0.0,
) -> tuple:
    """
    Check if an image should be skipped based on filter settings.

    Returns:
        (should_skip: bool, reason: str or None)
    """
    # Skip if output already exists
    if skip_existing and os.path.exists(output_path):
        return True, "output exists"

    # Check image dimensions for conditional processing and aspect ratio
    if conditional_enabled or aspect_filter_enabled:
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Conditional size filtering
                if conditional_enabled:
                    if min_width > 0 and width < min_width:
                        return True, f"width {width} < min {min_width}"
                    if min_height > 0 and height < min_height:
                        return True, f"height {height} < min {min_height}"
                    if max_width > 0 and width > max_width:
                        return True, f"width {width} > max {max_width}"
                    if max_height > 0 and height > max_height:
                        return True, f"height {height} > max {max_height}"

                # Aspect ratio filtering
                if aspect_filter_enabled:
                    aspect_ratio = width / height if height > 0 else 1.0

                    if aspect_mode == "landscape" and aspect_ratio <= 1.0:
                        return True, "not landscape"
                    elif aspect_mode == "portrait" and aspect_ratio >= 1.0:
                        return True, "not portrait"
                    elif aspect_mode == "square" and not (0.95 <= aspect_ratio <= 1.05):
                        return True, "not square"
                    elif aspect_mode == "custom":
                        if aspect_min_ratio > 0 and aspect_ratio < aspect_min_ratio:
                            return True, f"aspect {aspect_ratio:.2f} < min {aspect_min_ratio}"
                        if aspect_max_ratio > 0 and aspect_ratio > aspect_max_ratio:
                            return True, f"aspect {aspect_ratio:.2f} > max {aspect_max_ratio}"

        except Exception as e:
            # If we can't read the image, don't skip it - let the processing handle the error
            pass

    return False, None


def copy_metadata(source_path: str, dest_path: str) -> bool:
    """
    Copy EXIF/metadata from source image to destination image.
    Works with PNG and JPEG files.

    Returns True if metadata was copied, False otherwise.
    """
    try:
        # Open source to get EXIF
        with Image.open(source_path) as src_img:
            exif_data = src_img.info.get('exif')
            if not exif_data:
                return False

            # Open destination and save with EXIF
            with Image.open(dest_path) as dest_img:
                # Save back with EXIF data
                dest_img.save(dest_path, exif=exif_data)
                return True

    except Exception:
        # Silently fail - metadata preservation is optional
        return False


def get_image_metadata(path: str) -> dict:
    """
    Get metadata from an image file.

    Returns dict with available metadata.
    """
    metadata = {}
    try:
        from PIL.ExifTags import TAGS

        with Image.open(path) as img:
            # Basic info
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            metadata['size'] = img.size

            # EXIF data
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata[tag] = value

    except Exception:
        pass

    return metadata
