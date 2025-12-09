"""
Web image extractor using Playwright for JavaScript rendering.

Extracts images from web pages, supporting:
- Full JavaScript rendering (for lazy-loaded images)
- Browser cookies for authenticated pages
- Multiple browser cookie sources (Chrome, Firefox, Edge, etc.)
"""

import os
import tempfile
import hashlib
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

# Check for Playwright
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

# Check for browser_cookie3
BROWSER_COOKIE3_AVAILABLE = False
try:
    import browser_cookie3
    BROWSER_COOKIE3_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ExtractedImage:
    """Represents an extracted image from a web page."""
    url: str
    width: int
    height: int
    alt: str
    local_path: Optional[str] = None  # Set after download


class WebImageExtractor:
    """
    Extracts images from web pages using Playwright.

    Features:
    - Full JavaScript rendering for lazy-loaded images
    - Browser cookie support for authenticated pages
    - Filters small images (icons, buttons, etc.)
    """

    # Minimum image dimensions to extract (filters out icons/buttons)
    MIN_WIDTH = 100
    MIN_HEIGHT = 100

    def __init__(
        self,
        browser_cookie_source: str = "none",
        wait_time: float = 3.0,
        headless: bool = True,
    ):
        """
        Initialize the extractor.

        Args:
            browser_cookie_source: Browser to get cookies from (none, chrome, firefox, edge, chromium, brave)
            wait_time: Seconds to wait for JavaScript to load images
            headless: Run browser in headless mode
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is not installed. Install it with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )

        self.browser_cookie_source = browser_cookie_source.lower()
        self.wait_time = wait_time
        self.headless = headless
        self._cookies = []

        # Load cookies if browser specified
        if self.browser_cookie_source != "none":
            self._load_browser_cookies()

    def _load_browser_cookies(self):
        """Load cookies from the specified browser."""
        if not BROWSER_COOKIE3_AVAILABLE:
            print("browser_cookie3 not available, skipping cookie loading")
            return

        try:
            cookie_jar = None

            if self.browser_cookie_source == "chrome":
                cookie_jar = browser_cookie3.chrome()
            elif self.browser_cookie_source == "firefox":
                cookie_jar = browser_cookie3.firefox()
            elif self.browser_cookie_source == "edge":
                cookie_jar = browser_cookie3.edge()
            elif self.browser_cookie_source == "chromium":
                cookie_jar = browser_cookie3.chromium()
            elif self.browser_cookie_source == "brave":
                cookie_jar = browser_cookie3.brave()

            if cookie_jar:
                # Convert to Playwright cookie format
                for cookie in cookie_jar:
                    self._cookies.append({
                        "name": cookie.name,
                        "value": cookie.value,
                        "domain": cookie.domain,
                        "path": cookie.path,
                        "secure": cookie.secure,
                    })
                print(f"Loaded {len(self._cookies)} cookies from {self.browser_cookie_source}")

        except Exception as e:
            print(f"Failed to load cookies from {self.browser_cookie_source}: {e}")

    def extract_images(
        self,
        url: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[ExtractedImage]:
        """
        Extract images from a web page.

        Args:
            url: URL of the web page
            progress_callback: Optional callback for progress updates

        Returns:
            List of ExtractedImage objects
        """
        images = []

        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            print(msg)

        log(f"Opening page: {url}")

        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            # Add cookies if available
            if self._cookies:
                try:
                    # Filter cookies for this domain
                    parsed = urlparse(url)
                    domain = parsed.netloc

                    domain_cookies = []
                    for cookie in self._cookies:
                        cookie_domain = cookie.get("domain", "")
                        # Match domain (including subdomains)
                        if cookie_domain.lstrip(".") in domain or domain.endswith(cookie_domain.lstrip(".")):
                            # Playwright needs url for cookie
                            cookie_copy = cookie.copy()
                            cookie_copy["url"] = url
                            domain_cookies.append(cookie_copy)

                    if domain_cookies:
                        context.add_cookies(domain_cookies)
                        log(f"Added {len(domain_cookies)} cookies for {domain}")
                except Exception as e:
                    log(f"Warning: Failed to add cookies: {e}")

            page = context.new_page()

            try:
                # Navigate to page
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Wait for JavaScript to load
                log(f"Waiting {self.wait_time}s for JavaScript...")
                page.wait_for_timeout(int(self.wait_time * 1000))

                # Scroll down to trigger lazy loading
                log("Scrolling to load lazy images...")
                self._scroll_page(page)

                # Wait a bit more after scrolling
                page.wait_for_timeout(1000)

                # Extract all images
                log("Extracting images...")
                images = self._extract_images_from_page(page, url)

                log(f"Found {len(images)} images")

            except Exception as e:
                log(f"Error: {e}")
                raise
            finally:
                browser.close()

        return images

    def _scroll_page(self, page: "Page"):
        """Scroll the page to trigger lazy loading."""
        # Get page height
        height = page.evaluate("document.body.scrollHeight")
        viewport_height = page.evaluate("window.innerHeight")

        # Scroll in chunks
        current = 0
        while current < height:
            page.evaluate(f"window.scrollTo(0, {current})")
            page.wait_for_timeout(200)
            current += viewport_height
            # Re-check height (might have grown due to lazy loading)
            height = page.evaluate("document.body.scrollHeight")

        # Scroll back to top
        page.evaluate("window.scrollTo(0, 0)")

    def _extract_images_from_page(self, page: "Page", base_url: str) -> List[ExtractedImage]:
        """Extract image information from the page."""
        images = []
        seen_urls = set()

        # Get all img elements
        img_elements = page.query_selector_all("img")

        for img in img_elements:
            try:
                # Get src (prefer data-src for lazy loaded images)
                src = img.get_attribute("data-src") or img.get_attribute("src")
                if not src:
                    continue

                # Skip data: URLs and blob: URLs
                if src.startswith(("data:", "blob:")):
                    continue

                # Make absolute URL
                src = urljoin(base_url, src)

                # Skip duplicates
                if src in seen_urls:
                    continue
                seen_urls.add(src)

                # Get dimensions
                box = img.bounding_box()
                if box:
                    width = int(box["width"])
                    height = int(box["height"])
                else:
                    # Try natural dimensions
                    width = img.evaluate("el => el.naturalWidth") or 0
                    height = img.evaluate("el => el.naturalHeight") or 0

                # Filter small images
                if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                    continue

                # Get alt text
                alt = img.get_attribute("alt") or ""

                images.append(ExtractedImage(
                    url=src,
                    width=width,
                    height=height,
                    alt=alt[:100],  # Truncate long alt text
                ))

            except Exception as e:
                # Skip problematic images
                continue

        # Also check for background images in divs (common for manga readers)
        bg_elements = page.query_selector_all("[style*='background-image']")
        for el in bg_elements:
            try:
                style = el.get_attribute("style") or ""
                # Extract URL from background-image: url(...)
                import re
                match = re.search(r"background-image:\s*url\(['\"]?([^'\")\s]+)['\"]?\)", style)
                if match:
                    src = match.group(1)
                    if src.startswith(("data:", "blob:")):
                        continue

                    src = urljoin(base_url, src)
                    if src in seen_urls:
                        continue
                    seen_urls.add(src)

                    box = el.bounding_box()
                    if box:
                        width = int(box["width"])
                        height = int(box["height"])

                        if width >= self.MIN_WIDTH and height >= self.MIN_HEIGHT:
                            images.append(ExtractedImage(
                                url=src,
                                width=width,
                                height=height,
                                alt="",
                            ))
            except Exception:
                continue

        # Sort by size (largest first)
        images.sort(key=lambda x: x.width * x.height, reverse=True)

        return images

    def download_image(
        self,
        image: ExtractedImage,
        dest_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download an image to a local file.

        Args:
            image: ExtractedImage to download
            dest_dir: Directory to save to (uses temp dir if not specified)

        Returns:
            Local file path, or None on failure
        """
        import urllib.request

        try:
            # Parse URL for filename
            parsed = urlparse(image.url)
            filename = os.path.basename(parsed.path)

            # Generate filename if empty or invalid
            if not filename or '.' not in filename:
                # Use hash of URL as filename
                url_hash = hashlib.md5(image.url.encode()).hexdigest()[:8]
                filename = f"image_{url_hash}.jpg"

            # Determine destination
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
            else:
                fd, dest_path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1] or ".jpg")
                os.close(fd)

            # Download
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = urllib.request.Request(image.url, headers=headers)

            with urllib.request.urlopen(req, timeout=30) as response:
                with open(dest_path, "wb") as f:
                    f.write(response.read())

            image.local_path = dest_path
            return dest_path

        except Exception as e:
            print(f"Failed to download {image.url}: {e}")
            return None


def is_webpage_url(url: str) -> bool:
    """
    Check if a URL is likely a webpage (not a direct image link).

    Args:
        url: URL to check

    Returns:
        True if the URL appears to be a webpage
    """
    if not url:
        return False

    # Must be http/https
    if not url.lower().startswith(("http://", "https://")):
        return False

    # Check file extension
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Direct image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".avif")
    if path.endswith(image_extensions):
        return False

    # Likely a webpage
    return True
