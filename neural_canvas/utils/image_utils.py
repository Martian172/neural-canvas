"""
neural_canvas.utils.image_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utility functions for image manipulation used throughout Neural Canvas.

Functions:
    resize_with_aspect_ratio  – resize while preserving aspect ratio
    create_grid               – combine images into a tiled grid
    add_watermark             – overlay a text watermark
    convert_format            – convert image to a different file format
    calculate_image_stats     – compute brightness, contrast, and entropy
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ---------------------------------------------------------------------------
# resize_with_aspect_ratio
# ---------------------------------------------------------------------------


def resize_with_aspect_ratio(
    image: Image.Image,
    width: Optional[int] = None,
    height: Optional[int] = None,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """Resize an image to fit within *width* x *height* while preserving
    its original aspect ratio.

    Exactly one of *width* or *height* may be ``None``; in that case the
    missing dimension is computed automatically.  If both are provided the
    image is resized so that neither dimension exceeds the specified value
    (letterbox behaviour).

    Args:
        image: Source PIL image.
        width: Target width in pixels, or ``None`` to compute from height.
        height: Target height in pixels, or ``None`` to compute from width.
        resample: Pillow resampling filter (default: ``Image.LANCZOS``).

    Returns:
        Resized PIL image.

    Raises:
        ValueError: If neither *width* nor *height* is specified.

    Example::

        resized = resize_with_aspect_ratio(image, width=1024)
        resized = resize_with_aspect_ratio(image, width=800, height=600)
    """
    if width is None and height is None:
        raise ValueError("At least one of 'width' or 'height' must be specified.")

    orig_w, orig_h = image.size

    if width is None:
        # Compute width from height
        ratio = height / orig_h  # type: ignore[operator]
        new_w = max(1, int(orig_w * ratio))
        new_h = int(height)
    elif height is None:
        # Compute height from width
        ratio = width / orig_w
        new_w = int(width)
        new_h = max(1, int(orig_h * ratio))
    else:
        # Letterbox: fit inside width x height
        ratio = min(width / orig_w, height / orig_h)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))

    return image.resize((new_w, new_h), resample=resample)


# ---------------------------------------------------------------------------
# create_grid
# ---------------------------------------------------------------------------


def create_grid(
    images: Sequence[Image.Image],
    cols: Optional[int] = None,
    cell_size: Optional[Tuple[int, int]] = None,
    padding: int = 8,
    background_color: Tuple[int, int, int] = (30, 30, 30),
    border_color: Optional[Tuple[int, int, int]] = None,
    border_width: int = 2,
) -> Image.Image:
    """Tile *images* into a rectangular grid.

    Args:
        images: Sequence of PIL images to arrange.
        cols: Number of columns.  Defaults to ``ceil(sqrt(n))``.
        cell_size: ``(width, height)`` for each cell.  If ``None``, uses the
                   size of the first image.
        padding: Padding in pixels between cells.
        background_color: RGB background fill colour.
        border_color: Optional RGB border colour drawn around each cell.
        border_width: Width of the cell border in pixels.

    Returns:
        A new PIL image containing the grid.

    Raises:
        ValueError: If *images* is empty.

    Example::

        grid = create_grid(generated_images, cols=3)
        grid.save("gallery.png")
    """
    if not images:
        raise ValueError("'images' sequence must not be empty.")

    n = len(images)
    if cols is None:
        cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    if cell_size is None:
        cell_w, cell_h = images[0].size
    else:
        cell_w, cell_h = cell_size

    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), color=background_color)
    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)

        # Resize cell if necessary
        cell_img = img.convert("RGB")
        if cell_img.size != (cell_w, cell_h):
            cell_img = cell_img.resize((cell_w, cell_h), Image.LANCZOS)

        grid.paste(cell_img, (x, y))

        if border_color and border_width > 0:
            draw.rectangle(
                [x, y, x + cell_w - 1, y + cell_h - 1],
                outline=border_color,
                width=border_width,
            )

    return grid


# ---------------------------------------------------------------------------
# add_watermark
# ---------------------------------------------------------------------------


def add_watermark(
    image: Image.Image,
    text: str = "Neural Canvas",
    position: str = "bottom-right",
    opacity: float = 0.5,
    font_size: int = 20,
    color: Tuple[int, int, int] = (255, 255, 255),
    padding: int = 10,
) -> Image.Image:
    """Overlay a semi-transparent text watermark on *image*.

    Args:
        image: Source PIL image (any mode).
        text: Watermark text to render.
        position: Placement of the watermark.  One of:
                  ``"top-left"``, ``"top-right"``, ``"bottom-left"``,
                  ``"bottom-right"``, ``"center"``.
        opacity: Watermark opacity in ``[0.0, 1.0]``.
        font_size: Font size in points (falls back to default if TrueType
                   is unavailable).
        color: Text colour as an ``(R, G, B)`` tuple.
        padding: Padding from the image edge in pixels.

    Returns:
        A new PIL image with the watermark applied.

    Example::

        watermarked = add_watermark(img, "© 2024 Neural Canvas", opacity=0.4)
    """
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Attempt to load a TrueType font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w, img_h = img.size
    positions = {
        "top-left": (padding, padding),
        "top-right": (img_w - text_w - padding, padding),
        "bottom-left": (padding, img_h - text_h - padding),
        "bottom-right": (img_w - text_w - padding, img_h - text_h - padding),
        "center": ((img_w - text_w) // 2, (img_h - text_h) // 2),
    }
    pos = positions.get(position, positions["bottom-right"])

    alpha = int(opacity * 255)
    draw.text(pos, text, font=font, fill=(*color, alpha))

    watermarked = Image.alpha_composite(img, overlay)
    return watermarked.convert("RGB")


# ---------------------------------------------------------------------------
# convert_format
# ---------------------------------------------------------------------------


def convert_format(
    image: Union[Image.Image, str, Path],
    output_path: Union[str, Path],
    fmt: Optional[str] = None,
    quality: int = 92,
) -> str:
    """Convert an image to a different file format and save it.

    Args:
        image: A PIL ``Image`` object, or a path string/``Path`` to an image.
        output_path: Destination file path.  The extension determines the
                     format if *fmt* is ``None``.
        fmt: Explicit format string (e.g. ``"PNG"``, ``"JPEG"``).  When
             ``None``, inferred from *output_path*'s extension.
        quality: JPEG/WEBP save quality in ``[1, 100]``.  Ignored for PNG.

    Returns:
        The absolute path of the saved file as a string.

    Raises:
        FileNotFoundError: If *image* is a path that does not exist.
        ValueError: If the format cannot be determined.

    Example::

        path = convert_format("photo.jpg", "photo.webp")
        path = convert_format(pil_img, "out.png", fmt="PNG")
    """
    if isinstance(image, (str, Path)):
        src = Path(image)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")
        img = Image.open(src)
    else:
        img = image

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        ext = dst.suffix.lstrip(".").upper()
        if ext == "JPG":
            ext = "JPEG"
        fmt = ext

    if not fmt:
        raise ValueError(
            f"Cannot determine format from path '{dst}'. Specify fmt= explicitly."
        )

    save_kwargs: Dict[str, object] = {}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

    img.save(str(dst), format=fmt, **save_kwargs)  # type: ignore[arg-type]
    return str(dst.resolve())


# ---------------------------------------------------------------------------
# calculate_image_stats
# ---------------------------------------------------------------------------


def calculate_image_stats(image: Image.Image) -> Dict[str, float]:
    """Compute a set of perceptual statistics for *image*.

    Statistics computed:
    - **mean_brightness**: Average pixel luminance, normalised to ``[0, 1]``.
    - **contrast**: Standard deviation of per-pixel luminance.
    - **entropy**: Shannon entropy of the grayscale histogram (bits).
    - **colorfulness**: A measure of colour diversity based on Hasler & Süsstrunk (2003).
    - **mean_r / mean_g / mean_b**: Per-channel mean values (0–255).
    - **sharpness**: Variance of the Laplacian (proxy for edge detail).

    Args:
        image: PIL image to analyse.

    Returns:
        A dictionary mapping stat names to float values.

    Example::

        stats = calculate_image_stats(img)
        print(f"Brightness: {stats['mean_brightness']:.3f}")
        print(f"Entropy:    {stats['entropy']:.3f} bits")
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)

    # Luminance (BT.601)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mean_brightness = float(lum.mean() / 255.0)
    contrast = float(lum.std() / 255.0)

    # Entropy from histogram
    hist, _ = np.histogram(lum, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # remove zero bins
    entropy = float(-np.sum(hist * np.log2(hist)) * (255 / 256))

    # Colorfulness (Hasler & Süsstrunk 2003)
    rg = arr[:, :, 0] - arr[:, :, 1]
    yb = 0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]
    sigma_rg = float(rg.std())
    sigma_yb = float(yb.std())
    mu_rg = float(rg.mean())
    mu_yb = float(yb.mean())
    colorfulness = float(
        math.sqrt(sigma_rg**2 + sigma_yb**2)
        + 0.3 * math.sqrt(mu_rg**2 + mu_yb**2)
    )

    # Per-channel means
    mean_r = float(arr[:, :, 0].mean())
    mean_g = float(arr[:, :, 1].mean())
    mean_b = float(arr[:, :, 2].mean())

    # Sharpness: Laplacian variance on greyscale
    from PIL import ImageFilter as _IF

    grey = img_rgb.convert("L")
    lap = grey.filter(_IF.Kernel(size=(3, 3), kernel=(-1, -1, -1, -1, 8, -1, -1, -1, -1), scale=1))
    sharpness = float(np.array(lap, dtype=np.float32).var())

    return {
        "mean_brightness": round(mean_brightness, 4),
        "contrast": round(contrast, 4),
        "entropy": round(entropy, 4),
        "colorfulness": round(colorfulness, 4),
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "sharpness": round(sharpness, 2),
    }
