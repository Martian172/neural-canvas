"""
examples/basic_usage.py
~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates the core features of Neural Canvas:
  1. Applying style presets to a single image
  2. Using individual filters directly
  3. Extracting a colour palette
  4. Creating a gallery grid
  5. Computing image statistics
  6. Generating colour harmonies

Run with:
    python examples/basic_usage.py
"""

import sys
import time
from pathlib import Path

# Make the package importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np

from neural_canvas import StyleTransferPipeline
from neural_canvas.core.filters import (
    GlitchFilter,
    NeonFilter,
    SketchFilter,
    VintageFilter,
    WatercolorFilter,
)
from neural_canvas.core.palette import ColorHarmony, PaletteExtractor
from neural_canvas.utils.image_utils import (
    add_watermark,
    calculate_image_stats,
    create_grid,
    resize_with_aspect_ratio,
)

# ---------------------------------------------------------------------------
# Helper: create a vibrant synthetic test image (no external file needed)
# ---------------------------------------------------------------------------


def create_test_image(width: int = 640, height: int = 480, seed: int = 42) -> Image.Image:
    """Generate a colourful gradient image suitable for demonstrating filters."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((height, width, 3), dtype=np.float32)

    # Horizontal gradient
    c1 = rng.integers(50, 220, size=3).astype(np.float32)
    c2 = rng.integers(50, 220, size=3).astype(np.float32)
    for x in range(width):
        t = x / max(width - 1, 1)
        arr[:, x, :] = c1 * (1 - t) + c2 * t

    # Vertical vignette
    c3 = rng.integers(30, 180, size=3).astype(np.float32)
    for y in range(height):
        t = y / max(height - 1, 1)
        arr[y, :, :] = arr[y, :, :] * (1 - t * 0.5) + c3 * (t * 0.5)

    # Add noise blobs
    for _ in range(5):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        radius = int(rng.integers(40, 120))
        color = rng.integers(0, 256, size=3).astype(np.float32)
        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = np.clip(1.0 - dist / radius, 0, 1)[:, :, np.newaxis]
        arr = arr * (1 - mask * 0.6) + color * (mask * 0.6)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR = Path("examples/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Neural Canvas  –  Basic Usage Demo")
    print("=" * 60 + "\n")

    # Create test image
    print("  [1/5] Generating synthetic test image ...")
    test_img = create_test_image(width=640, height=480, seed=99)
    test_img_path = OUTPUT_DIR / "test_source.png"
    test_img.save(test_img_path)
    print(f"        Saved: {test_img_path}\n")

    # ------------------------------------------------------------------ #
    # 2. Apply all built-in styles                                         #
    # ------------------------------------------------------------------ #
    print("  [2/5] Applying style presets ...")
    pipeline = StyleTransferPipeline()
    styles = ["cyberpunk", "watercolor", "oil_painting", "sketch", "neon", "vintage", "glitch"]
    styled_images = []

    for style in styles:
        t0 = time.perf_counter()
        out_path = OUTPUT_DIR / f"style_{style}.png"
        result = pipeline.generate(
            input_path=test_img_path,
            style=style,
            output_path=str(out_path),
            seed=42,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        styled_images.append(Image.open(out_path))
        print(f"        {style:<15} -> {out_path.name}  ({elapsed:.0f}ms)")

    print()

    # ------------------------------------------------------------------ #
    # 3. Apply individual filters directly                                  #
    # ------------------------------------------------------------------ #
    print("  [3/5] Applying individual filters ...")
    filters_demo = [
        ("neon_direct", NeonFilter(intensity=0.95, color_boost=2.0)),
        ("glitch_heavy", GlitchFilter(strength=0.55, bands=25, seed=7)),
        ("vintage_soft", VintageFilter(grain_strength=0.25, vignette_strength=0.6)),
        ("watercolor_detailed", WatercolorFilter(blur_radius=1.8, edge_strength=0.85)),
        ("sketch_fine", SketchFilter(line_thickness=1.0, threshold=90)),
    ]

    filter_gallery = []
    for name, filt in filters_demo:
        out_path = OUTPUT_DIR / f"filter_{name}.png"
        result_img = filt.apply(test_img)
        result_img.save(out_path)
        filter_gallery.append(result_img)
        print(f"        {name:<25} -> {out_path.name}")

    print()

    # ------------------------------------------------------------------ #
    # 4. Colour palette extraction and harmony                             #
    # ------------------------------------------------------------------ #
    print("  [4/5] Colour palette tools ...")

    extractor = PaletteExtractor(n_colors=6)
    palette = extractor.extract(test_img)
    print(f"        Dominant colours extracted: {len(palette)}")
    for i, color in enumerate(palette):
        hex_val = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        print(f"          [{i+1}] rgb{color}  {hex_val}")

    swatch = extractor.extract_as_image(test_img, swatch_size=60)
    swatch_path = OUTPUT_DIR / "palette_swatches.png"
    swatch.save(swatch_path)
    print(f"        Palette swatch saved: {swatch_path}")

    # Colour harmonies from first dominant colour
    seed_color = palette[0]
    harmony = ColorHarmony(seed_color)
    print(f"\n        Seed colour: rgb{seed_color}")
    print(f"        Complementary:       {harmony.as_hex(harmony.complementary())}")
    print(f"        Analogous (±30°):    {harmony.as_hex(harmony.analogous())}")
    print(f"        Triadic:             {harmony.as_hex(harmony.triadic())}")
    print(f"        Tetradic:            {harmony.as_hex(harmony.tetradic())}")
    print()

    # ------------------------------------------------------------------ #
    # 5. Gallery grid + watermark + stats                                  #
    # ------------------------------------------------------------------ #
    print("  [5/5] Building gallery grid and computing stats ...")

    # Resize all styled images to uniform cells
    cell_w, cell_h = 320, 240
    thumbnails = [
        img.resize((cell_w, cell_h), Image.LANCZOS) for img in styled_images[:6]
    ]

    grid = create_grid(
        thumbnails,
        cols=3,
        padding=6,
        border_color=(80, 80, 80),
        border_width=1,
    )
    grid_path = OUTPUT_DIR / "gallery_grid.png"
    grid.save(grid_path)
    print(f"        Gallery grid saved:  {grid_path}")

    # Watermarked version
    watermarked = add_watermark(
        grid,
        text="Neural Canvas 2024",
        position="bottom-right",
        opacity=0.55,
        font_size=18,
    )
    wm_path = OUTPUT_DIR / "gallery_watermarked.png"
    watermarked.save(wm_path)
    print(f"        Watermarked grid:    {wm_path}")

    # Stats for original vs. cyberpunk
    orig_stats = calculate_image_stats(test_img)
    cyber_stats = calculate_image_stats(styled_images[0])
    print("\n        Image statistics comparison:")
    print(f"          {'Metric':<20} {'Original':>12} {'Cyberpunk':>12}")
    print(f"          {'-'*44}")
    for key in orig_stats:
        print(
            f"          {key:<20} {orig_stats[key]:>12.4f} {cyber_stats[key]:>12.4f}"
        )

    print("\n" + "=" * 60)
    print(f"  Done!  All output saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
