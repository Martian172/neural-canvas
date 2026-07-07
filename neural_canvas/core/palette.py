"""
neural_canvas.core.palette
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Color palette extraction and application tools.

Classes:
    :class:`PaletteExtractor`  – finds dominant colours in an image.
    :class:`PaletteApplier`    – maps an image's colours to a target palette.
    :class:`ColorHarmony`      – generates harmonious colour palettes from a seed colour.

Example::

    from PIL import Image
    from neural_canvas.core.palette import PaletteExtractor, ColorHarmony

    img = Image.open("photo.jpg")
    extractor = PaletteExtractor(n_colors=6)
    palette = extractor.extract(img)
    print(palette)  # [(R, G, B), ...]

    harmony = ColorHarmony((255, 80, 0))
    print(harmony.complementary())
    print(harmony.analogous())
    print(harmony.triadic())
"""

from __future__ import annotations

import colorsys
import math
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Type alias for an RGB colour
Color = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Palette Extractor
# ---------------------------------------------------------------------------


class PaletteExtractor:
    """Extracts the dominant colours from an image using k-means clustering.

    Args:
        n_colors: Number of dominant colours to extract.
        sample_pixels: Number of pixels to sample (sub-samples large images
                       for speed).  ``None`` uses all pixels.
        seed: Random seed for k-means initialisation.

    Example::

        extractor = PaletteExtractor(n_colors=5)
        palette = extractor.extract(Image.open("photo.jpg"))
        for color in palette:
            print(f"rgb{color}")
    """

    def __init__(
        self,
        n_colors: int = 6,
        sample_pixels: Optional[int] = 10_000,
        seed: int = 42,
    ) -> None:
        self.n_colors = max(1, n_colors)
        self.sample_pixels = sample_pixels
        self.seed = seed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kmeans(
        data: np.ndarray,
        k: int,
        max_iter: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        """Run a simple k-means clustering on *data*.

        Args:
            data: Float array of shape ``(N, D)``.
            k: Number of clusters.
            max_iter: Maximum iterations.
            seed: Random seed for centroid initialisation.

        Returns:
            Array of centroid vectors of shape ``(k, D)``.
        """
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(data), size=k, replace=False)
        centroids = data[idx].astype(np.float64)

        for _ in range(max_iter):
            # Assignment
            dists = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            # Update
            new_centroids = np.array(
                [
                    data[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c]
                    for c in range(k)
                ]
            )
            if np.allclose(centroids, new_centroids, atol=0.5):
                break
            centroids = new_centroids

        return centroids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: Image.Image) -> List[Color]:
        """Extract dominant colours from *image*.

        Args:
            image: A PIL image in any mode.

        Returns:
            A list of ``(R, G, B)`` tuples sorted by perceptual luminance,
            brightest first.

        Example::

            palette = PaletteExtractor(n_colors=8).extract(img)
        """
        img = image.convert("RGB")
        pixels = np.array(img, dtype=np.float32).reshape(-1, 3)

        # Sub-sample for speed
        if self.sample_pixels and len(pixels) > self.sample_pixels:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(pixels), size=self.sample_pixels, replace=False)
            pixels = pixels[idx]

        k = min(self.n_colors, len(pixels))
        centroids = self._kmeans(pixels, k=k, seed=self.seed)

        # Sort by perceived luminance (ITU-R BT.601)
        def luminance(rgb: np.ndarray) -> float:
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

        centroids = sorted(centroids, key=luminance, reverse=True)
        return [
            (int(np.clip(c[0], 0, 255)),
             int(np.clip(c[1], 0, 255)),
             int(np.clip(c[2], 0, 255)))
            for c in centroids
        ]

    def extract_as_image(
        self,
        image: Image.Image,
        swatch_size: int = 80,
    ) -> Image.Image:
        """Extract palette and render it as a horizontal swatch strip.

        Args:
            image: Source image.
            swatch_size: Width and height of each colour swatch in pixels.

        Returns:
            A PIL image containing the colour swatches side-by-side.
        """
        palette = self.extract(image)
        strip = Image.new("RGB", (swatch_size * len(palette), swatch_size))
        for i, color in enumerate(palette):
            swatch = Image.new("RGB", (swatch_size, swatch_size), color=color)
            strip.paste(swatch, (i * swatch_size, 0))
        return strip


# ---------------------------------------------------------------------------
# Palette Applier
# ---------------------------------------------------------------------------


class PaletteApplier:
    """Maps the colours of an image to a target colour palette via nearest-
    neighbour matching in RGB space.

    This technique is sometimes called *palette quantisation* or *colour remapping*.

    Args:
        palette: A list of ``(R, G, B)`` tuples defining the target palette.
        dithering: If ``True``, apply Floyd–Steinberg dithering to reduce
                   banding (currently uses Pillow's built-in quantise).

    Example::

        applier = PaletteApplier([(255, 0, 0), (0, 0, 255), (255, 255, 0)])
        result = applier.apply(image)
    """

    def __init__(
        self,
        palette: List[Color],
        dithering: bool = True,
    ) -> None:
        if not palette:
            raise ValueError("Palette must contain at least one colour.")
        self.palette = palette
        self.dithering = dithering

    def apply(self, image: Image.Image) -> Image.Image:
        """Remap *image* colours to the palette.

        Args:
            image: Source PIL image.

        Returns:
            Colour-remapped PIL image.
        """
        img = image.convert("RGB")
        palette_img = Image.new("P", (1, 1))

        flat_palette = []
        for r, g, b in self.palette:
            flat_palette.extend([r, g, b])
        # Pad to 256 colours (768 values) as required by Pillow
        flat_palette.extend([0] * (768 - len(flat_palette)))
        palette_img.putpalette(flat_palette)

        dither_mode = Image.Dither.FLOYDSTEINBERG if self.dithering else Image.Dither.NONE
        quantised = img.quantize(
            colors=len(self.palette),
            palette=palette_img,
            dither=dither_mode,
        )
        return quantised.convert("RGB")

    def blend_apply(
        self, image: Image.Image, blend_strength: float = 0.6
    ) -> Image.Image:
        """Apply palette remapping and blend with the original for a subtler effect.

        Args:
            image: Source PIL image.
            blend_strength: How much of the remapped image to use (0 = original,
                            1 = fully remapped).

        Returns:
            Blended PIL image.
        """
        remapped = self.apply(image)
        original = np.array(image.convert("RGB"), dtype=np.float32)
        recolored = np.array(remapped, dtype=np.float32)
        blended = original * (1.0 - blend_strength) + recolored * blend_strength
        return Image.fromarray(blended.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Color Harmony
# ---------------------------------------------------------------------------


class ColorHarmony:
    """Generates harmonious colour palettes from a single seed colour.

    Colour harmony rules are computed in HSV space:
    - **Complementary**: colour directly opposite on the wheel (+180°).
    - **Analogous**: colours adjacent on the wheel (±30°).
    - **Triadic**: three colours evenly spaced around the wheel (+120°).
    - **Split-complementary**: the complement and its two adjacent colours.
    - **Tetradic**: four colours at 90° intervals.

    Args:
        seed_color: An ``(R, G, B)`` tuple in ``[0, 255]``.

    Example::

        harmony = ColorHarmony((220, 50, 50))
        comp = harmony.complementary()
        print(comp)  # [(220, 50, 50), (50, 220, 220)]
    """

    def __init__(self, seed_color: Color) -> None:
        r, g, b = (v / 255.0 for v in seed_color)
        self._h, self._s, self._v = colorsys.rgb_to_hsv(r, g, b)
        self.seed_color = seed_color

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Color:
        """Convert HSV (0–1 each) to an integer ``(R, G, B)`` tuple."""
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def _rotate(self, degrees: float) -> Color:
        """Rotate the seed hue by *degrees* and return the new colour."""
        return self._hsv_to_rgb(
            self._h + degrees / 360.0, self._s, self._v
        )

    # ------------------------------------------------------------------
    # Harmony methods
    # ------------------------------------------------------------------

    def complementary(self) -> List[Color]:
        """Return the seed colour and its complement (+180°).

        Returns:
            List of two colours: ``[seed, complement]``.
        """
        return [self.seed_color, self._rotate(180)]

    def analogous(self, angle: float = 30.0) -> List[Color]:
        """Return the seed colour and two adjacent colours.

        Args:
            angle: Separation angle in degrees (default 30°).

        Returns:
            List of three colours: ``[left, seed, right]``.
        """
        return [self._rotate(-angle), self.seed_color, self._rotate(angle)]

    def triadic(self) -> List[Color]:
        """Return three colours evenly spaced at 120° intervals.

        Returns:
            List of three colours starting with the seed.
        """
        return [self.seed_color, self._rotate(120), self._rotate(240)]

    def split_complementary(self, angle: float = 30.0) -> List[Color]:
        """Return the seed and the two colours adjacent to its complement.

        Args:
            angle: Adjacency angle around the complement (default 30°).

        Returns:
            List of three colours: ``[seed, comp_left, comp_right]``.
        """
        return [
            self.seed_color,
            self._rotate(180 - angle),
            self._rotate(180 + angle),
        ]

    def tetradic(self) -> List[Color]:
        """Return four colours at 90° intervals.

        Returns:
            List of four colours starting with the seed.
        """
        return [
            self.seed_color,
            self._rotate(90),
            self._rotate(180),
            self._rotate(270),
        ]

    def monochromatic(self, steps: int = 5) -> List[Color]:
        """Return a monochromatic range by varying the value (brightness).

        Args:
            steps: Number of shades to generate.

        Returns:
            List of colours from dark to light.
        """
        return [
            self._hsv_to_rgb(self._h, self._s, i / (steps - 1))
            for i in range(steps)
        ]

    def as_hex(self, palette: List[Color]) -> List[str]:
        """Convert a colour list to HTML hex strings.

        Args:
            palette: List of ``(R, G, B)`` tuples.

        Returns:
            List of ``"#RRGGBB"`` strings.

        Example::

            harmony = ColorHarmony((100, 200, 50))
            hexes = harmony.as_hex(harmony.triadic())
        """
        return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in palette]
