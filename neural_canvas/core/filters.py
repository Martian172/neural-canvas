"""
neural_canvas.core.filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Artistic image filter implementations for Neural Canvas.

Each filter is a class with a single public method::

    filter.apply(image: PIL.Image.Image) -> PIL.Image.Image

Filters are stateless once constructed – you can safely apply the same
filter instance to multiple images concurrently.

Available filters:
    - :class:`GlitchFilter`     – digital corruption artifacts
    - :class:`VintageFilter`    – sepia tone, grain, vignette
    - :class:`NeonFilter`       – neon glow and colour boost
    - :class:`WatercolorFilter` – soft edges and colour bleeding
    - :class:`SketchFilter`     – pencil line drawing effect
"""

from __future__ import annotations

import abc
import random
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseFilter(abc.ABC):
    """Abstract base class for all Neural Canvas filters.

    Sub-classes must implement :meth:`apply`.

    Attributes:
        name: Human-readable name for this filter.
    """

    name: str = "base"

    @abc.abstractmethod
    def apply(self, image: Image.Image) -> Image.Image:
        """Apply the filter to *image* and return a new image.

        Args:
            image: The source PIL image.  Must be in ``"RGB"`` mode; the
                   implementation may convert internally and convert back.

        Returns:
            A new PIL ``Image`` object with the filter applied.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    # Convenience: allow ``filter(image)`` syntax
    def __call__(self, image: Image.Image) -> Image.Image:
        return self.apply(image)


# ---------------------------------------------------------------------------
# Glitch Filter
# ---------------------------------------------------------------------------


class GlitchFilter(BaseFilter):
    """Simulates digital data corruption with horizontal band offsets and
    RGB channel separation.

    Args:
        strength: Overall glitch intensity in the range ``[0.0, 1.0]``.
                  Controls the maximum pixel offset per band.
        bands: Number of horizontal glitch bands to create.
        seed: Optional random seed for reproducibility.

    Example::

        img = GlitchFilter(strength=0.4, bands=15).apply(Image.open("photo.jpg"))
    """

    name = "glitch"

    def __init__(
        self,
        strength: float = 0.3,
        bands: int = 12,
        seed: Optional[int] = None,
    ) -> None:
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.bands = max(1, int(bands))
        self.seed = seed

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply glitch distortion to *image*.

        Splits the image into horizontal bands and shifts each band by a
        random horizontal offset.  Also applies a small RGB channel
        displacement to simulate data corruption.

        Args:
            image: Source PIL image.

        Returns:
            Glitched PIL image.
        """
        rng = np.random.default_rng(self.seed)
        img = image.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        h, w, _ = arr.shape

        # ---- Band-based horizontal shift ----
        max_shift = int(w * self.strength * 0.15)
        if max_shift > 0:
            band_height = max(1, h // self.bands)
            for y in range(0, h, band_height):
                if rng.random() < self.strength:
                    shift = int(rng.integers(-max_shift, max_shift + 1))
                    arr[y: y + band_height, :, :] = np.roll(
                        arr[y: y + band_height, :, :], shift, axis=1
                    )

        # ---- RGB channel separation ----
        channel_shift = int(w * self.strength * 0.02)
        if channel_shift > 0:
            r_shift = int(rng.integers(-channel_shift, channel_shift + 1))
            b_shift = int(rng.integers(-channel_shift, channel_shift + 1))
            arr[:, :, 0] = np.roll(arr[:, :, 0], r_shift, axis=1)  # red channel
            arr[:, :, 2] = np.roll(arr[:, :, 2], b_shift, axis=1)  # blue channel

        return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Vintage Filter
# ---------------------------------------------------------------------------


class VintageFilter(BaseFilter):
    """Applies a vintage/retro look: sepia tone, film grain, and vignette.

    Args:
        grain_strength: Intensity of film-grain noise in ``[0.0, 1.0]``.
        vignette_strength: Darkness of the vignette edge in ``[0.0, 1.0]``.
        sepia_strength: How strongly to tint the image sepia in ``[0.0, 1.0]``.
        seed: Optional random seed for the grain pattern.

    Example::

        img = VintageFilter(grain_strength=0.35, vignette_strength=0.55).apply(photo)
    """

    name = "vintage"

    def __init__(
        self,
        grain_strength: float = 0.3,
        vignette_strength: float = 0.4,
        sepia_strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        self.grain_strength = float(np.clip(grain_strength, 0.0, 1.0))
        self.vignette_strength = float(np.clip(vignette_strength, 0.0, 1.0))
        self.sepia_strength = float(np.clip(sepia_strength, 0.0, 1.0))
        self.seed = seed

    @staticmethod
    def _sepia(arr: np.ndarray, strength: float) -> np.ndarray:
        """Apply sepia-tone blending to an RGB array.

        Args:
            arr: Float array of shape ``(H, W, 3)`` with values in ``[0, 255]``.
            strength: Blend weight (0 = no change, 1 = full sepia).

        Returns:
            Modified array with sepia tone applied.
        """
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        new_r = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255)
        new_g = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255)
        new_b = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255)

        sepia = np.stack([new_r, new_g, new_b], axis=-1)
        return (arr * (1.0 - strength) + sepia * strength).astype(np.float32)

    @staticmethod
    def _vignette(arr: np.ndarray, strength: float) -> np.ndarray:
        """Apply elliptical vignette darkening to an RGB array.

        Args:
            arr: Float array of shape ``(H, W, 3)``.
            strength: How dark the edges become (0 = no change, 1 = black edges).

        Returns:
            Vignetted array.
        """
        h, w = arr.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y = np.linspace(0, h - 1, h)
        x = np.linspace(0, w - 1, w)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(((X - cx) / (w / 2.0)) ** 2 + ((Y - cy) / (h / 2.0)) ** 2)
        dist = np.clip(dist, 0, 1)
        mask = 1.0 - dist * strength
        mask = np.clip(mask, 0, 1)[:, :, np.newaxis]
        return (arr * mask).astype(np.float32)

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply vintage effect to *image*.

        Args:
            image: Source PIL image.

        Returns:
            Vintage-styled PIL image.
        """
        rng = np.random.default_rng(self.seed)
        arr = np.array(image.convert("RGB"), dtype=np.float32)

        # Sepia tone
        arr = self._sepia(arr, self.sepia_strength)

        # Film grain
        if self.grain_strength > 0:
            noise = rng.normal(0, self.grain_strength * 25, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255)

        # Vignette
        arr = self._vignette(arr, self.vignette_strength)

        # Fade colours slightly
        arr = np.clip(arr * 0.95 + 10, 0, 255)

        return Image.fromarray(arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Neon Filter
# ---------------------------------------------------------------------------


class NeonFilter(BaseFilter):
    """Creates a neon glow effect by boosting saturation and adding a
    Gaussian-blurred luminance overlay.

    Args:
        intensity: Glow intensity in ``[0.0, 1.0]``.  Controls how much of
                   the blurred overlay is blended into the original.
        color_boost: Colour saturation multiplier (>1 intensifies colours).
        glow_radius: Radius of the Gaussian blur used for the glow overlay.

    Example::

        img = NeonFilter(intensity=0.9, color_boost=1.8).apply(night_photo)
    """

    name = "neon"

    def __init__(
        self,
        intensity: float = 0.8,
        color_boost: float = 1.6,
        glow_radius: int = 8,
    ) -> None:
        self.intensity = float(np.clip(intensity, 0.0, 1.0))
        self.color_boost = float(max(0.1, color_boost))
        self.glow_radius = max(1, int(glow_radius))

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply neon glow effect to *image*.

        Steps:
        1. Boost colour saturation.
        2. Generate a Gaussian-blurred glow layer.
        3. Screen-blend the glow onto the original.
        4. Darken shadows to deepen the neon feel.

        Args:
            image: Source PIL image.

        Returns:
            Neon-lit PIL image.
        """
        img = image.convert("RGB")

        # --- Boost saturation ---
        img = ImageEnhance.Color(img).enhance(self.color_boost)

        # --- Build glow layer ---
        blurred = img.filter(ImageFilter.GaussianBlur(radius=self.glow_radius))

        # --- Screen blend: result = 1 - (1 - a) * (1 - b) ---
        arr_base = np.array(img, dtype=np.float32) / 255.0
        arr_blur = np.array(blurred, dtype=np.float32) / 255.0

        glow_blend = 1.0 - (1.0 - arr_base) * (1.0 - arr_blur * self.intensity)
        glow_blend = np.clip(glow_blend, 0.0, 1.0)

        # --- Deepen shadows ---
        glow_blend = np.power(glow_blend, 1.0 / (1.0 + self.intensity * 0.4))

        result_arr = (glow_blend * 255).astype(np.uint8)
        return Image.fromarray(result_arr, mode="RGB")


# ---------------------------------------------------------------------------
# Watercolor Filter
# ---------------------------------------------------------------------------


class WatercolorFilter(BaseFilter):
    """Simulates a watercolor painting by smoothing flat regions while
    preserving and enhancing edges, then overlaying a subtle paper texture.

    Args:
        blur_radius: Radius of the smoothing blur applied to non-edge regions.
        edge_strength: How prominently edges are drawn back onto the image.
        paper_texture_opacity: Opacity of the paper grain overlay in ``[0, 1]``.

    Example::

        img = WatercolorFilter(blur_radius=3.0, edge_strength=0.7).apply(landscape)
    """

    name = "watercolor"

    def __init__(
        self,
        blur_radius: float = 2.5,
        edge_strength: float = 0.6,
        paper_texture_opacity: float = 0.12,
    ) -> None:
        self.blur_radius = max(0.5, float(blur_radius))
        self.edge_strength = float(np.clip(edge_strength, 0.0, 1.0))
        self.paper_texture_opacity = float(
            np.clip(paper_texture_opacity, 0.0, 1.0)
        )

    @staticmethod
    def _paper_texture(size: tuple[int, int], seed: int = 0) -> np.ndarray:
        """Generate a random paper-grain texture.

        Args:
            size: ``(width, height)`` of the texture to generate.
            seed: Random seed.

        Returns:
            Float array of shape ``(H, W)`` with values in ``[0, 1]``.
        """
        rng = np.random.default_rng(seed)
        w, h = size
        grain = rng.uniform(0.85, 1.15, (h, w)).astype(np.float32)
        # Smooth slightly so it looks like paper fibre
        from scipy.ndimage import uniform_filter
        grain = uniform_filter(grain, size=3)
        return grain

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply watercolor effect to *image*.

        Args:
            image: Source PIL image.

        Returns:
            Watercolor-styled PIL image.
        """
        img = image.convert("RGB")
        arr = np.array(img, dtype=np.float32)

        # --- Multi-pass bilateral-like smoothing using Pillow ---
        smooth = img
        for _ in range(3):
            smooth = smooth.filter(
                ImageFilter.GaussianBlur(radius=self.blur_radius)
            )

        # --- Extract edges from original image ---
        grey = ImageOps.grayscale(img)
        edges = grey.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=0.5))
        edge_arr = np.array(edges, dtype=np.float32) / 255.0

        # --- Compose: smooth + edges ---
        smooth_arr = np.array(smooth, dtype=np.float32)
        for c in range(3):
            smooth_arr[:, :, c] = np.clip(
                smooth_arr[:, :, c] * (1.0 - edge_arr * self.edge_strength),
                0,
                255,
            )

        # --- Paper texture overlay ---
        if self.paper_texture_opacity > 0:
            texture = self._paper_texture(image.size)[:, :, np.newaxis]
            smooth_arr = np.clip(smooth_arr * texture, 0, 255)

        return Image.fromarray(smooth_arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Sketch Filter
# ---------------------------------------------------------------------------


class SketchFilter(BaseFilter):
    """Converts an image into a pencil-sketch style drawing.

    The algorithm:
    1. Converts to greyscale.
    2. Inverts the greyscale.
    3. Blurs the inverted image.
    4. Colour-dodges the greyscale with the blurred image to produce sketch lines.
    5. Optionally re-introduces a faint tint for "coloured pencil" look.

    Args:
        line_thickness: Blur radius used in step 3; larger → thicker lines.
        threshold: Greyscale value (0–255) below which pixels are forced black.
        color_tint: Optional ``(R, G, B)`` tuple to tint the output (e.g. sepia).

    Example::

        img = SketchFilter(line_thickness=1.5, threshold=100).apply(portrait)
    """

    name = "sketch"

    def __init__(
        self,
        line_thickness: float = 1.5,
        threshold: int = 128,
        color_tint: Optional[tuple[int, int, int]] = None,
    ) -> None:
        self.line_thickness = max(0.5, float(line_thickness))
        self.threshold = int(np.clip(threshold, 0, 255))
        self.color_tint = color_tint

    def apply(self, image: Image.Image) -> Image.Image:
        """Apply pencil-sketch effect to *image*.

        Args:
            image: Source PIL image.

        Returns:
            Sketch-styled PIL image.
        """
        img = image.convert("L")  # greyscale
        inverted = ImageOps.invert(img)
        blurred = inverted.filter(
            ImageFilter.GaussianBlur(radius=self.line_thickness * 5)
        )

        # Colour-dodge blend
        inv_arr = np.array(inverted, dtype=np.float32)
        blur_arr = np.array(blurred, dtype=np.float32)

        # Dodge formula: result = base / (1 - overlay/255)
        # Clamp to avoid division by zero / overflow
        denominator = 1.0 - (blur_arr / 255.0)
        denominator = np.clip(denominator, 1e-5, 1.0)
        sketch_arr = np.clip(inv_arr / denominator, 0, 255).astype(np.uint8)

        # Apply threshold to darken lines
        sketch_arr = np.where(sketch_arr < self.threshold, 0, sketch_arr)

        # Convert back to RGB
        sketch_img = Image.fromarray(sketch_arr, mode="L").convert("RGB")

        # Optional colour tint (e.g. sepia pencil)
        if self.color_tint:
            arr = np.array(sketch_img, dtype=np.float32)
            r_scale = self.color_tint[0] / 128.0 if self.color_tint[0] else 1.0
            g_scale = self.color_tint[1] / 128.0 if self.color_tint[1] else 1.0
            b_scale = self.color_tint[2] / 128.0 if self.color_tint[2] else 1.0
            arr[:, :, 0] = np.clip(arr[:, :, 0] * r_scale, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * g_scale, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * b_scale, 0, 255)
            sketch_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

        return sketch_img
