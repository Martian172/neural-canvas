"""
neural_canvas.core.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main pipeline orchestrator for Neural Canvas.

This module provides the ``StyleTransferPipeline`` class that coordinates
loading styles, applying filter chains, handling batch processing, and
saving results.

Example::

    pipeline = StyleTransferPipeline()
    result = pipeline.generate("input.jpg", style="cyberpunk", output_path="out.png")
    print(result.elapsed_ms)
"""

from __future__ import annotations

import os
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

from neural_canvas.config import NeuralCanvasConfig, STYLE_PRESETS
from neural_canvas.core.filters import (
    BaseFilter,
    GlitchFilter,
    VintageFilter,
    NeonFilter,
    WatercolorFilter,
    SketchFilter,
)
from neural_canvas.utils.image_utils import resize_with_aspect_ratio, convert_format

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of built-in filter constructors keyed by filter name
# ---------------------------------------------------------------------------
_FILTER_REGISTRY: Dict[str, Callable[..., BaseFilter]] = {
    "glitch": GlitchFilter,
    "vintage": VintageFilter,
    "neon": NeonFilter,
    "watercolor": WatercolorFilter,
    "sketch": SketchFilter,
}


@dataclass
class GenerationResult:
    """Holds the result of a single image generation/transformation.

    Attributes:
        output_path: Absolute path to the saved output file.
        style_name: Name of the style that was applied.
        elapsed_ms: Wall-clock time taken by the operation in milliseconds.
        input_path: Source image path (may be ``None`` for synthetic images).
        filters_applied: Ordered list of filter names that were applied.
        metadata: Arbitrary key/value metadata (seed, dimensions, …).
        success: Whether the operation completed without errors.
        error: Error message if ``success`` is ``False``.
    """

    output_path: str
    style_name: str
    elapsed_ms: float
    input_path: Optional[str] = None
    filters_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def __repr__(self) -> str:  # pragma: no cover
        status = "OK" if self.success else f"ERR({self.error})"
        return (
            f"<GenerationResult style={self.style_name!r} "
            f"elapsed={self.elapsed_ms:.1f}ms status={status}>"
        )


class StyleTransferPipeline:
    """Orchestrates style loading, filter application, and batch processing.

    The pipeline maintains a registry of available styles and their
    associated filter chains.  At runtime you can register custom styles
    or filters using :meth:`register_style` and :meth:`register_filter`.

    Args:
        config: A :class:`~neural_canvas.config.NeuralCanvasConfig` instance.
                Defaults to the global default configuration.

    Example::

        pipeline = StyleTransferPipeline()

        # Single image
        result = pipeline.generate(
            "landscape.jpg",
            style="watercolor",
            output_path="landscape_wc.png",
        )

        # Batch
        results = pipeline.batch_generate(
            input_dir="./photos",
            style="cyberpunk",
            output_dir="./cyberpunk_art",
        )
    """

    # ------------------------------------------------------------------
    # Built-in style presets that map to filter chains + parameters
    # ------------------------------------------------------------------
    _BUILTIN_STYLES: Dict[str, Dict[str, Any]] = {
        "cyberpunk": {
            "description": "Electric neon glows, dark backgrounds, ultraviolet palette",
            "filters": [
                ("neon", {"intensity": 0.9, "color_boost": 1.8}),
                ("glitch", {"strength": 0.2, "bands": 8}),
            ],
            "brightness": 0.7,
            "contrast": 1.4,
            "saturation": 1.6,
            "color_tint": (0, 0, 40),  # subtle blue shift
        },
        "watercolor": {
            "description": "Soft edges, color bleeding, paper texture",
            "filters": [
                ("watercolor", {"blur_radius": 2.5, "edge_strength": 0.6}),
            ],
            "brightness": 1.1,
            "contrast": 0.95,
            "saturation": 1.2,
            "color_tint": (10, 5, 0),
        },
        "oil_painting": {
            "description": "Thick brush strokes, rich saturated colors",
            "filters": [
                ("watercolor", {"blur_radius": 1.5, "edge_strength": 0.9}),
            ],
            "brightness": 0.95,
            "contrast": 1.2,
            "saturation": 1.5,
            "color_tint": (15, 8, 0),
        },
        "sketch": {
            "description": "Pencil lines, grayscale detail, cross-hatching",
            "filters": [
                ("sketch", {"line_thickness": 1.5, "threshold": 128}),
            ],
            "brightness": 1.05,
            "contrast": 1.1,
            "saturation": 0.0,
            "color_tint": (0, 0, 0),
        },
        "neon": {
            "description": "Saturated highlights, dark shadows, neon glow",
            "filters": [
                ("neon", {"intensity": 1.0, "color_boost": 2.0}),
            ],
            "brightness": 0.65,
            "contrast": 1.6,
            "saturation": 2.0,
            "color_tint": (10, 0, 20),
        },
        "vintage": {
            "description": "Sepia tone, grain, vignette, faded colors",
            "filters": [
                ("vintage", {"grain_strength": 0.4, "vignette_strength": 0.5}),
            ],
            "brightness": 0.95,
            "contrast": 0.9,
            "saturation": 0.6,
            "color_tint": (20, 10, -10),
        },
        "glitch": {
            "description": "Digital corruption artifacts, color channel separation",
            "filters": [
                ("glitch", {"strength": 0.6, "bands": 20}),
            ],
            "brightness": 1.0,
            "contrast": 1.2,
            "saturation": 1.4,
            "color_tint": (0, 0, 0),
        },
        "comic": {
            "description": "Bold outlines, flat colors, pop art feel",
            "filters": [
                ("sketch", {"line_thickness": 2.0, "threshold": 100}),
                ("neon", {"intensity": 0.4, "color_boost": 1.5}),
            ],
            "brightness": 1.1,
            "contrast": 1.5,
            "saturation": 1.8,
            "color_tint": (0, 0, 0),
        },
    }

    def __init__(self, config: Optional[NeuralCanvasConfig] = None) -> None:
        self.config = config or NeuralCanvasConfig()
        self._styles: Dict[str, Dict[str, Any]] = dict(self._BUILTIN_STYLES)
        self._filter_registry: Dict[str, Callable[..., BaseFilter]] = dict(
            _FILTER_REGISTRY
        )
        logger.debug("StyleTransferPipeline initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register_style(
        self, name: str, style_config: Dict[str, Any], overwrite: bool = False
    ) -> None:
        """Register a custom style preset.

        Args:
            name: Unique identifier for the style (lowercase, no spaces).
            style_config: Style configuration dict matching the internal
                          preset schema (``filters``, ``brightness``, etc.).
            overwrite: If ``True``, overwrite an existing style with the
                       same name.

        Raises:
            ValueError: If *name* is already registered and ``overwrite``
                        is ``False``.

        Example::

            pipeline.register_style("my_style", {
                "description": "My custom style",
                "filters": [("neon", {"intensity": 0.5})],
                "brightness": 1.0,
                "contrast": 1.0,
                "saturation": 1.0,
                "color_tint": (0, 0, 0),
            })
        """
        if name in self._styles and not overwrite:
            raise ValueError(
                f"Style '{name}' already registered. Pass overwrite=True to replace it."
            )
        self._styles[name] = style_config
        logger.info("Registered style '%s'", name)

    def register_filter(
        self,
        name: str,
        filter_cls: Callable[..., BaseFilter],
        overwrite: bool = False,
    ) -> None:
        """Register a custom filter class.

        Args:
            name: Identifier to use when referencing the filter in style configs.
            filter_cls: A callable that returns a :class:`BaseFilter` instance.
            overwrite: If ``True``, overwrite an existing filter with the same name.

        Raises:
            ValueError: If *name* is already registered and ``overwrite`` is ``False``.
        """
        if name in self._filter_registry and not overwrite:
            raise ValueError(
                f"Filter '{name}' already registered. Pass overwrite=True to replace it."
            )
        self._filter_registry[name] = filter_cls
        logger.info("Registered filter '%s'", name)

    # ------------------------------------------------------------------
    # Style management helpers
    # ------------------------------------------------------------------

    def list_styles(self) -> List[Dict[str, str]]:
        """Return a list of available styles with their descriptions.

        Returns:
            A list of dicts with ``name`` and ``description`` keys.
        """
        return [
            {"name": name, "description": cfg.get("description", "")}
            for name, cfg in self._styles.items()
        ]

    def load_style(self, style_name: str) -> Dict[str, Any]:
        """Load and return a style preset by name.

        Args:
            style_name: The name of the style to load.

        Returns:
            The style configuration dict.

        Raises:
            KeyError: If the style is not found in the registry.
        """
        if style_name not in self._styles:
            available = ", ".join(sorted(self._styles.keys()))
            raise KeyError(
                f"Unknown style '{style_name}'. Available styles: {available}"
            )
        return self._styles[style_name]

    # ------------------------------------------------------------------
    # Core image processing helpers
    # ------------------------------------------------------------------

    def _build_filter_chain(
        self, style_config: Dict[str, Any]
    ) -> List[BaseFilter]:
        """Instantiate the filter chain from a style configuration.

        Args:
            style_config: Style configuration dict.

        Returns:
            An ordered list of :class:`BaseFilter` instances.
        """
        chain: List[BaseFilter] = []
        for filter_name, filter_kwargs in style_config.get("filters", []):
            if filter_name not in self._filter_registry:
                logger.warning("Unknown filter '%s', skipping.", filter_name)
                continue
            filter_instance = self._filter_registry[filter_name](**filter_kwargs)
            chain.append(filter_instance)
        return chain

    def _apply_color_corrections(
        self, image: Image.Image, style_config: Dict[str, Any]
    ) -> Image.Image:
        """Apply brightness, contrast, saturation, and tint adjustments.

        Args:
            image: Source PIL image.
            style_config: Style configuration dict with numeric adjustment fields.

        Returns:
            Corrected PIL image.
        """
        # Brightness
        brightness = style_config.get("brightness", 1.0)
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)

        # Contrast
        contrast = style_config.get("contrast", 1.0)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)

        # Saturation / colour
        saturation = style_config.get("saturation", 1.0)
        if saturation != 1.0:
            image = ImageEnhance.Color(image).enhance(saturation)

        # Colour tint
        tint = style_config.get("color_tint", (0, 0, 0))
        if any(v != 0 for v in tint):
            image = self._apply_tint(image, tint)

        return image

    @staticmethod
    def _apply_tint(
        image: Image.Image, tint: tuple[int, int, int]
    ) -> Image.Image:
        """Add a subtle color tint to an RGB image.

        Args:
            image: Source image (will be converted to RGBA internally).
            tint: ``(R, G, B)`` offset values in the range ``[-255, 255]``.

        Returns:
            Tinted image in the original mode.
        """
        original_mode = image.mode
        arr = np.array(image.convert("RGB"), dtype=np.int16)
        for channel, offset in enumerate(tint):
            arr[:, :, channel] = np.clip(arr[:, :, channel] + offset, 0, 255)
        result = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        if original_mode == "RGBA":
            result = result.convert("RGBA")
        return result

    def apply_filter(
        self,
        image: Union[Image.Image, str, Path],
        filter_name: str,
        **filter_kwargs: Any,
    ) -> Image.Image:
        """Apply a single named filter to an image.

        Args:
            image: A PIL ``Image`` or a path string/Path to an image file.
            filter_name: Name of the filter to apply (e.g. ``"glitch"``).
            **filter_kwargs: Additional keyword arguments forwarded to the
                             filter constructor.

        Returns:
            Filtered PIL image.

        Raises:
            KeyError: If the filter name is not in the registry.
            FileNotFoundError: If *image* is a path and the file does not exist.

        Example::

            pipeline = StyleTransferPipeline()
            img = pipeline.apply_filter("photo.jpg", "neon", intensity=0.7)
            img.save("neon.png")
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path).convert("RGB")

        if filter_name not in self._filter_registry:
            available = ", ".join(sorted(self._filter_registry.keys()))
            raise KeyError(
                f"Unknown filter '{filter_name}'. Available: {available}"
            )

        filt = self._filter_registry[filter_name](**filter_kwargs)
        return filt.apply(image)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_path: Union[str, Path],
        style: str = "cyberpunk",
        output_path: Optional[Union[str, Path]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: int = 42,
        intensity: float = 1.0,
    ) -> GenerationResult:
        """Apply a style to a single image and save the result.

        Args:
            input_path: Path to the source image.
            style: Name of the style preset to apply.
            output_path: Where to save the result.  Defaults to
                         ``<input_stem>_<style>.png`` in the same directory.
            width: Resize output to this width (aspect ratio preserved when
                   only one dimension is specified).
            height: Resize output to this height.
            seed: Random seed for reproducible stochastic filters.
            intensity: Global intensity multiplier (0.0 – 2.0) applied to
                       the style's saturation/contrast settings.

        Returns:
            A :class:`GenerationResult` describing the outcome.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            KeyError: If *style* is not recognised.
        """
        t_start = time.perf_counter()
        random.seed(seed)
        np.random.seed(seed)

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_{style}.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        style_config = self.load_style(style)

        # Optionally scale the style's intensity-sensitive parameters
        if intensity != 1.0:
            style_config = dict(style_config)
            style_config["saturation"] = (
                (style_config.get("saturation", 1.0) - 1.0) * intensity + 1.0
            )
            style_config["contrast"] = (
                (style_config.get("contrast", 1.0) - 1.0) * intensity + 1.0
            )

        logger.info("Generating '%s' style for %s", style, input_path)

        image: Image.Image = Image.open(input_path).convert("RGB")

        # Resize if requested
        if width or height:
            image = resize_with_aspect_ratio(image, width=width, height=height)

        # Build and apply filter chain
        filter_chain = self._build_filter_chain(style_config)
        filters_applied: List[str] = []
        for filt in filter_chain:
            image = filt.apply(image)
            filters_applied.append(type(filt).__name__)

        # Apply colour corrections
        image = self._apply_color_corrections(image, style_config)

        # Save
        image.save(str(output_path))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info("Saved to %s (%.1f ms)", output_path, elapsed_ms)

        return GenerationResult(
            output_path=str(output_path),
            style_name=style,
            elapsed_ms=elapsed_ms,
            input_path=str(input_path),
            filters_applied=filters_applied,
            metadata={
                "seed": seed,
                "width": image.width,
                "height": image.height,
                "intensity": intensity,
            },
            success=True,
        )

    def batch_generate(
        self,
        input_dir: Union[str, Path],
        style: str = "cyberpunk",
        output_dir: Optional[Union[str, Path]] = None,
        max_workers: int = 4,
        recursive: bool = False,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        seed: int = 42,
        intensity: float = 1.0,
    ) -> List[GenerationResult]:
        """Batch-process all images in a directory.

        Args:
            input_dir: Directory containing source images.
            style: Style preset to apply to every image.
            output_dir: Destination directory (created if absent).  Defaults
                        to ``<input_dir>_<style>/``.
            max_workers: Maximum number of parallel worker threads.
            recursive: If ``True``, walk subdirectories as well.
            extensions: Tuple of file extensions to include.
            seed: Base random seed; each image uses ``seed + index``.
            intensity: Global intensity multiplier passed to each
                       :meth:`generate` call.

        Returns:
            A list of :class:`GenerationResult` objects – one per image.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_{style}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect images
        if recursive:
            image_paths = [
                p
                for p in input_dir.rglob("*")
                if p.suffix.lower() in extensions
            ]
        else:
            image_paths = [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            ]

        if not image_paths:
            logger.warning("No images found in %s", input_dir)
            return []

        logger.info(
            "Batch generating '%s' style for %d images using %d workers",
            style,
            len(image_paths),
            max_workers,
        )

        results: List[GenerationResult] = []

        def _process(idx: int, path: Path) -> GenerationResult:
            out_path = output_dir / f"{path.stem}_{style}{path.suffix}"
            try:
                return self.generate(
                    input_path=path,
                    style=style,
                    output_path=out_path,
                    seed=seed + idx,
                    intensity=intensity,
                )
            except Exception as exc:
                logger.error("Failed to process %s: %s", path, exc)
                return GenerationResult(
                    output_path=str(out_path),
                    style_name=style,
                    elapsed_ms=0.0,
                    input_path=str(path),
                    success=False,
                    error=str(exc),
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_process, i, p): p
                for i, p in enumerate(image_paths)
            }
            for future in as_completed(future_map):
                results.append(future.result())

        successes = sum(1 for r in results if r.success)
        logger.info(
            "Batch complete: %d/%d images processed successfully",
            successes,
            len(results),
        )
        return results
