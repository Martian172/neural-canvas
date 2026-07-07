"""
neural_canvas.config
~~~~~~~~~~~~~~~~~~~~

Configuration management for Neural Canvas.

Provides:
    - :class:`NeuralCanvasConfig` – runtime configuration dataclass
    - :data:`STYLE_PRESETS`       – dict of built-in style preset metadata
    - :func:`load_config`         – load config from environment / .env file
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Built-in style preset metadata (display info only — full configs live in pipeline)
# ---------------------------------------------------------------------------

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "cyberpunk": {
        "display_name": "Cyberpunk",
        "description": "Electric neon glows, dark backgrounds, ultraviolet palette",
        "tags": ["neon", "dark", "futuristic", "urban"],
        "recommended_intensity": 0.9,
    },
    "watercolor": {
        "display_name": "Watercolor",
        "description": "Soft edges, color bleeding, paper grain texture",
        "tags": ["soft", "artistic", "pastel", "natural"],
        "recommended_intensity": 0.7,
    },
    "oil_painting": {
        "display_name": "Oil Painting",
        "description": "Thick brush strokes, rich saturated colors, impasto effect",
        "tags": ["classic", "rich", "textured", "traditional"],
        "recommended_intensity": 0.8,
    },
    "sketch": {
        "display_name": "Pencil Sketch",
        "description": "Fine pencil lines, grayscale detail, cross-hatching shadows",
        "tags": ["minimal", "monochrome", "line-art", "drawing"],
        "recommended_intensity": 0.8,
    },
    "neon": {
        "display_name": "Neon Noir",
        "description": "Saturated highlights, dark shadows, neon glow, moody atmosphere",
        "tags": ["neon", "dark", "glow", "night"],
        "recommended_intensity": 1.0,
    },
    "vintage": {
        "display_name": "Vintage",
        "description": "Sepia tone, film grain, vignette, faded retro colors",
        "tags": ["retro", "sepia", "grain", "nostalgic"],
        "recommended_intensity": 0.7,
    },
    "glitch": {
        "display_name": "Glitch Art",
        "description": "Digital corruption artifacts, RGB channel displacement",
        "tags": ["digital", "experimental", "abstract", "corrupt"],
        "recommended_intensity": 0.6,
    },
    "comic": {
        "display_name": "Comic Book",
        "description": "Bold outlines, flat colors, pop art feel",
        "tags": ["bold", "flat", "pop-art", "illustration"],
        "recommended_intensity": 0.9,
    },
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class NeuralCanvasConfig:
    """Runtime configuration for the Neural Canvas pipeline.

    All attributes have sensible defaults and can be overridden via
    environment variables (see :func:`load_config`) or direct assignment.

    Attributes:
        default_style: Style preset applied when none is specified.
        default_output_dir: Default directory for generated images.
        max_image_width: Maximum allowed width for output images.
        max_image_height: Maximum allowed height for output images.
        default_seed: Default random seed used across stochastic operations.
        default_intensity: Default effect intensity multiplier.
        api_host: Default host for the FastAPI server.
        api_port: Default port for the FastAPI server.
        log_level: Python logging level string (e.g. ``"INFO"``).
        watermark_text: Text applied by the watermark utility (empty = no watermark).
        allowed_extensions: Image file extensions recognised by the batch processor.
    """

    default_style: str = "cyberpunk"
    default_output_dir: str = "./neural_canvas_output"
    max_image_width: int = 4096
    max_image_height: int = 4096
    default_seed: int = 42
    default_intensity: float = 0.8
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    log_level: str = "INFO"
    watermark_text: str = ""
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
    )

    def __post_init__(self) -> None:
        # Normalise output dir to an absolute path
        self.default_output_dir = str(Path(self.default_output_dir).resolve())


# ---------------------------------------------------------------------------
# Environment-based config loader
# ---------------------------------------------------------------------------


def load_config(env_file: Optional[str] = None) -> NeuralCanvasConfig:
    """Load a :class:`NeuralCanvasConfig` from environment variables.

    If a ``.env`` file path is provided (or ``.env`` exists in the current
    directory), variables from that file are loaded first via *python-dotenv*.

    Environment variables recognised (all prefixed with ``NEURAL_CANVAS_``):

    ============================================ ===========================
    Variable                                     Maps to
    ============================================ ===========================
    ``NEURAL_CANVAS_DEFAULT_STYLE``              ``default_style``
    ``NEURAL_CANVAS_DEFAULT_OUTPUT_DIR``         ``default_output_dir``
    ``NEURAL_CANVAS_MAX_IMAGE_WIDTH``            ``max_image_width``
    ``NEURAL_CANVAS_MAX_IMAGE_HEIGHT``           ``max_image_height``
    ``NEURAL_CANVAS_DEFAULT_SEED``               ``default_seed``
    ``NEURAL_CANVAS_DEFAULT_INTENSITY``          ``default_intensity``
    ``NEURAL_CANVAS_API_HOST``                   ``api_host``
    ``NEURAL_CANVAS_API_PORT``                   ``api_port``
    ``NEURAL_CANVAS_LOG_LEVEL``                  ``log_level``
    ``NEURAL_CANVAS_WATERMARK_TEXT``             ``watermark_text``
    ============================================ ===========================

    Args:
        env_file: Path to a ``.env`` file.  Pass ``None`` to use ``".env"``
                  in the current working directory (if it exists).

    Returns:
        A fully populated :class:`NeuralCanvasConfig` instance.

    Example::

        cfg = load_config(".env.production")
        pipeline = StyleTransferPipeline(config=cfg)
    """
    # Attempt to load .env file
    try:
        from dotenv import load_dotenv

        if env_file:
            load_dotenv(env_file)
        elif Path(".env").exists():
            load_dotenv(".env")
    except ImportError:
        pass  # python-dotenv not installed; silently continue

    def _int(key: str, default: int) -> int:
        val = os.environ.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def _float(key: str, default: float) -> float:
        val = os.environ.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def _str(key: str, default: str) -> str:
        return os.environ.get(key, default)

    return NeuralCanvasConfig(
        default_style=_str("NEURAL_CANVAS_DEFAULT_STYLE", "cyberpunk"),
        default_output_dir=_str(
            "NEURAL_CANVAS_DEFAULT_OUTPUT_DIR", "./neural_canvas_output"
        ),
        max_image_width=_int("NEURAL_CANVAS_MAX_IMAGE_WIDTH", 4096),
        max_image_height=_int("NEURAL_CANVAS_MAX_IMAGE_HEIGHT", 4096),
        default_seed=_int("NEURAL_CANVAS_DEFAULT_SEED", 42),
        default_intensity=_float("NEURAL_CANVAS_DEFAULT_INTENSITY", 0.8),
        api_host=_str("NEURAL_CANVAS_API_HOST", "127.0.0.1"),
        api_port=_int("NEURAL_CANVAS_API_PORT", 8080),
        log_level=_str("NEURAL_CANVAS_LOG_LEVEL", "INFO"),
        watermark_text=_str("NEURAL_CANVAS_WATERMARK_TEXT", ""),
    )
