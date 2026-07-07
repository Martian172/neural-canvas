"""
Neural Canvas - AI-Powered Artistic Image Generation Pipeline

A modular Python framework for AI-powered image generation,
style transfer, and artistic transformations.

Example:
    >>> from neural_canvas import StyleTransferPipeline
    >>> pipeline = StyleTransferPipeline()
    >>> result = pipeline.generate("photo.jpg", style="cyberpunk", output_path="art.png")
    >>> print(result.output_path)
    art.png
"""

__version__ = "0.1.0"
__author__ = "Neural Canvas Team"
__email__ = "hello@neural-canvas.ai"
__license__ = "MIT"

from neural_canvas.core.pipeline import StyleTransferPipeline, GenerationResult
from neural_canvas.core.filters import (
    GlitchFilter,
    VintageFilter,
    NeonFilter,
    WatercolorFilter,
    SketchFilter,
    BaseFilter,
)
from neural_canvas.core.palette import (
    PaletteExtractor,
    PaletteApplier,
    ColorHarmony,
)
from neural_canvas.config import NeuralCanvasConfig, STYLE_PRESETS

__all__ = [
    # Core pipeline
    "StyleTransferPipeline",
    "GenerationResult",
    # Filters
    "BaseFilter",
    "GlitchFilter",
    "VintageFilter",
    "NeonFilter",
    "WatercolorFilter",
    "SketchFilter",
    # Palette
    "PaletteExtractor",
    "PaletteApplier",
    "ColorHarmony",
    # Config
    "NeuralCanvasConfig",
    "STYLE_PRESETS",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
