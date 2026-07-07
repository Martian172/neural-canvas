"""
neural_canvas.core - Core processing components

This sub-package contains the central processing logic:
  - pipeline.py  : StyleTransferPipeline orchestrator
  - filters.py   : Artistic image filter implementations
  - palette.py   : Color palette extraction and application
"""

from neural_canvas.core.pipeline import StyleTransferPipeline, GenerationResult
from neural_canvas.core.filters import (
    BaseFilter,
    GlitchFilter,
    VintageFilter,
    NeonFilter,
    WatercolorFilter,
    SketchFilter,
)
from neural_canvas.core.palette import PaletteExtractor, PaletteApplier, ColorHarmony

__all__ = [
    "StyleTransferPipeline",
    "GenerationResult",
    "BaseFilter",
    "GlitchFilter",
    "VintageFilter",
    "NeonFilter",
    "WatercolorFilter",
    "SketchFilter",
    "PaletteExtractor",
    "PaletteApplier",
    "ColorHarmony",
]
