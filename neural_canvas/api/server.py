"""
neural_canvas.api.server
~~~~~~~~~~~~~~~~~~~~~~~~

FastAPI REST API for Neural Canvas.

Start with:
    uvicorn neural_canvas.api.server:app --reload --port 8080

Or via CLI:
    neural-canvas serve --port 8080

Endpoints:
    GET  /health         – health check
    GET  /styles         – list available styles
    POST /generate       – generate styled art from parameters
    POST /transform      – apply filter to uploaded image (multipart)
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from neural_canvas.core.pipeline import StyleTransferPipeline
from neural_canvas.core.filters import (
    GlitchFilter,
    NeonFilter,
    SketchFilter,
    VintageFilter,
    WatercolorFilter,
)
from neural_canvas.utils.image_utils import calculate_image_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Neural Canvas API",
    description=(
        "AI-Powered Artistic Image Generation Pipeline.\n\n"
        "Transform images with style presets or apply individual artistic filters."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton pipeline (initialised at startup)
_pipeline: Optional[StyleTransferPipeline] = None


def get_pipeline() -> StyleTransferPipeline:
    """Return the global pipeline instance, creating it if necessary."""
    global _pipeline
    if _pipeline is None:
        _pipeline = StyleTransferPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic models – Requests
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``.

    Attributes:
        style: Name of the style preset to apply.
        width: Target output width in pixels.
        height: Target output height in pixels.
        seed: Random seed for reproducibility.
        intensity: Effect intensity multiplier in ``[0.1, 2.0]``.
        output_format: Image format for the returned base64 string.
        input_base64: Optional base64-encoded input image.  When omitted a
                      synthetic gradient canvas is generated.
    """

    style: str = Field(default="cyberpunk", description="Style preset name")
    width: int = Field(default=512, ge=64, le=4096, description="Output width px")
    height: int = Field(default=512, ge=64, le=4096, description="Output height px")
    seed: int = Field(default=42, description="Random seed")
    intensity: float = Field(
        default=0.8, ge=0.1, le=2.0, description="Effect intensity"
    )
    output_format: str = Field(default="PNG", description="Output image format")
    input_base64: Optional[str] = Field(
        default=None, description="Base64-encoded input image (optional)"
    )

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"PNG", "JPEG", "WEBP", "BMP"}
        if v.upper() not in allowed:
            raise ValueError(f"output_format must be one of {allowed}")
        return v.upper()

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        return v.lower().strip()


class TransformRequest(BaseModel):
    """Parameters for ``POST /transform`` (form fields alongside the file upload)."""

    filter_name: str = Field(default="neon", description="Filter to apply")
    intensity: float = Field(default=0.8, ge=0.0, le=1.0)
    output_format: str = Field(default="PNG")


# ---------------------------------------------------------------------------
# Pydantic models – Responses
# ---------------------------------------------------------------------------


class StyleInfo(BaseModel):
    """Describes a single available style."""

    name: str
    description: str


class StylesResponse(BaseModel):
    """Response for ``GET /styles``."""

    styles: List[StyleInfo]
    total: int


class GenerateResponse(BaseModel):
    """Response for ``POST /generate``."""

    image_base64: str = Field(description="Base64-encoded output image")
    style_applied: str
    width: int
    height: int
    elapsed_ms: float
    metadata: Dict[str, Any]


class TransformResponse(BaseModel):
    """Response for ``POST /transform``."""

    image_base64: str
    filter_applied: str
    width: int
    height: int
    elapsed_ms: float
    stats: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str
    version: str
    styles_loaded: int


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Serialize a PIL image to a base64 string.

    Args:
        image: PIL image to encode.
        fmt: Image format (``"PNG"``, ``"JPEG"``, etc.).

    Returns:
        Base64-encoded string of the image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _base64_to_image(b64_string: str) -> Image.Image:
    """Decode a base64 string back to a PIL image.

    Args:
        b64_string: Base64-encoded image bytes.

    Returns:
        Decoded PIL image.

    Raises:
        ValueError: If the string cannot be decoded as an image.
    """
    try:
        raw = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc


def _synthetic_canvas(
    width: int, height: int, seed: int = 42
) -> Image.Image:
    """Generate a colourful synthetic gradient canvas.

    This is used as a fallback when no input image is provided to
    ``/generate``.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        seed: Random seed for colour selection.

    Returns:
        A PIL image with a synthetic gradient.
    """
    rng = np.random.default_rng(seed)
    # Pick two random colours
    c1 = rng.integers(0, 256, size=3)
    c2 = rng.integers(0, 256, size=3)

    # Linear gradient across the canvas
    arr = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(width):
        t = i / max(width - 1, 1)
        arr[:, i, :] = c1 * (1 - t) + c2 * t

    # Add vertical gradient component
    c3 = rng.integers(0, 256, size=3)
    for j in range(height):
        t = j / max(height - 1, 1)
        arr[j, :, :] = arr[j, :, :] * (1 - t * 0.4) + c3 * (t * 0.4)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Filter map for /transform
# ---------------------------------------------------------------------------

_FILTER_MAP = {
    "neon": NeonFilter,
    "glitch": GlitchFilter,
    "vintage": VintageFilter,
    "watercolor": WatercolorFilter,
    "sketch": SketchFilter,
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Return the API health status."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        styles_loaded=len(pipeline.list_styles()),
    )


@app.get(
    "/styles",
    response_model=StylesResponse,
    tags=["Styles"],
    summary="List all available art styles",
)
async def list_styles() -> StylesResponse:
    """Return all available style presets with descriptions."""
    pipeline = get_pipeline()
    styles = [
        StyleInfo(name=s["name"], description=s["description"])
        for s in pipeline.list_styles()
    ]
    return StylesResponse(styles=styles, total=len(styles))


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["Generation"],
    summary="Generate artwork using a style preset",
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Apply a style preset to an image (or a synthetic canvas) and return
    the result as a base64-encoded image.

    If ``input_base64`` is provided, the supplied image is used as the source;
    otherwise a colourful synthetic gradient is generated.

    Raises:
        422: If request validation fails.
        400: If the requested style does not exist.
        500: If an unexpected error occurs during processing.
    """
    t_start = time.perf_counter()
    pipeline = get_pipeline()

    # Validate style early for a better error message
    try:
        pipeline.load_style(request.style)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Prepare input image
    if request.input_base64:
        try:
            source = _base64_to_image(request.input_base64)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            )
        source = source.resize((request.width, request.height), Image.LANCZOS)
    else:
        source = _synthetic_canvas(request.width, request.height, seed=request.seed)

    try:
        # Write to a temp in-memory approach: apply style directly
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            source.save(tmp_in.name)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        result = pipeline.generate(
            input_path=tmp_in_path,
            style=request.style,
            output_path=tmp_out_path,
            seed=request.seed,
            intensity=request.intensity,
        )

        output_img = Image.open(tmp_out_path).convert("RGB")
        img_b64 = _image_to_base64(output_img, request.output_format)

        # Cleanup temp files
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)

    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return GenerateResponse(
        image_base64=img_b64,
        style_applied=request.style,
        width=output_img.width,
        height=output_img.height,
        elapsed_ms=elapsed_ms,
        metadata={
            "seed": request.seed,
            "filters_applied": result.filters_applied,
            "intensity": request.intensity,
        },
    )


@app.post(
    "/transform",
    response_model=TransformResponse,
    tags=["Filters"],
    summary="Apply an artistic filter to an uploaded image",
)
async def transform(
    file: UploadFile = File(..., description="Image file to transform"),
    filter_name: str = Form(default="neon", description="Filter name"),
    intensity: float = Form(default=0.8, ge=0.0, le=1.0),
    output_format: str = Form(default="PNG"),
) -> TransformResponse:
    """Upload an image and apply an artistic filter to it.

    Args:
        file: Multipart-uploaded image file.
        filter_name: Name of the filter to apply.
        intensity: Filter intensity (0.0 – 1.0).
        output_format: Desired output format (``"PNG"``, ``"JPEG"``, ``"WEBP"``).

    Returns:
        Base64-encoded filtered image and statistics.

    Raises:
        400: If the filter name is invalid.
        422: If the uploaded file cannot be read as an image.
        500: If processing fails.
    """
    t_start = time.perf_counter()

    if filter_name not in _FILTER_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown filter '{filter_name}'. Available: {list(_FILTER_MAP.keys())}",
        )

    # Read uploaded image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not read image: {exc}",
        )

    # Apply filter
    try:
        filt_cls = _FILTER_MAP[filter_name]
        # Pass intensity to filters that accept it
        import inspect

        sig = inspect.signature(filt_cls.__init__)
        if "intensity" in sig.parameters:
            filt = filt_cls(intensity=intensity)
        else:
            filt = filt_cls()

        result_img = filt.apply(image)
    except Exception as exc:
        logger.exception("Filter application failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Filter failed: {exc}",
        )

    img_b64 = _image_to_base64(result_img, output_format.upper())
    stats = calculate_image_stats(result_img)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return TransformResponse(
        image_base64=img_b64,
        filter_applied=filter_name,
        width=result_img.width,
        height=result_img.height,
        elapsed_ms=elapsed_ms,
        stats=stats,
    )
