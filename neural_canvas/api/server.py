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
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
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

_HOME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Neural Canvas — AI Art Generation</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#080812;--s1:#0f0f20;--s2:#161628;--border:#252545;--p:#8b5cf6;--p2:#a78bfa;--cyan:#22d3ee;--pink:#ec4899;--green:#10b981;--text:#e2e8f0;--muted:#64748b}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
/* Animated bg */
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 80% 60% at 20% 20%,rgba(139,92,246,.15) 0,transparent 60%),radial-gradient(ellipse 60% 50% at 80% 80%,rgba(34,211,238,.1) 0,transparent 60%);pointer-events:none}
nav{display:flex;align-items:center;justify-content:space-between;padding:.9rem 2rem;background:rgba(15,15,32,.8);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
.logo{font-weight:900;font-size:1.1rem;background:linear-gradient(135deg,var(--p2),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-links{display:flex;gap:1rem}
.nav-links a{color:var(--muted);text-decoration:none;font-size:.85rem;padding:.35rem .75rem;border-radius:8px;transition:.2s}
.nav-links a:hover{background:var(--s2);color:var(--text)}
.btn{padding:.5rem 1.2rem;border-radius:9px;border:none;cursor:pointer;font-weight:600;font-size:.85rem;transition:.2s}
.btn-primary{background:linear-gradient(135deg,var(--p),var(--pink));color:#fff;box-shadow:0 0 20px rgba(139,92,246,.3)}
.btn-primary:hover{opacity:.85;transform:scale(1.04)}
.btn-ghost{background:var(--s2);color:var(--text);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--p2);color:var(--p2)}
/* Hero */
.hero{text-align:center;padding:5rem 2rem 3rem}
.badge{display:inline-block;padding:.3rem .9rem;background:rgba(139,92,246,.15);border:1px solid rgba(139,92,246,.4);border-radius:999px;font-size:.75rem;color:var(--p2);margin-bottom:1.5rem;font-weight:600;letter-spacing:.05em}
h1{font-size:clamp(2.5rem,6vw,4.5rem);font-weight:900;line-height:1.1;margin-bottom:1.2rem}
h1 span{background:linear-gradient(135deg,var(--p2),var(--cyan),var(--pink));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-sub{color:var(--muted);font-size:1.15rem;max-width:580px;margin:0 auto 2.5rem;line-height:1.7}
.hero-cta{display:flex;gap:1rem;justify-content:center;flex-wrap:wrap}
/* Styles grid */
.section{padding:4rem 2rem;max-width:1200px;margin:0 auto}
.section-title{font-size:1.5rem;font-weight:700;margin-bottom:.5rem}
.section-sub{color:var(--muted);margin-bottom:2rem;font-size:.9rem}
.styles-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}
.style-card{background:var(--s1);border:1px solid var(--border);border-radius:14px;padding:1.2rem;cursor:pointer;transition:.25s;position:relative;overflow:hidden}
.style-card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--p),var(--cyan));opacity:0;transition:.25s}
.style-card:hover{border-color:var(--p2);transform:translateY(-4px)}
.style-card:hover::before{opacity:.06}
.style-card.active{border-color:var(--p);box-shadow:0 0 0 1px var(--p),0 8px 30px rgba(139,92,246,.2)}
.style-icon{font-size:1.8rem;margin-bottom:.6rem}
.style-name{font-weight:700;font-size:.9rem;margin-bottom:.25rem;text-transform:capitalize}
.style-desc{font-size:.72rem;color:var(--muted);line-height:1.5}
/* Generator */
.gen-panel{background:var(--s1);border:1px solid var(--border);border-radius:18px;padding:2rem;margin-top:3rem}
.gen-grid{display:grid;grid-template-columns:1fr 1fr;gap:2rem}
@media(max-width:700px){.gen-grid{grid-template-columns:1fr}}
.field-label{font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem}
.field-row{margin-bottom:1.2rem}
input[type=range]{width:100%;accent-color:var(--p);cursor:pointer}
.range-val{font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--p2)}
.output-area{display:flex;align-items:center;justify-content:center;background:var(--s2);border:1px solid var(--border);border-radius:12px;min-height:340px;position:relative;overflow:hidden}
.output-area img{max-width:100%;max-height:400px;border-radius:10px;display:block}
.placeholder-text{color:var(--muted);font-size:.9rem;text-align:center;padding:2rem}
.spinner{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--p2);border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 1rem}
@keyframes spin{to{transform:rotate(360deg)}}
.gen-btn{width:100%;padding:.8rem;border-radius:11px;background:linear-gradient(135deg,var(--p),var(--pink));border:none;color:#fff;font-weight:700;font-size:1rem;cursor:pointer;margin-top:1rem;transition:.2s;box-shadow:0 0 25px rgba(139,92,246,.25)}
.gen-btn:hover{opacity:.88;transform:scale(1.02)}
.gen-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
/* Endpoint list */
.ep-list{display:flex;flex-direction:column;gap:.75rem}
.ep{display:flex;align-items:center;gap:.8rem;background:var(--s1);border:1px solid var(--border);border-radius:10px;padding:.75rem 1.1rem}
.method{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:700;padding:.2rem .55rem;border-radius:5px;min-width:46px;text-align:center}
.GET{background:rgba(16,185,129,.15);color:var(--green)}
.POST{background:rgba(139,92,246,.15);color:var(--p2)}
.ep-path{font-family:'JetBrains Mono',monospace;font-size:.82rem;color:var(--cyan)}
.ep-desc{font-size:.78rem;color:var(--muted);margin-left:auto}
footer{text-align:center;padding:2rem;color:var(--muted);font-size:.8rem;border-top:1px solid var(--border);margin-top:4rem}
</style>
</head>
<body>
<nav>
  <div class="logo">🎨 Neural Canvas</div>
  <div class="nav-links">
    <a href="/docs">API Docs</a>
    <a href="/styles">Styles JSON</a>
    <a href="/health">Health</a>
  </div>
  <a href="/docs" class="btn btn-primary">Try API</a>
</nav>

<div class="hero">
  <div class="badge">v0.1.0 — Open Source AI Art</div>
  <h1>Transform Images with<br/><span>AI-Powered Style</span></h1>
  <p class="hero-sub">Apply stunning artistic styles, glitch effects, neon glows, watercolor washes & more — all via a simple REST API or Python SDK.</p>
  <div class="hero-cta">
    <button class="btn btn-primary" onclick="document.getElementById('generator').scrollIntoView({behavior:'smooth'})">Generate Art</button>
    <a href="/docs" class="btn btn-ghost">API Explorer</a>
  </div>
</div>

<div class="section">
  <div class="section-title">Available Styles</div>
  <div class="section-sub">Choose a preset — each applies real image transformations via numpy & Pillow.</div>
  <div class="styles-grid" id="styles-grid">Loading styles...</div>
</div>

<div class="section" id="generator">
  <div class="section-title">Generate Art</div>
  <div class="section-sub">Pick a style above, adjust parameters, and click Generate.</div>
  <div class="gen-panel">
    <div class="gen-grid">
      <div>
        <div class="field-row">
          <div class="field-label">Selected Style</div>
          <div id="selected-style-name" style="font-weight:700;color:var(--p2);font-size:1rem">cyberpunk</div>
        </div>
        <div class="field-row">
          <div class="field-label">Width <span class="range-val" id="w-val">512</span>px</div>
          <input type="range" min="128" max="1024" step="64" value="512" oninput="document.getElementById('w-val').textContent=this.value" id="inp-w"/>
        </div>
        <div class="field-row">
          <div class="field-label">Height <span class="range-val" id="h-val">512</span>px</div>
          <input type="range" min="128" max="1024" step="64" value="512" oninput="document.getElementById('h-val').textContent=this.value" id="inp-h"/>
        </div>
        <div class="field-row">
          <div class="field-label">Intensity <span class="range-val" id="int-val">0.8</span></div>
          <input type="range" min="0.1" max="2.0" step="0.1" value="0.8" oninput="document.getElementById('int-val').textContent=parseFloat(this.value).toFixed(1)" id="inp-int"/>
        </div>
        <div class="field-row">
          <div class="field-label">Seed</div>
          <input type="range" min="1" max="999" step="1" value="42" oninput="document.getElementById('seed-val').textContent=this.value" id="inp-seed"/>
          <span class="range-val" id="seed-val">42</span>
        </div>
        <button class="gen-btn" id="gen-btn" onclick="generateArt()">Generate Art</button>
      </div>
      <div>
        <div class="field-label" style="margin-bottom:.75rem">Output</div>
        <div class="output-area" id="output-area">
          <div class="placeholder-text">Your generated art will appear here.<br/>Select a style and click Generate.</div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title">REST API Endpoints</div>
  <div class="section-sub">Full OpenAPI docs at <a href="/docs" style="color:var(--p2)">/docs</a></div>
  <div class="ep-list">
    <div class="ep"><span class="method GET">GET</span><span class="ep-path">/health</span><span class="ep-desc">Health check & version</span></div>
    <div class="ep"><span class="method GET">GET</span><span class="ep-path">/styles</span><span class="ep-desc">List all art style presets</span></div>
    <div class="ep"><span class="method POST">POST</span><span class="ep-path">/generate</span><span class="ep-desc">Generate art from style + params (returns base64 PNG)</span></div>
    <div class="ep"><span class="method POST">POST</span><span class="ep-path">/transform</span><span class="ep-desc">Apply filter to an uploaded image</span></div>
  </div>
</div>

<footer>Neural Canvas v0.1.0 · MIT License · <a href="https://github.com/Martian172/neural-canvas" style="color:var(--p2)">github.com/Martian172/neural-canvas</a></footer>

<script>
let selectedStyle = 'cyberpunk';

async function loadStyles() {
  const grid = document.getElementById('styles-grid');
  try {
    const r = await fetch('/styles');
    const data = await r.json();
    const icons = {cyberpunk:'⚡',watercolor:'💧',oil_painting:'🎨',sketch:'✏️',neon:'🌟',glitch:'👾',vintage:'📷',abstract:'🌀'};
    grid.innerHTML = data.styles.map(s => `
      <div class="style-card${s.name===selectedStyle?' active':''}" onclick="selectStyle('${s.name}',this)">
        <div class="style-icon">${icons[s.name]||'🖼️'}</div>
        <div class="style-name">${s.name.replace(/_/g,' ')}</div>
        <div class="style-desc">${s.description}</div>
      </div>`).join('');
  } catch(e) { grid.innerHTML = '<p style="color:var(--muted)">Could not load styles.</p>'; }
}

function selectStyle(name, el) {
  selectedStyle = name;
  document.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('selected-style-name').textContent = name.replace(/_/g,' ');
}

async function generateArt() {
  const btn = document.getElementById('gen-btn');
  const out = document.getElementById('output-area');
  btn.disabled = true;
  out.innerHTML = '<div><div class="spinner"></div><p style="color:var(--muted);font-size:.85rem">Generating...</p></div>';
  try {
    const payload = {
      style: selectedStyle,
      width: parseInt(document.getElementById('inp-w').value),
      height: parseInt(document.getElementById('inp-h').value),
      intensity: parseFloat(document.getElementById('inp-int').value),
      seed: parseInt(document.getElementById('inp-seed').value),
      output_format: 'PNG'
    };
    const r = await fetch('/generate', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await r.json();
    if (data.image_base64) {
      out.innerHTML = `<img src="data:image/png;base64,${data.image_base64}" alt="Generated ${selectedStyle} art"/>`;
    } else {
      out.innerHTML = '<div class="placeholder-text" style="color:#ef4444">Generation failed: ' + (data.detail||'unknown error') + '</div>';
    }
  } catch(e) {
    out.innerHTML = '<div class="placeholder-text" style="color:#ef4444">Request failed: ' + e.message + '</div>';
  }
  btn.disabled = false;
}

loadStyles();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage():
    """Serve the Neural Canvas web UI."""
    return HTMLResponse(_HOME_HTML)


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


if __name__ == "__main__":
    # Allows launching the API directly (e.g. VS Code Run button):
    #   python neural_canvas/api/server.py
    import os

    import uvicorn

    host = os.environ.get("NEURAL_CANVAS_HOST", "127.0.0.1")
    port = int(os.environ.get("NEURAL_CANVAS_PORT", "8002"))
    print(f"Neural Canvas API + web UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
