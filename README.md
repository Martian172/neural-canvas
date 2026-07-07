# 🎨 Neural Canvas

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/neural-canvas/)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](https://github.com/yourusername/neural-canvas/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Neural Canvas — AI-Powered Artistic Image Generation Pipeline**

*Transform ordinary images into extraordinary masterpieces with neural style transfer, artistic filters, and a plug-and-play pipeline.*

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API](#-rest-api) • [CLI](#-cli) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖌️ **Style Transfer** | Apply iconic art styles: cyberpunk, watercolor, oil painting, sketch, neon, and more |
| ⚡ **Batch Generation** | Process hundreds of images in parallel with progress tracking |
| 🌐 **REST API** | FastAPI-powered server for integrating into any application |
| 🖥️ **CLI** | Beautiful terminal interface with emoji feedback |
| 🔌 **Plugin System** | Register custom filters and styles at runtime |
| 🎨 **Palette Tools** | Extract dominant colors, apply harmonious color schemes |
| 📊 **Image Statistics** | Analyze brightness, contrast, entropy of generated art |
| 🐳 **Docker Ready** | Containerized deployment out of the box |

---

## 🚀 Quick Start

### Installation

```bash
pip install neural-canvas
```

Or install from source:

```bash
git clone https://github.com/yourusername/neural-canvas.git
cd neural-canvas
pip install -e ".[dev]"
```

### Generate Your First Artwork

```python
from neural_canvas import StyleTransferPipeline

# Initialize pipeline
pipeline = StyleTransferPipeline()

# Generate cyberpunk art
result = pipeline.generate(
    input_path="photo.jpg",
    style="cyberpunk",
    output_path="artwork.png"
)
print(f"Artwork saved to: {result.output_path}")
print(f"   Style applied: {result.style_name}")
print(f"   Processing time: {result.elapsed_ms:.1f}ms")
```

### Apply a Filter

```python
from neural_canvas.core.filters import NeonFilter, GlitchFilter
from PIL import Image

img = Image.open("photo.jpg")

# Chain multiple filters
neon = NeonFilter(intensity=0.8, color_boost=1.5)
glitch = GlitchFilter(strength=0.3, bands=12)

result = glitch.apply(neon.apply(img))
result.save("neon_glitch.png")
```

### Batch Generation

```python
from neural_canvas import StyleTransferPipeline

pipeline = StyleTransferPipeline()

results = pipeline.batch_generate(
    input_dir="./photos/",
    style="watercolor",
    output_dir="./watercolor_art/",
    max_workers=4
)

print(f"Processed {len(results)} images")
```

### CLI Usage

```bash
# Generate a single artwork
neural-canvas generate photo.jpg --style cyberpunk --output art.png

# List available styles
neural-canvas list-styles

# Batch process a directory
neural-canvas batch ./photos/ --style oil_painting --output ./output/

# Start the API server
neural-canvas serve --host 0.0.0.0 --port 8080
```

---

## Architecture

```
neural-canvas/
|
+-- StyleTransferPipeline      <- Core orchestrator
|         |
|         +-- Filters           <- Artistic transformations
|         |     +-- GlitchFilter
|         |     +-- VintageFilter
|         |     +-- NeonFilter
|         |     +-- WatercolorFilter
|         |     +-- SketchFilter
|         |
|         +-- Palette           <- Color intelligence
|         |     +-- PaletteExtractor
|         |     +-- PaletteApplier
|         |     +-- ColorHarmony
|         |
|         +-- Config            <- Style presets & settings
|
+-- FastAPI Server             <- REST API layer
|         +-- POST /generate
|         +-- POST /transform
|         +-- GET  /styles
|         +-- GET  /health
|
+-- Click CLI                  <- Terminal interface
          +-- generate
          +-- list-styles
          +-- batch
          +-- serve
```

### Processing Flow

```
Input Image
    |
    v
[  Pre-processing  ]  <- Resize, normalize, validate
    |
    v
[ Style Preset Load]  <- Load parameters for chosen style
    |
    v
[  Filter Chain    ]  <- Apply ordered filter stack
    |
    v
[ Post-processing  ]  <- Sharpen, watermark, format convert
    |
    v
Output Image
```

---

## Available Styles

| Style | Description | Best For |
|---|---|---|
| `cyberpunk` | Neon glows, dark backgrounds, electric blues and magentas | Urban photography |
| `watercolor` | Soft edges, color bleeding, paper texture | Landscapes, portraits |
| `oil_painting` | Thick brush strokes, rich colors, impasto effect | Any subject |
| `sketch` | Pencil lines, grayscale detail, cross-hatching | Architecture, still life |
| `neon` | Saturated highlights, dark shadows, glow effects | Night shots, abstracts |
| `vintage` | Sepia tone, grain, vignette, faded colors | Portraits, street photography |
| `glitch` | Digital corruption artifacts, color channel separation | Digital art |
| `comic` | Bold outlines, flat colors, halftone dots | Characters, action shots |

---

## REST API

Start the server:

```bash
neural-canvas serve --port 8080
# or
uvicorn neural_canvas.api.server:app --reload
```

### Endpoints

#### `POST /generate`

```json
{
  "style": "cyberpunk",
  "width": 1024,
  "height": 768,
  "seed": 42,
  "intensity": 0.85,
  "output_format": "png"
}
```

Response:
```json
{
  "image_base64": "iVBORw0KGgo...",
  "style_applied": "cyberpunk",
  "width": 1024,
  "height": 768,
  "elapsed_ms": 312.5,
  "metadata": { "seed": 42, "filters_applied": ["neon", "color_grade"] }
}
```

#### `POST /transform`

Upload an image and apply a transformation filter.

#### `GET /styles`

Returns all available styles with metadata.

#### `GET /health`

Health check endpoint.

### API Client Example

```python
import requests

resp = requests.post("http://localhost:8080/generate", json={
    "style": "watercolor",
    "width": 800,
    "height": 600,
    "seed": 123,
})

data = resp.json()
print(f"Generated in {data['elapsed_ms']}ms")
```

---

## CLI

```
Usage: neural-canvas [OPTIONS] COMMAND [ARGS]...

  Neural Canvas - AI-Powered Artistic Image Generation

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  generate     Generate artwork from an input image
  list-styles  List all available art styles
  batch        Batch process a directory of images
  serve        Start the Neural Canvas API server
```

### `generate` Command

```bash
neural-canvas generate INPUT_IMAGE [OPTIONS]

Options:
  --style TEXT           Art style to apply [default: cyberpunk]
  --output TEXT          Output file path [default: output.png]
  --width INTEGER        Output width in pixels [default: 1024]
  --height INTEGER       Output height in pixels [default: 768]
  --seed INTEGER         Random seed for reproducibility [default: 42]
  --intensity FLOAT      Effect intensity (0.0-1.0) [default: 0.8]
  --format [png|jpg|webp] Output format [default: png]
```

### `batch` Command

```bash
neural-canvas batch INPUT_DIR [OPTIONS]

Options:
  --style TEXT        Style to apply to all images
  --output TEXT       Output directory
  --workers INTEGER   Number of parallel workers [default: 4]
  --recursive         Process subdirectories recursively
```

---

## Generated Art Styles

> *Neural Canvas supports the following artistic transformations, each with fine-tuned parameters for stunning results:*

- **Cyberpunk**: Electric neon grids, rain-soaked streets effect, ultra-violet palette
- **Watercolor**: Wet-on-wet diffusion, color bleeds, paper grain texture overlay
- **Oil Painting**: Thick impasto strokes, rich saturation, Rembrandt-style lighting
- **Sketch**: Fine pencil lines, cross-hatching shadows, negative space preservation
- **Neon Noir**: Black background, saturated neon outlines, moody atmosphere

---

## Examples

See the [`examples/`](examples/) directory for complete scripts:

- [`basic_usage.py`](examples/basic_usage.py) - Single image generation
- [`batch_generation.py`](examples/batch_generation.py) - Process many images
- [`api_client.py`](examples/api_client.py) - Use the REST API

---

## Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=neural_canvas --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
make install-dev

# Run linter
make lint

# Run tests
make test
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Pillow](https://python-pillow.org/) for image processing
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API layer
- [Click](https://click.palletsprojects.com/) for the CLI framework
- [SciPy](https://scipy.org/) for signal processing filters

---

<div align="center">
Made with love by the Neural Canvas team

Star this repo if you find it useful!
</div>
