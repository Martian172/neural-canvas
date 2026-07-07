# Changelog

All notable changes to Neural Canvas will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

### Added
- Initial release of Neural Canvas 🎨
- `StyleTransferPipeline` with 5 built-in style presets:
  - `cyberpunk` — neon-lit urban dystopia aesthetic
  - `watercolor` — soft, painterly watercolor effect
  - `oil_painting` — rich textured oil painting look
  - `sketch` — pencil sketch / line art conversion
  - `neon` — glowing neon glow effect
- Artistic filter classes: `GlitchFilter`, `VintageFilter`, `NeonFilter`, `WatercolorFilter`, `SketchFilter`
- Color palette extraction and application (`PaletteExtractor`, `PaletteApplier`)
- FastAPI REST API with endpoints for generation and transformation
- CLI with `generate`, `list-styles`, `batch`, and `serve` commands
- Image utilities: resize, grid layout, watermark, format conversion
- Batch generation support
- Full test suite with pytest
- GitHub Actions CI for Python 3.10 and 3.11
