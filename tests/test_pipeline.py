"""Tests for neural_canvas.core.pipeline module."""
import pytest
from PIL import Image
import numpy as np
from neural_canvas.core.pipeline import StyleTransferPipeline
from neural_canvas.core.filters import (
    GlitchFilter, VintageFilter, NeonFilter, WatercolorFilter, SketchFilter
)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def pipeline():
    return StyleTransferPipeline()


class TestStyleTransferPipeline:
    def test_init(self, pipeline):
        assert pipeline is not None
        assert len(pipeline.styles) > 0

    def test_list_styles(self, pipeline):
        styles = pipeline.list_styles()
        assert isinstance(styles, list)
        assert "cyberpunk" in styles
        assert "watercolor" in styles
        assert "sketch" in styles

    def test_load_style(self, pipeline):
        config = pipeline.load_style("cyberpunk")
        assert config is not None
        assert "filter" in config

    def test_load_unknown_style(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.load_style("nonexistent_style_xyz")

    def test_generate(self, pipeline, sample_image):
        result = pipeline.generate(sample_image, style="sketch")
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_generate_watercolor(self, pipeline, sample_image):
        result = pipeline.generate(sample_image, style="watercolor")
        assert isinstance(result, Image.Image)

    def test_generate_neon(self, pipeline, sample_image):
        result = pipeline.generate(sample_image, style="neon")
        assert isinstance(result, Image.Image)

    def test_generate_cyberpunk(self, pipeline, sample_image):
        result = pipeline.generate(sample_image, style="cyberpunk")
        assert isinstance(result, Image.Image)

    def test_apply_filter_invalid(self, pipeline, sample_image):
        with pytest.raises((ValueError, KeyError)):
            pipeline.apply_filter(sample_image, "definitely_not_a_filter")

    def test_batch_generate(self, pipeline, sample_image):
        images = [sample_image, sample_image]
        results = pipeline.batch_generate(images, style="sketch")
        assert len(results) == 2
        for r in results:
            assert isinstance(r, Image.Image)
