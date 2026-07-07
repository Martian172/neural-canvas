"""Tests for neural_canvas.core.filters module."""
import pytest
from PIL import Image
import numpy as np
from neural_canvas.core.filters import (
    GlitchFilter, VintageFilter, NeonFilter, WatercolorFilter, SketchFilter
)


@pytest.fixture
def sample_image():
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestGlitchFilter:
    def test_apply(self, sample_image):
        f = GlitchFilter()
        result = f.apply(sample_image)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_apply_with_strength(self, sample_image):
        f = GlitchFilter(strength=0.8)
        result = f.apply(sample_image)
        assert isinstance(result, Image.Image)


class TestVintageFilter:
    def test_apply(self, sample_image):
        f = VintageFilter()
        result = f.apply(sample_image)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size


class TestNeonFilter:
    def test_apply(self, sample_image):
        f = NeonFilter()
        result = f.apply(sample_image)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size


class TestWatercolorFilter:
    def test_apply(self, sample_image):
        f = WatercolorFilter()
        result = f.apply(sample_image)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size


class TestSketchFilter:
    def test_apply(self, sample_image):
        f = SketchFilter()
        result = f.apply(sample_image)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size

    def test_apply_with_tint(self, sample_image):
        f = SketchFilter(color_tint=(200, 180, 150))
        result = f.apply(sample_image)
        assert isinstance(result, Image.Image)
