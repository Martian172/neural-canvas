"""Tests for neural_canvas.core.pipeline module."""
import numpy as np
import pytest
from PIL import Image

from neural_canvas.core.pipeline import GenerationResult, StyleTransferPipeline


@pytest.fixture
def sample_image_path(tmp_path):
    """Save a random test image to disk and return its path."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / "sample.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def pipeline():
    return StyleTransferPipeline()


class TestStyleTransferPipeline:
    def test_init(self, pipeline):
        assert pipeline is not None
        assert len(pipeline.list_styles()) > 0

    def test_list_styles(self, pipeline):
        styles = pipeline.list_styles()
        assert isinstance(styles, list)
        names = [s["name"] for s in styles]
        assert "cyberpunk" in names
        assert "watercolor" in names
        assert "sketch" in names

    def test_load_style(self, pipeline):
        config = pipeline.load_style("cyberpunk")
        assert config is not None
        assert "filters" in config

    def test_load_unknown_style(self, pipeline):
        with pytest.raises(KeyError):
            pipeline.load_style("nonexistent_style_xyz")

    @pytest.mark.parametrize("style", ["sketch", "watercolor", "neon", "cyberpunk"])
    def test_generate(self, pipeline, sample_image_path, tmp_path, style):
        out_path = tmp_path / f"out_{style}.png"
        result = pipeline.generate(
            input_path=sample_image_path, style=style, output_path=out_path
        )
        assert isinstance(result, GenerationResult)
        assert result.success
        assert out_path.exists()
        output = Image.open(out_path)
        assert output.size == (256, 256)

    def test_apply_filter_invalid(self, pipeline, sample_image_path):
        image = Image.open(sample_image_path)
        with pytest.raises((ValueError, KeyError)):
            pipeline.apply_filter(image, "definitely_not_a_filter")

    def test_batch_generate(self, pipeline, tmp_path):
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        for i in range(2):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(input_dir / f"img_{i}.png")

        results = pipeline.batch_generate(
            input_dir=input_dir, style="sketch", output_dir=tmp_path / "outputs"
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, GenerationResult)
            assert r.success
