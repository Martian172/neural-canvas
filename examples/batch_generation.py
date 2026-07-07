"""
examples/batch_generation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch-process a directory of images with a single style preset using
StyleTransferPipeline.batch_generate (multi-threaded).

Run with:
    python examples/batch_generation.py
"""
import sys
from pathlib import Path

# Make the package importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from neural_canvas import StyleTransferPipeline

OUTPUT_ROOT = Path(__file__).parent / "output"
INPUT_DIR = OUTPUT_ROOT / "batch_inputs"
GALLERY_DIR = OUTPUT_ROOT / "batch_gallery"


def make_sample_images(n: int = 6, size: int = 320) -> None:
    """Create a handful of colourful gradient test images."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(n):
        c1 = rng.integers(30, 225, size=3).astype(np.float32)
        c2 = rng.integers(30, 225, size=3).astype(np.float32)
        t = np.linspace(0, 1, size, dtype=np.float32)[None, :, None]
        arr = c1 * (1 - t) + c2 * t
        arr = np.broadcast_to(arr, (size, size, 3)).astype(np.uint8)
        Image.fromarray(arr).save(INPUT_DIR / f"sample_{i}.png")
    print(f"Created {n} sample images in {INPUT_DIR}")


def main() -> None:
    make_sample_images()

    pipeline = StyleTransferPipeline()
    style = "watercolor"

    print(f"Batch-generating '{style}' versions with 4 worker threads ...")
    results = pipeline.batch_generate(
        input_dir=INPUT_DIR,
        style=style,
        output_dir=GALLERY_DIR,
        max_workers=4,
        seed=42,
    )

    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_ms = sum(r.elapsed_ms for r in ok)

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed")
    print(f"Total processing time: {total_ms:.0f}ms "
          f"(avg {total_ms / max(len(ok), 1):.0f}ms per image)")
    print(f"Gallery saved to: {GALLERY_DIR}")
    for r in failed:
        print(f"  FAILED {r.input_path}: {r.error}")


if __name__ == "__main__":
    main()
