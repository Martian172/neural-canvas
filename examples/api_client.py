"""
examples/api_client.py
~~~~~~~~~~~~~~~~~~~~~~

Call the Neural Canvas REST API from Python.

Start the server first (in another terminal):
    python -m uvicorn neural_canvas.api.server:app --port 8002

Then run:
    python examples/api_client.py
"""
import base64
import sys
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8002"
OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    # 1. Health check
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5).json()
    except requests.ConnectionError:
        sys.exit(
            f"Could not reach {BASE_URL} — start the server first:\n"
            "  python -m uvicorn neural_canvas.api.server:app --port 8002"
        )
    print(f"Server healthy: v{health['version']}, {health['styles_loaded']} styles loaded")

    # 2. List available styles
    styles = requests.get(f"{BASE_URL}/styles", timeout=10).json()
    print("\nAvailable styles:")
    for s in styles["styles"]:
        print(f"  - {s['name']}: {s['description']}")

    # 3. Generate art (synthetic canvas — no input image needed)
    payload = {
        "style": "cyberpunk",
        "width": 512,
        "height": 512,
        "seed": 42,
        "intensity": 0.9,
    }
    print(f"\nGenerating {payload['style']} art ...")
    resp = requests.post(f"{BASE_URL}/generate", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # 4. Decode the base64 image and save it
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "api_generated.png"
    out_path.write_bytes(base64.b64decode(data["image_base64"]))
    print(f"Saved {data['width']}x{data['height']} image to {out_path}")
    print(f"Server processing time: {data['elapsed_ms']:.0f}ms")


if __name__ == "__main__":
    main()
