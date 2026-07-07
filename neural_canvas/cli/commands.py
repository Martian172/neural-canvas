"""
neural_canvas.cli.commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click-based command-line interface for Neural Canvas.

Commands:
    generate     – Apply a style to a single image.
    list-styles  – Display all available styles.
    batch        – Process an entire directory of images.
    serve        – Launch the FastAPI development server.

Usage::

    neural-canvas --help
    neural-canvas generate photo.jpg --style cyberpunk --output art.png
    neural-canvas list-styles
    neural-canvas batch ./photos/ --style watercolor --output ./wc_art/
    neural-canvas serve --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import click

# Windows consoles default to a legacy codepage (e.g. cp1252) that cannot
# encode the emoji used in CLI output; force UTF-8 with replacement so
# output never crashes regardless of terminal settings.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STYLES_EMOJI = {
    "cyberpunk": "🌆",
    "watercolor": "🎨",
    "oil_painting": "🖼️",
    "sketch": "✏️",
    "neon": "💡",
    "vintage": "📷",
    "glitch": "⚡",
    "comic": "💥",
}


def _style_emoji(style: str) -> str:
    return STYLES_EMOJI.get(style, "🎭")


def _print_banner() -> None:
    """Print the Neural Canvas ASCII banner."""
    banner = click.style(
        r"""
  _   _                      _    ____
 | \ | | ___ _   _ _ __ __ _| |  / ___|__ _ _ ____   ____ _ ___
 |  \| |/ _ \ | | | '__/ _` | | | |   / _` | '_ \ \ / / _` / __|
 | |\  |  __/ |_| | | | (_| | | | |__| (_| | | | \ V / (_| \__ \
 |_| \_|\___|\__,_|_|  \__,_|_|  \____\__,_|_| |_|\_/ \__,_|___/
""",
        fg="cyan",
        bold=True,
    )
    subtitle = click.style(
        "  AI-Powered Artistic Image Generation Pipeline  v0.1.0\n",
        fg="bright_blue",
    )
    click.echo(banner)
    click.echo(subtitle)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="neural-canvas")
def cli() -> None:
    """Neural Canvas - AI-Powered Artistic Image Generation Pipeline.

    Transform ordinary images into extraordinary artwork using neural style
    transfer, artistic filters, and a powerful batch processing engine.
    """
    pass


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cli.command("generate")
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--style",
    "-s",
    default="cyberpunk",
    show_default=True,
    help="Art style preset to apply.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path. Defaults to <input_stem>_<style>.png",
)
@click.option(
    "--width",
    "-w",
    default=None,
    type=int,
    help="Resize output to this width (px).",
)
@click.option(
    "--height",
    "-h",
    default=None,
    type=int,
    help="Resize output to this height (px).",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--intensity",
    default=0.8,
    show_default=True,
    type=float,
    help="Effect intensity [0.0 - 2.0].",
)
@click.option(
    "--format",
    "output_format",
    default="png",
    show_default=True,
    type=click.Choice(["png", "jpg", "webp", "bmp"], case_sensitive=False),
    help="Output image format.",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def generate_cmd(
    input_image: str,
    style: str,
    output: Optional[str],
    width: Optional[int],
    height: Optional[int],
    seed: int,
    intensity: float,
    output_format: str,
    verbose: bool,
) -> None:
    """Generate artwork from INPUT_IMAGE using a style preset.

    \b
    Examples:
        neural-canvas generate photo.jpg --style cyberpunk
        neural-canvas generate photo.jpg --style watercolor --output wc.png
        neural-canvas generate photo.jpg --style sketch --width 1024 --seed 7
    """
    _print_banner()

    from neural_canvas import StyleTransferPipeline

    emoji = _style_emoji(style)
    click.echo(
        f"  {emoji}  Applying style: {click.style(style, fg='yellow', bold=True)}"
    )
    click.echo(f"  📂  Input: {click.style(input_image, fg='cyan')}")

    # Determine output path
    if output is None:
        inp = Path(input_image)
        output = str(inp.parent / f"{inp.stem}_{style}.{output_format}")

    click.echo(f"  💾  Output: {click.style(output, fg='cyan')}")
    click.echo()

    pipeline = StyleTransferPipeline()

    with click.progressbar(
        length=1,
        label=click.style("  Processing", fg="bright_cyan"),
        show_eta=False,
        fill_char=click.style("█", fg="cyan"),
        empty_char="░",
    ) as bar:
        try:
            result = pipeline.generate(
                input_path=input_image,
                style=style,
                output_path=output,
                width=width,
                height=height,
                seed=seed,
                intensity=intensity,
            )
            bar.update(1)
        except KeyError as exc:
            click.echo(f"\n  ❌  Error: {exc}", err=True)
            sys.exit(1)
        except FileNotFoundError as exc:
            click.echo(f"\n  ❌  Error: {exc}", err=True)
            sys.exit(1)

    click.echo()
    click.echo(
        click.style("  ✅  Artwork generated successfully!", fg="green", bold=True)
    )
    click.echo(f"  ⏱️   Time: {result.elapsed_ms:.1f}ms")
    click.echo(f"  🎨  Filters applied: {', '.join(result.filters_applied) or 'none'}")

    if verbose:
        click.echo()
        click.echo("  Metadata:")
        for k, v in result.metadata.items():
            click.echo(f"    {k}: {v}")

    click.echo()
    click.echo(f"  Saved to: {click.style(result.output_path, fg='green', bold=True)}")


# ---------------------------------------------------------------------------
# list-styles
# ---------------------------------------------------------------------------


@cli.command("list-styles")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON.")
def list_styles_cmd(json_output: bool) -> None:
    """List all available art style presets.

    \b
    Example:
        neural-canvas list-styles
        neural-canvas list-styles --json-output
    """
    from neural_canvas import StyleTransferPipeline

    pipeline = StyleTransferPipeline()
    styles = pipeline.list_styles()

    if json_output:
        import json
        click.echo(json.dumps(styles, indent=2))
        return

    _print_banner()
    click.echo(click.style("  Available Style Presets\n", fg="yellow", bold=True))

    max_name_len = max(len(s["name"]) for s in styles)
    divider = "  " + "-" * (max_name_len + 50)
    click.echo(divider)

    header = (
        "  "
        + click.style("Style".ljust(max_name_len + 2), fg="cyan", bold=True)
        + click.style("Description", fg="cyan", bold=True)
    )
    click.echo(header)
    click.echo(divider)

    for s in styles:
        emoji = _style_emoji(s["name"])
        name_col = click.style(
            f"{emoji}  {s['name']}".ljust(max_name_len + 6), fg="yellow"
        )
        desc_col = s["description"]
        click.echo(f"  {name_col}  {desc_col}")

    click.echo(divider)
    click.echo(f"\n  Total: {click.style(str(len(styles)), fg='green', bold=True)} styles\n")
    click.echo(
        "  Usage: "
        + click.style("neural-canvas generate image.jpg --style <name>", fg="cyan")
    )


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@cli.command("batch")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--style", "-s", default="cyberpunk", show_default=True, help="Style to apply.")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory. Defaults to <input_dir>_<style>/",
)
@click.option(
    "--workers",
    "-w",
    default=4,
    show_default=True,
    type=int,
    help="Number of parallel worker threads.",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Recurse into subdirectories.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Base random seed.",
)
@click.option("--intensity", default=0.8, show_default=True, type=float)
def batch_cmd(
    input_dir: str,
    style: str,
    output: Optional[str],
    workers: int,
    recursive: bool,
    seed: int,
    intensity: float,
) -> None:
    """Batch-process all images in INPUT_DIR.

    \b
    Examples:
        neural-canvas batch ./photos/ --style watercolor
        neural-canvas batch ./photos/ --style neon --output ./neon_art/ --workers 8
        neural-canvas batch ./photos/ --recursive --style vintage
    """
    from neural_canvas import StyleTransferPipeline

    _print_banner()
    emoji = _style_emoji(style)
    click.echo(
        f"  {emoji}  Batch processing: {click.style(style, fg='yellow', bold=True)} style"
    )
    click.echo(f"  📂  Input dir:  {click.style(input_dir, fg='cyan')}")
    if output:
        click.echo(f"  💾  Output dir: {click.style(output, fg='cyan')}")
    click.echo(f"  👷  Workers: {workers}")
    click.echo(f"  🔁  Recursive: {'Yes' if recursive else 'No'}")
    click.echo()

    pipeline = StyleTransferPipeline()

    t_start = time.perf_counter()
    results = pipeline.batch_generate(
        input_dir=input_dir,
        style=style,
        output_dir=output,
        max_workers=workers,
        recursive=recursive,
        seed=seed,
        intensity=intensity,
    )
    elapsed = time.perf_counter() - t_start

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    click.echo(click.style("  Batch Results", fg="yellow", bold=True))
    click.echo(f"  {'─' * 40}")
    click.echo(
        f"  ✅  Succeeded: {click.style(str(len(successes)), fg='green', bold=True)}"
    )

    if failures:
        click.echo(
            f"  ❌  Failed:    {click.style(str(len(failures)), fg='red', bold=True)}"
        )
        for fail in failures[:5]:
            click.echo(f"       • {fail.input_path}: {fail.error}", err=True)
        if len(failures) > 5:
            click.echo(f"       … and {len(failures) - 5} more", err=True)

    click.echo(f"  ⏱️   Total time: {elapsed:.2f}s")
    if successes:
        avg_ms = sum(r.elapsed_ms for r in successes) / len(successes)
        click.echo(f"  📊  Avg per image: {avg_ms:.1f}ms")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", "-p", default=8080, show_default=True, type=int, help="Bind port.")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development mode).")
@click.option("--workers", default=1, show_default=True, type=int, help="Number of uvicorn workers.")
def serve_cmd(host: str, port: int, reload: bool, workers: int) -> None:
    """Start the Neural Canvas API server.

    \b
    Examples:
        neural-canvas serve
        neural-canvas serve --host 0.0.0.0 --port 8080
        neural-canvas serve --reload --workers 4
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(
            click.style(
                "  ❌  uvicorn not found. Install it with: pip install uvicorn[standard]",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    _print_banner()
    click.echo(click.style("  Starting Neural Canvas API Server", fg="green", bold=True))
    click.echo(f"  🌐  Host: {click.style(host, fg='cyan')}")
    click.echo(f"  🔌  Port: {click.style(str(port), fg='cyan')}")
    click.echo(f"  👷  Workers: {workers}")
    click.echo(f"  🔄  Reload: {'enabled' if reload else 'disabled'}")
    click.echo()
    click.echo(
        f"  API docs: {click.style(f'http://{host}:{port}/docs', fg='yellow', bold=True)}"
    )
    click.echo(
        f"  Health:   {click.style(f'http://{host}:{port}/health', fg='yellow', bold=True)}"
    )
    click.echo()
    click.echo(click.style("  Press CTRL+C to stop the server.\n", fg="bright_black"))

    uvicorn.run(
        "neural_canvas.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
