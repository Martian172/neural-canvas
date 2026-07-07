"""
setup.py for neural-canvas
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="neural-canvas",
    version="0.1.0",
    description="AI-Powered Artistic Image Generation Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-canvas",
    author="Neural Canvas Team",
    author_email="hello@neural-canvas.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="ai, art, image-generation, style-transfer, neural-network, generative-art",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "click>=8.1.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "httpx>=0.24.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-canvas=neural_canvas.cli.commands:cli",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/neural-canvas/issues",
        "Source": "https://github.com/yourusername/neural-canvas",
        "Documentation": "https://neural-canvas.readthedocs.io",
    },
    include_package_data=True,
    zip_safe=False,
)
