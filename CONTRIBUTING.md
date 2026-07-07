# Contributing to Neural Canvas

Thank you for considering contributing to Neural Canvas! 🎨

## How to Contribute

### Reporting Bugs
- Open an issue with a clear title and description
- Include steps to reproduce, expected vs actual behavior
- Attach sample images if relevant

### Suggesting Features
- Open an issue tagged `enhancement`
- Describe the feature and its use case

### Pull Requests
1. Fork the repo and create your branch from `main`
2. Install dev dependencies: `make install`
3. Write tests for new functionality
4. Ensure tests pass: `make test`
5. Lint your code: `make lint`
6. Submit a pull request with a clear description

## Development Setup

```bash
git clone https://github.com/Martian172/neural-canvas.git
cd neural-canvas
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
make install
make test
```

## Code Style
- Follow PEP 8
- Max line length: 120 characters
- Use type hints everywhere
- Write docstrings for all public functions

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
