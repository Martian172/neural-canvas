# Neural Canvas — Makefile

.PHONY: install test lint serve clean docs

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=neural_canvas

lint:
	flake8 neural_canvas --max-line-length=120
	black --check neural_canvas

format:
	black neural_canvas

serve:
	python -m neural_canvas.api.server

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/

docs:
	sphinx-build -b html docs/ docs/_build/

.DEFAULT_GOAL := help
help:
	@echo "Available commands:"
	@echo "  make install   - Install package in dev mode"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run linter"
	@echo "  make format    - Auto-format code"
	@echo "  make serve     - Start API server"
	@echo "  make clean     - Clean build artifacts"
