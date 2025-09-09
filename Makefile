.PHONY: help install install-dev test test-cov lint typecheck format clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install .

install-dev:  ## Install development dependencies
	pip install -e .[dev]

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest

test-fast:  ## Run tests in parallel
	pytest tests/ -n auto

lint:  ## Run all linting tools
	@echo "Running black..."
	black --check --diff hal/ tests/
	@echo "Running isort..."
	isort --check-only --diff hal/ tests/
	@echo "Running pylint..."
	pylint hal/

typecheck:  ## Run type checking
	mypy hal/

format:  ## Format code with black and isort
	black hal/ tests/
	isort hal/ tests/

clean:  ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete

pre-commit:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

ci: test lint typecheck  ## Run all CI checks locally

all: format lint typecheck test-cov  ## Run all checks and tests
