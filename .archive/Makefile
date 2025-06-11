# Prime Resonance Field (RFH3) - Development Makefile
# Published by UOR Foundation (https://uor.foundation)
# Repository: https://github.com/UOR-Foundation/factorizer

.PHONY: help install install-dev test test-verbose test-coverage clean lint format type-check build upload upload-test docs benchmark demo

# Default target
help:
	@echo "Prime Resonance Field (RFH3) - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install      Install package in current environment"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  clean        Clean build artifacts and cache files"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-verbose Run tests with verbose output"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  test-unit    Run only unit tests"
	@echo "  test-integration Run only integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run all linting tools"
	@echo "  format       Auto-format code with black and isort"
	@echo "  type-check   Run mypy type checking"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo ""
	@echo "Packaging:"
	@echo "  build        Build distribution packages"
	@echo "  upload-test  Upload to TestPyPI"
	@echo "  upload       Upload to PyPI"
	@echo ""
	@echo "Benchmarking:"
	@echo "  benchmark    Run performance benchmarks"
	@echo "  demo         Run demonstration factorizations"
	@echo ""
	@echo "Development:"
	@echo "  check        Run all checks (lint, type-check, test)"
	@echo "  pre-commit   Run pre-commit checks"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,benchmark]"

# Testing
test:
	python -m pytest tests/ -v

test-verbose:
	python -m pytest tests/ -v -s

test-coverage:
	python -m pytest tests/ --cov=prime_resonance_field --cov-report=html --cov-report=term

test-unit:
	python -m pytest tests/ -m "unit" -v

test-integration:
	python -m pytest tests/ -m "integration" -v

test-specific:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-specific TEST=path/to/test"; \
		exit 1; \
	fi
	python -m pytest $(TEST) -v

# Code Quality
lint:
	flake8 . tests/
	black --check . tests/
	isort --check-only . tests/

format:
	black . tests/
	isort . tests/

type-check:
	mypy .

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

docs-clean:
	cd docs && make clean

# Packaging
build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*

# Benchmarking
benchmark:
	python -m prime_resonance_field.benchmark

benchmark-quick:
	python -m prime_resonance_field.benchmark --quick

benchmark-extensive:
	python -m prime_resonance_field.benchmark --extensive

demo:
	python -c "from prime_resonance_field import RFH3; rfh3 = RFH3(); print('143 =', rfh3.factor(143)); print('10403 =', rfh3.factor(10403))"

# Development
check: lint type-check test

pre-commit: format check

pre-release: clean format check test-coverage docs build
	@echo "Pre-release checks completed successfully!"

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

deep-clean: clean
	rm -rf .tox/
	rm -rf .venv/
	rm -rf venv/
	rm -rf env/

# Performance Profiling
profile:
	python -m cProfile -o profile.stats -m prime_resonance_field.benchmark --profile
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

profile-memory:
	python -m memory_profiler tests/test_memory_usage.py

# Research and Development
research-notes:
	@echo "Opening research documentation..."
	@if command -v code >/dev/null 2>&1; then \
		code README.md; \
	else \
		${EDITOR:-nano} README.md; \
	fi

validate-mathematics:
	python tests/validate_mathematical_properties.py

# Continuous Integration helpers
ci-setup:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"

ci-test:
	python -m pytest tests/ --junitxml=test-results.xml --cov=prime_resonance_field --cov-report=xml

# Security
security-check:
	pip-audit
	bandit -r prime_resonance_field/

# Release Management
tag-release:
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make tag-release VERSION=x.y.z"; \
		exit 1; \
	fi
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

# Environment Management
venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

conda-env:
	conda env create -f environment.yml
	@echo "Conda environment created. Activate with: conda activate prime-resonance-field"

# Docker Support
docker-build:
	docker build -t prime-resonance-field:latest .

docker-run:
	docker run -it --rm prime-resonance-field:latest

docker-test:
	docker run --rm prime-resonance-field:latest python -m pytest

# Advanced Testing
test-matrix:
	tox

test-parallel:
	python -m pytest tests/ -n auto

stress-test:
	python tests/stress_test.py

# Maintenance
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

check-outdated:
	pip list --outdated

# Quick Development Cycle
dev: format check test
	@echo "Development cycle completed!"

# Research Reproducibility
reproduce-results:
	python scripts/reproduce_paper_results.py

generate-figures:
	python scripts/generate_performance_figures.py

# Jupyter Notebook Support
notebook:
	jupyter lab notebooks/

notebook-clean:
	find notebooks/ -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

# Version Information
version:
	@python -c "from prime_resonance_field import __version__; print(__version__)"

info:
	@echo "Prime Resonance Field (RFH3)"
	@echo "Version: $$(python -c 'from prime_resonance_field import __version__; print(__version__)')"
	@echo "Publisher: UOR Foundation"
	@echo "Repository: https://github.com/UOR-Foundation/factorizer"
	@echo "Homepage: https://uor.foundation"
