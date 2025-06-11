# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The factorizer project implements RFH3 (Resonance Field Hypothesis v3) - an adaptive resonance field architecture for prime factorization. It's a sophisticated mathematical algorithm that uses multi-scale harmonic analysis and machine learning.

## Essential Commands

### Development Commands
```bash
# Installation
make install-dev      # Install with dev dependencies

# Testing
make test            # Run all tests
make test-unit       # Unit tests only
make test-integration # Integration tests only
python tests/run_all_tests.py  # Alternative test runner

# Code Quality - ALWAYS RUN BEFORE COMPLETING TASKS
make lint            # Run flake8, black, isort checks
make format          # Auto-format code
make type-check      # Run mypy type checking

# Combined checks
make check           # Run all checks (lint, type-check, test)
make pre-commit      # Format and check before commit
```

### Benchmarking
```bash
make benchmark       # Run performance benchmarks
make benchmark-quick # Quick benchmark suite
make demo           # Run demo factorizations
```

## Architecture

### Core Components

1. **rfh3.py** - Main factorizer entry point with RFH3Factorizer class
   - Multi-phase factorization strategy
   - Adaptive resonance field navigation
   - Learning-enhanced search

2. **core/** - Core mathematical components
   - `lazy_iterator.py`: Memory-efficient resonance node generation
   - `multi_scale_resonance.py`: Multi-resolution harmonic analysis
   - `state_manager.py`: Computation state management
   - `hierarchical_search.py`: Coarse-to-fine field exploration

3. **algorithms/** - Specialized factorization algorithms
   - `balanced_search.py`: For balanced semiprimes (p ≈ q)
   - `fermat_enhanced.py`: Enhanced Fermat factorization
   - `pollard_rho.py`: Optimized Pollard's Rho

4. **learning/** - Machine learning components
   - `pattern_learner.py`: Learns successful factorization patterns
   - `zone_predictor.py`: Predicts high-resonance zones

### Key Design Patterns

- **Lazy Evaluation**: LazyResonanceIterator generates nodes on-demand to conserve memory
- **Multi-Phase Strategy**: Progressive factorization phases from simple to complex
- **Adaptive Thresholds**: Dynamic adjustment based on number characteristics
- **Pattern Learning**: Improves performance by learning from successful factorizations

## Testing Strategy

- Tests are in `/tests/` mirroring the source structure
- Each module has corresponding unit tests
- Integration tests in `tests/integration/`
- Always run `make test` before considering work complete
- For specific tests: `python -m pytest tests/path/to/test.py::TestClass::test_method`

## Important Notes

- The project uses mathematical concepts like resonance fields and harmonic analysis
- Performance is critical - benchmark changes with `make benchmark-quick`
- Memory efficiency is important for large numbers (up to 256-bit)
- The CLI tools `rfh3-factor` and `rfh3-benchmark` are defined in pyproject.toml
- Archive folder contains experimental implementations and previous versions