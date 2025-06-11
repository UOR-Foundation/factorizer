# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Pattern is a Rust implementation of integer factorization through empirical observation. The project follows a unique philosophy: patterns are discovered from data, not imposed through algorithms. This is a research project that emphasizes recognition over computation.

## Common Development Commands

### Building
```bash
cargo build              # Debug build
cargo build --release    # Release build (optimized)
make build              # Same as cargo build
make release            # Same as cargo build --release
```

### Testing
```bash
cargo test              # Run all tests
cargo test -- --nocapture    # Run tests with output
make test               # Run all tests
make test-verbose       # Run tests with output
make test-one TEST=test_name # Run specific test
```

### Code Quality
```bash
cargo check             # Check compilation
cargo clippy -- -D warnings  # Run linter
cargo fmt               # Format code
make lint               # Run clippy
make format             # Format code
make format-check       # Check formatting
make pre-commit         # Run all checks (format, lint, test)
```

### Running Examples
```bash
cargo run --example observe      # Run pattern observation
cargo run --example recognize    # Run pattern recognition  
cargo run --example discover     # Run pattern discovery
cargo run --example visualize    # Run visualization
make examples           # Run all examples
```

### CLI Usage
```bash
# Observe patterns in number range
cargo run -- observe --start 1 --end 1000000 --output data/observations.json

# Recognize factors of a number
cargo run -- recognize 143 --pattern data/analysis/universal.json

# Discover patterns from observations
cargo run -- discover --input data/observations.json --output data/patterns.json

# Analyze patterns at different scales
cargo run -- analyze large
```

### Benchmarking
```bash
cargo bench             # Run all benchmarks
```

## High-Level Architecture

The codebase follows a data-driven architecture where patterns emerge from empirical observation:

### Core Philosophy Flow
1. **Observer** (`src/observer/`) - Collects empirical factorization data
2. **Pattern Discovery** (`src/relationship_discovery/`) - Finds invariants and relationships
3. **Pattern Implementation** (`src/pattern/`) - Three-stage recognition process

### Key Modules

**`src/observer/`** - Empirical data collection
- `collector.rs` - Gathers factorization data across ranges
- `analyzer.rs` - Discovers patterns without assumptions
- `constants.rs` - Extracts universal constants from patterns

**`src/pattern/`** - The Pattern implementation (3 stages)
- `recognition.rs` - Stage 1: Extract pattern signature from number
- `formalization.rs` - Stage 2: Express recognition mathematically
- `execution.rs` - Stage 3: Manifest factors from formalization

**`src/types/`** - Core data structures
- `number.rs` - Arbitrary precision arithmetic
- `signature.rs` - Pattern signatures for recognition
- `observation.rs` - Empirical observation data
- `pattern.rs` - Discovered pattern representations

**`src/emergence/`** - Pattern emergence analysis
- `invariants.rs` - Universal relationships
- `scaling.rs` - How patterns transform with scale
- `universal.rs` - Constants that appear everywhere

**`src/relationship_discovery/`** - Mathematical relationship finding
- `correlations.rs` - Statistical correlations
- `networks.rs` - Relationship networks
- `synthesis.rs` - Pattern synthesis

### Data Flow
1. Numbers → Observer → Observations (empirical data)
2. Observations → Analyzer → Patterns (discovered relationships)
3. Number → Pattern.recognize() → Recognition (signature extraction)
4. Recognition → Pattern.formalize() → Formalization (mathematical expression)
5. Formalization → Pattern.execute() → Factors (decoded result)

### Key Design Principles
- **No algorithms** - Only pattern recognition from empirical data
- **Emergence over engineering** - Code structure follows discovered patterns
- **Data-first development** - Every function exists because data revealed its necessity
- **Arbitrary precision** - Uses `rug` (GMP bindings) for exact arithmetic at any scale

## Important Patterns

The project discovers and uses several key patterns:
- **Modular DNA** - Characteristic modular signatures
- **Resonance Fields** - Harmonic relationships between factors
- **Quantum Neighborhoods** - Regions where factors manifest
- **Scale Invariants** - Patterns that hold across all scales

## Performance Notes

- Uses `rayon` for parallel observation collection
- Lazy evaluation for resonance field generation
- Pattern caching for discovered relationships
- Profile-guided optimization available: `make pgo`