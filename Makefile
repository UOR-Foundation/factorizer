.PHONY: all build test check lint format clean help bench docs release

# Default target
all: check test

# Build the project
build:
	@echo "Building project..."
	@cargo build

# Build in release mode
release:
	@echo "Building release..."
	@cargo build --release

# Run tests
test:
	@echo "Running tests..."
	@cargo test

# Run tests with output
test-verbose:
	@echo "Running tests (verbose)..."
	@cargo test -- --nocapture

# Run specific test
test-one:
	@echo "Usage: make test-one TEST=test_name"
	@cargo test $(TEST) -- --nocapture

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	@cargo bench

# Check code (compile without producing artifacts)
check:
	@echo "Checking code..."
	@cargo check --all-targets

# Run clippy linter
lint:
	@echo "Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

# Format code
format:
	@echo "Formatting code..."
	@cargo fmt

# Check formatting
format-check:
	@echo "Checking formatting..."
	@cargo fmt -- --check

# Run all checks (format, lint, test)
pre-commit: format-check lint check test
	@echo "All checks passed!"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@cargo clean
	@rm -rf target/

# Generate documentation
docs:
	@echo "Generating documentation..."
	@cargo doc --no-deps --open

# Generate documentation with private items
docs-private:
	@echo "Generating documentation (with private items)..."
	@cargo doc --no-deps --document-private-items --open

# Run examples
example-observe:
	@echo "Running observe example..."
	@cargo run --example observe

example-recognize:
	@echo "Running recognize example..."
	@cargo run --example recognize

example-discover:
	@echo "Running discover example..."
	@cargo run --example discover

example-visualize:
	@echo "Running visualize example..."
	@cargo run --example visualize

# Run all examples
examples: example-observe example-recognize example-discover example-visualize

# Development setup
setup:
	@echo "Setting up development environment..."
	@rustup component add rustfmt clippy
	@echo "Development environment ready!"

# Create data directory if it doesn't exist
data-dir:
	@mkdir -p data

# Run with specific features
run-feature:
	@echo "Usage: make run-feature FEATURES=feature1,feature2"
	@cargo run --features $(FEATURES)

# Update dependencies
update:
	@echo "Updating dependencies..."
	@cargo update

# Check for outdated dependencies
outdated:
	@echo "Checking for outdated dependencies..."
	@cargo install cargo-outdated || true
	@cargo outdated

# Security audit
audit:
	@echo "Running security audit..."
	@cargo install cargo-audit || true
	@cargo audit

# Coverage report (requires cargo-tarpaulin)
coverage:
	@echo "Generating coverage report..."
	@cargo install cargo-tarpaulin || true
	@cargo tarpaulin --out Html --output-dir target/coverage

# Size optimization analysis
size:
	@echo "Analyzing binary size..."
	@cargo build --release
	@ls -lh target/release/rust_pattern_solver

# Profile-guided optimization
pgo:
	@echo "Building with profile-guided optimization..."
	@cargo install cargo-pgo || true
	@cargo pgo build
	@cargo pgo optimize

# Help
help:
	@echo "The Pattern - Rust Implementation"
	@echo ""
	@echo "Available targets:"
	@echo "  make all          - Run checks and tests (default)"
	@echo "  make build        - Build the project"
	@echo "  make release      - Build in release mode"
	@echo "  make test         - Run tests"
	@echo "  make test-verbose - Run tests with output"
	@echo "  make bench        - Run benchmarks"
	@echo "  make check        - Check code compilation"
	@echo "  make lint         - Run clippy linter"
	@echo "  make format       - Format code"
	@echo "  make format-check - Check code formatting"
	@echo "  make pre-commit   - Run all checks before committing"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docs         - Generate documentation"
	@echo "  make examples     - Run all examples"
	@echo "  make setup        - Setup development environment"
	@echo "  make update       - Update dependencies"
	@echo "  make audit        - Run security audit"
	@echo "  make coverage     - Generate test coverage report"
	@echo "  make help         - Show this help message"