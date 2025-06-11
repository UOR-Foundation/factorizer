#!/bin/bash
set -e

echo "🚀 Setting up Rust Pattern Solver development environment..."

# Update rust toolchain to latest stable
echo "📦 Updating Rust toolchain to latest stable..."
rustup update stable
rustup component add rustfmt clippy rust-analyzer rust-src

# Install additional cargo tools (only if not in Dockerfile)
echo "🛠️ Installing additional cargo tools..."
# Check if tools are already installed to avoid redundancy
command -v cargo-audit >/dev/null 2>&1 || cargo install cargo-audit
command -v cargo-outdated >/dev/null 2>&1 || cargo install cargo-outdated
command -v cargo-criterion >/dev/null 2>&1 || cargo install cargo-criterion
command -v cargo-deny >/dev/null 2>&1 || cargo install cargo-deny
command -v cargo-tarpaulin >/dev/null 2>&1 || cargo install cargo-tarpaulin

# Install system dependencies for rug (GMP, MPFR, MPC)
echo "📚 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    m4 \
    pkg-config \
    build-essential \
    cmake \
    valgrind \
    hyperfine \
    heaptrack

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p target
mkdir -p benches/results

# Set up git hooks (optional)
if [ -d .git ]; then
    echo "🪝 Setting up git hooks..."
    mkdir -p .git/hooks
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Rust Pattern Solver

echo "Running pre-commit checks..."

# Format check
if ! cargo fmt -- --check; then
    echo "❌ Formatting errors detected. Run 'cargo fmt' to fix."
    exit 1
fi

# Clippy check
if ! cargo clippy --all-targets --all-features -- -D warnings; then
    echo "❌ Clippy warnings detected."
    exit 1
fi

# Test check
if ! cargo test --quiet; then
    echo "❌ Tests failed."
    exit 1
fi

echo "✅ All pre-commit checks passed!"
EOF

    chmod +x .git/hooks/pre-commit
fi

# Build the project to cache dependencies
echo "🔨 Building project (this may take a while on first run)..."
cargo build

# Generate initial documentation
echo "📚 Generating documentation..."
cargo doc --no-deps

# Set up environment
echo "🌍 Setting up environment..."
# Determine which shell config to use
if [ -n "$ZSH_VERSION" ] || [ -f ~/.zshrc ]; then
    SHELL_RC=~/.zshrc
else
    SHELL_RC=~/.bashrc
fi

echo 'export RUST_BACKTRACE=1' >> $SHELL_RC
echo 'export RUST_LOG=debug' >> $SHELL_RC

# Create a welcome message
cat > ~/.welcome_message << 'EOF'

╔════════════════════════════════════════════════════════════════════╗
║                    🌟 Rust Pattern Solver 🌟                       ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Welcome to The Pattern implementation in Rust!                    ║
║                                                                    ║
║  Quick commands:                                                   ║
║  - make help     : Show all available commands                     ║
║  - make all      : Run checks and tests                          ║
║  - make examples : Run all examples                               ║
║  - cargo watch   : Auto-rebuild on changes                        ║
║                                                                    ║
║  Philosophy:                                                       ║
║  "The Pattern is not an algorithm—it's a recognition."            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

EOF

echo 'cat ~/.welcome_message' >> $SHELL_RC

echo "✅ Development environment setup complete!"
echo ""
echo "Run 'make help' to see available commands."