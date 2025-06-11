# Rust Pattern Solver - Development Container

This directory contains the development container configuration for the Rust Pattern Solver project.

## Quick Start

### Option 1: Using Pre-built Image (Recommended)
1. Open this project in VS Code
2. When prompted, click "Reopen in Container"
3. Wait for the container to build (first time only)
4. Start coding!

### Option 2: Using Docker Compose (Advanced)
1. Rename `devcontainer-compose.json` to `devcontainer.json`
2. Open in VS Code and reopen in container
3. This option provides volume persistence and resource limits

### Option 3: Using Custom Dockerfile
1. Modify the `Dockerfile` as needed
2. Update `devcontainer.json` to use `"dockerFile": "Dockerfile"`
3. Rebuild the container

## Features

### Pre-installed Tools
- **Rust Toolchain**: Latest stable (1.79+) on Debian 12 (Bookworm) with rustfmt, clippy, and rust-analyzer
- **Cargo Extensions**:
  - `cargo-watch`: Auto-rebuild on file changes
  - `cargo-edit`: Add/update dependencies from CLI
  - `cargo-audit`: Security vulnerability audit
  - `cargo-outdated`: Check for outdated dependencies
  - `cargo-criterion`: Benchmarking framework
  - `cargo-tarpaulin`: Code coverage
  - `cargo-expand`: Macro expansion
  - `cargo-deny`: License and security checking

### System Dependencies
- GMP, MPFR, MPC: Required for arbitrary precision arithmetic (rug crate)
- Debugging tools: gdb, valgrind, heaptrack
- Performance tools: hyperfine
- Build tools: cmake, pkg-config

### VS Code Extensions
When the container starts, you'll need to install the following recommended extensions:
- rust-analyzer: Official Rust language server
- Even Better TOML: TOML file support
- crates: Cargo.toml dependency management
- CodeLLDB: Debugger for Rust

These are listed in `.vscode/extensions.json` and VS Code will prompt you to install them.

## Configuration

### Environment Variables
- `RUST_BACKTRACE=1`: Enable backtraces
- `RUST_LOG=debug`: Set logging level

### VS Code Settings
- Format on save enabled
- Clippy as the default checker
- All inlay hints enabled
- Full rust-analyzer features

## Usage

### Common Commands
```bash
# Run all checks and tests
make all

# Watch for changes and auto-rebuild
cargo watch -x check -x test

# Run examples
make examples

# Generate documentation
make docs

# Run benchmarks
make bench
```

### Debugging
1. Set breakpoints in VS Code
2. Press F5 or use the Debug panel
3. Select the appropriate debug configuration

### Performance Profiling
```bash
# CPU profiling with valgrind
valgrind --tool=callgrind target/release/rust_pattern_solver

# Memory profiling
heaptrack target/release/rust_pattern_solver

# Benchmark comparisons
hyperfine 'target/release/rust_pattern_solver'
```

## Troubleshooting

### Container Build Fails
- Ensure Docker is running
- Check available disk space
- Try rebuilding: Command Palette → "Remote-Containers: Rebuild Container"

### Rust-analyzer Not Working
- Run `rustup component add rust-analyzer`
- Restart VS Code
- Check the Rust-analyzer output panel

### Slow Performance
- Increase Docker memory allocation
- Use the docker-compose configuration for better resource management
- Ensure target directory is in a volume (not bind mounted)

## Customization

To customize the development environment:

1. Edit `.devcontainer/devcontainer.json` for VS Code settings
2. Modify `.devcontainer/post-create.sh` for additional setup steps
3. Update `.devcontainer/Dockerfile` for system dependencies

## Resources

- [VS Code Dev Containers](https://code.visualstudio.com/docs/remote/containers)
- [Rust Analyzer Manual](https://rust-analyzer.github.io/manual.html)
- [The Pattern README](../README.md)