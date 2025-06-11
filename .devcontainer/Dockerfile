# Use the official Rust image as base - latest stable on Debian 12 (Bookworm)
FROM rust:latest-slim-bookworm

# Install essential development tools and dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    pkg-config \
    cmake \
    # Git and version control
    git \
    git-lfs \
    # GMP, MPFR, MPC for arbitrary precision (required by rug)
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    m4 \
    # SSL/TLS
    libssl-dev \
    ca-certificates \
    # Debugging and profiling
    gdb \
    valgrind \
    heaptrack \
    hyperfine \
    # Utilities
    curl \
    wget \
    sudo \
    locales \
    zsh \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user
USER $USERNAME

# Install Rust components
RUN rustup update stable \
    && rustup component add rustfmt clippy rust-analyzer rust-src

# Install essential cargo extensions
RUN cargo install --locked \
    cargo-watch \
    cargo-edit \
    cargo-expand

# Install oh-my-zsh for better terminal experience
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && echo 'export RUST_BACKTRACE=1' >> ~/.zshrc \
    && echo 'export RUST_LOG=debug' >> ~/.zshrc

# Set up working directory
WORKDIR /workspace

# Set shell to zsh
ENV SHELL=/usr/bin/zsh

# Expose any ports if needed (none for this project)
# EXPOSE 8080

# Default command
CMD ["/usr/bin/zsh"]