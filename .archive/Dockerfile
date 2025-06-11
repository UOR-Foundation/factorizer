# Prime Resonance Field (RFH3) - Docker Image
# Published by UOR Foundation (https://uor.foundation)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -e ".[dev,docs,benchmark]"

# Copy source code
COPY prime_resonance_field/ ./prime_resonance_field/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY Makefile ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash rfh3user
RUN chown -R rfh3user:rfh3user /app
USER rfh3user

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from prime_resonance_field import RFH3; print('RFH3 OK')" || exit 1

# Default command
CMD ["python", "-m", "prime_resonance_field.cli", "info"]

# Expose port for documentation server
EXPOSE 8000

# Labels
LABEL maintainer="UOR Foundation <research@uor.foundation>"
LABEL version="3.0.0"
LABEL description="Prime Resonance Field (RFH3) - Adaptive Integer Factorization"
LABEL org.opencontainers.image.source="https://github.com/UOR-Foundation/factorizer"
LABEL org.opencontainers.image.documentation="https://github.com/UOR-Foundation/factorizer/blob/main/prime_resonance_field/README.md"
LABEL org.opencontainers.image.vendor="UOR Foundation"
LABEL org.opencontainers.image.licenses="MIT"
