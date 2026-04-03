# Multi-stage build using official Meta OpenEnv base
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available for potential VCS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build mode: 'standalone' for public PyPI use
ARG BUILD_MODE=standalone
ARG ENV_NAME=smart-grid-demand-response

# Copy the canonical structure (root files and the server/ folder)
COPY . /app/env
WORKDIR /app/env

# Use 'uv' for super-fast and reproducible dependency sync
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Build logic: Sync dependencies and create the .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# Final runtime stage using the same base image
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment and our code from the builder stage
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Environment configuration
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Health check to ensure the FastAPI server is responsive
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/state || exit 1

# Run the environment server using' uvicorn'
# We 'cd' into /app/env so that root-level imports (like 'models') work naturally
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
