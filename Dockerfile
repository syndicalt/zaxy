# Zaxy production Dockerfile
# Multi-stage build for minimal image size

FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip install --no-cache-dir dist/*.whl

# ------------------------------------------------------------------
# Runtime stage
# ------------------------------------------------------------------
FROM python:3.13-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r zaxy && useradd -r -g zaxy zaxy

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Create directories
RUN mkdir -p /app/.eventloom /app/.volumes && \
    chown -R zaxy:zaxy /app

USER zaxy

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD zaxy status || exit 1

EXPOSE 8080

ENTRYPOINT ["zaxy"]
CMD ["serve"]
