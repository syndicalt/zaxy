# Zaxy production Dockerfile
# Multi-stage build for minimal image size

FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and build the wheel from the package sources.
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip install --no-cache-dir dist/*.whl

# ------------------------------------------------------------------
# Runtime stage
# ------------------------------------------------------------------
FROM python:3.13-slim

WORKDIR /app

ENV ZAXY_ENV=production
# Anchor HOME at the app dir so the LadybugDB extension cache (HOME/.lbdb)
# lives under the writable, zaxy-owned tree at both build and runtime.
ENV HOME=/app

# Create non-root user
RUN groupadd -r zaxy && useradd -r -g zaxy zaxy

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Pre-seed the LadybugDB `vector` extension into the image cache (HOME/.lbdb).
# Since 2.3 the embedded engine is LadybugDB, which downloads this extension on
# first use rather than bundling it; baking it into the image makes approximate
# (HNSW) vector search work in the container with no runtime network fetch and
# no writable-HOME surprise under the non-root user. Requires network at build
# time and fails the build loudly if the extension cannot be fetched — so an
# image that "builds" is always ANN-ready rather than silently exact-only.
RUN python -c "import os, tempfile, ladybug; conn = ladybug.Connection(ladybug.Database(os.path.join(tempfile.mkdtemp(), 'seed'))); conn.execute('INSTALL vector'); conn.execute('LOAD vector'); print('LadybugDB vector extension pre-seeded to', os.path.join(os.environ['HOME'], '.lbdb'))"

# Create embedded Eventloom/projection directories and hand the whole app tree
# (including the pre-seeded HOME/.lbdb cache) to the non-root user.
RUN mkdir -p /app/.eventloom/projections && \
    chown -R zaxy:zaxy /app

USER zaxy

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD zaxy status || exit 1

EXPOSE 8080

ENTRYPOINT ["zaxy"]
CMD ["serve", "--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]
