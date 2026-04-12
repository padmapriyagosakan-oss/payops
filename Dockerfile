# ---------------------------------------------------------------------------
# PayOps OpenEnv — Production Dockerfile
# ---------------------------------------------------------------------------
# Builds a lightweight, reproducible container that runs the FastAPI server.
# Compatible with HuggingFace Spaces (port 7860) and local development (8000).
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . /app/payops_env

# Also copy openenv.yaml to WORKDIR so validators can find it at /app/openenv.yaml
COPY openenv.yaml /app/openenv.yaml

# Both /app (for payops_env.*) and /app/payops_env (for server.*) on PYTHONPATH
# This matches the openenv.yaml app: server.app:app path and platform example format
ENV PYTHONPATH="/app:/app/payops_env"

# HuggingFace Spaces requires port 7860; default to 7860
ENV PORT=7860

# Runtime env vars (overridden at deploy time via HF Space secrets)
# HF_TOKEN / OPENAI_API_KEY are injected at runtime — not baked into the image
ENV API_BASE_URL="" \
    MODEL_NAME="" \
    PAYOPS_BASE_URL="http://localhost:7860"

# Switch to non-root user
USER appuser

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]

