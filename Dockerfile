# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# System dependencies for Camoufox/Firefox in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    libxt6 \
  && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies (better layer caching — source changes won't invalidate this)
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the source
COPY . /app
RUN uv sync --frozen --no-dev

# Download the Camoufox Firefox binary at build time
# Set a stable data dir so the binary survives user/path changes
ENV CAMOUFOX_DATA_DIR=/opt/camoufox
RUN uv run python -m camoufox fetch

# No default CMD — overridden per-service in docker-compose
