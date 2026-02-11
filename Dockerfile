# Multi-stage build for BONGAS-ML
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY bongas_ml/ ./bongas_ml/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create outputs directory
RUN mkdir -p outputs

# Create non-root user
RUN groupadd -r bongas && useradd -r -g bongas bongas
RUN chown -R bongas:bongas /app
USER bongas

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import bongas_ml; print('BONGAS-ML is healthy')" || exit 1

# Default command
CMD ["python", "-m", "bongas_ml.scripts.train", "--help"]

# Development stage
FROM base as development
USER root
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy pre-commit
USER bongas
CMD ["bash"]

# Production stage
FROM base as production
USER bongas
CMD ["python", "-m", "bongas_ml.scripts.nightly_job", "--help"]