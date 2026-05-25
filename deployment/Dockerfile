# syntax=docker/dockerfile:1
#
# Production image for the tooth-resorption Flask inference API.
#
# Build from the repository root so the build context contains pyproject.toml,
# src/, and deployment/:
#
#     docker build -f deployment/Dockerfile -t trd-api .
#
# Run it with the host's models directory mounted read-only so model artefacts
# are never baked into the image:
#
#     docker run --rm -p 5000:5000 -v $(pwd)/models:/app/models:ro trd-api

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src/ src/
COPY deployment/ deployment/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e ".[deployment]"

EXPOSE 5000

ENV TRD_DEVICE=cpu \
    TRD_PORT=5000 \
    FLASK_ENV=production

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health').getcode()==200 else 1)" || exit 1

CMD ["python", "-m", "deployment.api.flask_api"]
