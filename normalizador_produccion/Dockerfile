# syntax=docker/dockerfile:1
#
# CANDIDATE / staging image of the Cali address normalizer: ONNX Runtime backend, no torch, non-root.
# The serving tooling is production-grade; the model weights in modelo/ are the retraining CANDIDATE
# (verdict NO APRUEBA, not promoted). Not fit for production until the project owner approves it.
#
#   docker build -t normalizador-direcciones:0.2.0-candidato .
#   docker run --rm -p 8000:8000 -v /ruta/al/indice:/data/indice:ro normalizador-direcciones:0.2.0-candidato
#
# The index bundle (documents + embeddings, derived from cadastre / nomenclature data) is NOT in this
# image: mount it at /data/indice, or set INDEX_URL + INDEX_SHA256. Optional private variant with the
# bundle inside (never push it to a public registry):
#
#   docker build --target con-indice --build-context indice=/ruta/al/indice -t normalizador-privado:0.2.0-candidato .

# Base image: the floating tag python:<version>-slim is convenient but not reproducible (the content
# changes with every upstream rebuild). For a reproducible build, resolve the digest once and pin it,
# e.g. `docker buildx imagetools inspect python:3.12-slim` and use FROM python:3.12-slim@sha256:<digest>
# in both stages (see GUIA_DESPLIEGUE.md, "Fijar la imagen base").
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------------------------
# 1. build: install wheels into a self-contained virtualenv and strip what serving never loads
# ---------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN pip install --only-binary=:all: -r /tmp/requirements.txt \
    && pip uninstall -y pip setuptools wheel || true

# Test suites, C headers and Cython sources bundled inside the wheels are dead weight at run time.
# (__pycache__ is kept: precompiled bytecode shortens the cold start.)
RUN find /opt/venv -depth -type d \( -name tests -o -name test \) -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python*/site-packages/pyarrow/include \
              /opt/venv/lib/python*/site-packages/pyarrow/src \
              /opt/venv/lib/python*/site-packages/numpy/_core/include \
    && find /opt/venv -name "*.pyx" -delete -o -name "*.pxd" -delete -o -name "*.h" -delete

# Smoke test inside the build stage: the serving stack imports without torch.
COPY src/ /app/src/
RUN PYTHONPATH=/app/src python -c "import sys, onnxruntime, cali_address.api.main, servicio.app; \
    assert 'torch' not in sys.modules, 'torch must not be imported'; print('imports ok')"

# ---------------------------------------------------------------------------------------------
# 2. runtime: slim base + venv + code + model (no compiler, no pip, no curl)
# ---------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="normalizador-direcciones"       org.opencontainers.image.description="Imagen candidata / staging del normalizador de direcciones de Cali (ONNX Runtime, sin torch). NO apta para produccion hasta aprobacion del titular."       org.opencontainers.image.version="0.2.0"       org.opencontainers.image.licenses="Apache-2.0 (codigo); pesos del modelo: ver NOTICE"       org.opencontainers.image.source="https://github.com/Juanpgm/HIIG"       co.hiig.status="candidato-no-aprobado"

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app \
    && rm -rf /usr/local/lib/python*/ensurepip /usr/local/lib/python*/idlelib /usr/local/lib/python*/tkinter \
    && mkdir -p /data/indice /data/basemaps \
    && chown app:app /data/indice

COPY --from=build /opt/venv /opt/venv
COPY --chown=root:root src/ /app/src/
COPY --chown=root:root modelo/ /app/modelo/
COPY --chown=root:root docs/ /app/docs/
COPY --chown=root:root LICENSE NOTICE /app/
RUN python -m compileall -q /app/src

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    NORMALIZER_BACKEND=onnx \
    NORMALIZER_BACKEND_FALLBACK=true \
    NORMALIZER_ONNX_THREADS=auto \
    NORMALIZER_ONNX_INDEX=fp32 \
    UVICORN_WORKERS=1 \
    LOG_LEVEL=info \
    MODEL_DIR=/app/modelo \
    INDEX_DIR=/data/indice \
    BASEMAPS_DIR=/data/basemaps \
    RUNTIME_DIR=/tmp/normalizador \
    MAX_UPLOAD_MB=25 \
    MAX_ROWS=20000 \
    WARMUP_TIMEOUT_S=300

WORKDIR /app
USER 10001:10001
EXPOSE 8000

# 200 only when model + index are loaded (503 while loading). start-period covers the cold start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4).status == 200 else 1)"]

ENTRYPOINT ["python", "-m", "servicio"]
CMD ["serve"]

# ---------------------------------------------------------------------------------------------
# 3. con-indice (optional, PRIVATE registries only): bakes a local bundle into the image
# ---------------------------------------------------------------------------------------------
FROM runtime AS con-indice
COPY --from=indice --chown=root:root . /data/indice/

# Default target when --target is not given.
FROM runtime
