#!/usr/bin/env bash
# Build the image, run it with a mounted index bundle, measure cold start and latency, clean up.
# Works on a Linux VM (or Docker Desktop with Git Bash). Uses only invented addresses.
#
#   INDEX_DIR=/ruta/al/indice [BASEMAPS_DIR=/ruta/basemaps] [CPUS=4] [MEMORY=4g] bash scripts/smoke_docker.sh
#
# Requires: docker, python3 (for scripts/medir_servicio.py; stdlib only).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))"  # pwd -W: Windows path under Git Bash
IMAGE="${IMAGE:-normalizador-direcciones:smoke}"
NAME="${NAME:-normalizador-smoke}"
PORT="${PORT:-18000}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-4g}"
PYTHON="${PYTHON:-python3}"
: "${INDEX_DIR:?set INDEX_DIR to the index bundle directory (see scripts/preparar_indice.py)}"
export MSYS_NO_PATHCONV=1  # Git Bash on Windows: do not rewrite the container paths

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== build"
t0=$(date +%s)
docker build -t "$IMAGE" "$HERE"
echo "build_seconds=$(( $(date +%s) - t0 ))"
echo "image_bytes=$(docker image inspect "$IMAGE" --format '{{.Size}}')"
docker image ls "$IMAGE"

echo "== missing index must fail fast with a clear message"
set +e
docker run --rm "$IMAGE" check >/tmp/smoke_missing.log 2>&1
code=$?
set -e
tail -3 /tmp/smoke_missing.log
[ "$code" -eq 3 ] || { echo "expected exit code 3 without an index, got $code"; exit 1; }

echo "== bad env must fail with exit code 2"
set +e
docker run --rm -e NORMALIZER_ONNX_THREADS=many "$IMAGE" check >/dev/null 2>&1
code=$?
set -e
[ "$code" -eq 2 ] || { echo "expected exit code 2 for a bad env value, got $code"; exit 1; }

echo "== run (cpus=$CPUS memory=$MEMORY, read-only root fs, non-root)"
cleanup
mounts=(-v "$INDEX_DIR:/data/indice:ro")
if [ -n "${BASEMAPS_DIR:-}" ]; then mounts+=(-v "$BASEMAPS_DIR:/data/basemaps:ro"); fi
docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" --cpus "$CPUS" --memory "$MEMORY" \
  --read-only --tmpfs /tmp:size=256m --cap-drop ALL --security-opt no-new-privileges \
  "${mounts[@]}" "$IMAGE" >/dev/null
"$PYTHON" "$HERE/scripts/medir_servicio.py" --url "http://127.0.0.1:$PORT" --esperar 300 --n 200 --lote 1000 \
  || { docker logs "$NAME" | tail -30; exit 1; }

echo "== container stats and user"
docker stats --no-stream --format 'mem={{.MemUsage}} cpu={{.CPUPerc}}' "$NAME"
docker exec "$NAME" id
docker exec "$NAME" python -c "import importlib.util as u; print('torch installed:', u.find_spec('torch') is not None)"
docker inspect "$NAME" --format 'health={{.State.Health.Status}}'
docker logs "$NAME" 2>&1 | grep -E '^\[servicio\]' || true
echo "== ok"
