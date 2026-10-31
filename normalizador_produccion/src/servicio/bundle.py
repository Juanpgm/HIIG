"""Model files (baked into the image), index bundle (supplied at run time) and runtime layout.

The model directory holds what is small and model-specific (ONNX graph, its metadata and the decision
configuration). The index bundle holds everything derived from the reference data (cadastral and
nomenclature documents and their embeddings); it is never part of the image or of git. At start the
service links both into one runtime directory, because ``cali_address.inference.AddressNormalizer``
reads a single artifacts directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

__all__ = [
    "BundleError",
    "IndexMissingError",
    "IndexIncompatibleError",
    "ModelMissingError",
    "MODEL_FILES",
    "INDEX_REQUIRED",
    "INDEX_OPTIONAL",
    "BASEMAP_FILES",
    "MANIFEST_FILENAME",
    "resolve_backend",
    "check_model_dir",
    "missing_index_files",
    "check_index",
    "download_index",
    "ensure_index",
    "assemble_runtime",
    "build_manifest",
    "sha256_file",
    "DEFAULT_MAX_DOWNLOAD_BYTES",
]

ONNX_FP32 = "model.onnx"
ONNX_INT8 = "model.int8.onnx"
ONNX_META = "onnx_meta.json"
#: Model directory: graph per backend + shared metadata/config. Only ``onnx_meta.json`` and the graph
#: of the chosen backend are required; the rest is optional (the service runs without the calibrated
#: decision layer if ``decision.json``/``reliability_calibrated.json`` are absent).
MODEL_FILES = (ONNX_FP32, ONNX_INT8, ONNX_META, "tuning.json", "decision.json", "reliability_calibrated.json",
               "reliability.json")
#: Index bundle files the service cannot start without.
INDEX_REQUIRED = ("catastro_docs.parquet", "catastro_emb.npy", "index_meta.json")
#: Optional index files: parcel centroid sidecar and startup caches (rebuilt if missing or stale).
INDEX_OPTIONAL = (
    "catastro_own_centroids.parquet",
    "interp_index_v1.pkl", "interp_index_v1.json",
    "catalog_keys_v1.pkl", "catalog_keys_v1.json",
    "gazetteer.pkl",
)
#: IDESC basemaps (optional: without them the gazetteer is disabled and the service still runs).
BASEMAP_FILES = ("barrios_veredas.geojson", "comunas_corregimientos.geojson")
MANIFEST_FILENAME = "indice_manifest.json"
MANIFEST_FORMAT = 1

_HELP = (
    "Opciones: (1) monte un volumen con el paquete del indice en INDEX_DIR; "
    "(2) defina INDEX_URL e INDEX_SHA256 para descargarlo al arrancar; "
    "(3) construyalo con scripts/preparar_indice.py. Ver GUIA_DESPLIEGUE.md, seccion 'El indice'."
)


class BundleError(RuntimeError):
    """Base class: the service cannot start with the files it was given."""


class ModelMissingError(BundleError):
    pass


class IndexMissingError(BundleError):
    pass


class IndexIncompatibleError(BundleError):
    pass


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise BundleError(f"no se pudo leer {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"{path} debe ser un objeto JSON")
    return data


# ---------------------------------------------------------------------------------------------
# Model directory
# ---------------------------------------------------------------------------------------------
def resolve_backend(requested: str, model_dir: str, allow_fallback: bool) -> tuple[str, list[str]]:
    """``(effective backend, warnings)``. ``onnx-int8`` falls back to ``onnx`` when its graph is absent."""
    warnings: list[str] = []
    if requested == "onnx-int8" and not os.path.isfile(os.path.join(model_dir, ONNX_INT8)):
        if allow_fallback and os.path.isfile(os.path.join(model_dir, ONNX_FP32)):
            warnings.append(
                f"{ONNX_INT8} no esta en {model_dir}: se usa el backend onnx (fp32). "
                "Defina NORMALIZER_BACKEND_FALLBACK=false para que esto sea un error."
            )
            return "onnx", warnings
        raise ModelMissingError(
            f"NORMALIZER_BACKEND=onnx-int8 pero {ONNX_INT8} no esta en {model_dir}"
            + ("" if allow_fallback else " (NORMALIZER_BACKEND_FALLBACK=false)")
        )
    return requested, warnings


def check_model_dir(model_dir: str, backend: str) -> dict:
    """Validate the model directory for ``backend``; return the parsed ``onnx_meta.json``."""
    if not os.path.isdir(model_dir):
        raise ModelMissingError(f"MODEL_DIR={model_dir} no existe")
    graph = ONNX_FP32 if backend == "onnx" else ONNX_INT8
    missing = [n for n in (graph, ONNX_META) if not os.path.isfile(os.path.join(model_dir, n))]
    if missing:
        raise ModelMissingError(f"faltan archivos del modelo en {model_dir}: {', '.join(missing)}")
    meta = _read_json(os.path.join(model_dir, ONNX_META))
    for key in ("max_len", "d_out", "xy_mean", "xy_std"):
        if key not in meta:
            raise ModelMissingError(f"{ONNX_META} no tiene la clave {key!r}")
    return meta


# ---------------------------------------------------------------------------------------------
# Index bundle
# ---------------------------------------------------------------------------------------------
def missing_index_files(index_dir: str) -> list[str]:
    return [n for n in INDEX_REQUIRED if not os.path.isfile(os.path.join(index_dir, n))]


def _index_shapes(index_dir: str) -> tuple[int, tuple[int, ...]]:
    """``(document rows, embedding shape)`` read from metadata only (no full load)."""
    import numpy as np
    import pyarrow.parquet as pq

    rows = pq.read_metadata(os.path.join(index_dir, "catastro_docs.parquet")).num_rows
    emb = np.load(os.path.join(index_dir, "catastro_emb.npy"), mmap_mode="r")
    return int(rows), tuple(int(x) for x in emb.shape)


def _verify_manifest_files(index_dir: str, manifest: dict) -> None:
    """Check every bundle file against the manifest: presence, byte size and sha256.

    The ``.pkl`` caches are loaded with ``pickle`` (arbitrary code execution if tampered), so a file that
    is present in the bundle but absent from the manifest is rejected too. Cost: one sequential read of
    the bundle at start (a few seconds for the 401,860-document index).
    """
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise IndexIncompatibleError(f"{MANIFEST_FILENAME} no lista archivos (campo 'files' vacio o ausente)")
    present = [n for n in INDEX_REQUIRED + INDEX_OPTIONAL if os.path.isfile(os.path.join(index_dir, n))]
    unlisted = [n for n in present if n not in files]
    if unlisted:
        raise IndexIncompatibleError(
            f"archivos del indice que el manifiesto no lista: {', '.join(unlisted)}. "
            "No se cargan archivos sin huella verificada; regenere el manifiesto con scripts/preparar_indice.py."
        )
    for name, info in files.items():
        if name not in INDEX_REQUIRED + INDEX_OPTIONAL:
            continue  # never linked into the runtime directory
        path = os.path.join(index_dir, name)
        if not os.path.isfile(path):
            raise IndexIncompatibleError(f"el manifiesto lista {name} pero el archivo no esta en {index_dir}")
        size = info.get("bytes") if isinstance(info, dict) else None
        digest = info.get("sha256") if isinstance(info, dict) else None
        if size is not None and os.path.getsize(path) != size:
            raise IndexIncompatibleError(
                f"{name} pesa {os.path.getsize(path)} bytes y el manifiesto dice {size}: archivo truncado o distinto")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise IndexIncompatibleError(f"el manifiesto no trae un sha256 valido para {name}")
        if sha256_file(path) != digest.lower():
            raise IndexIncompatibleError(
                f"{name} no coincide con el sha256 del manifiesto: archivo modificado o corrupto. "
                "No se arranca con un indice que no se puede verificar."
            )


def check_index(index_dir: str, model_meta: dict) -> list[str]:
    """Validate the bundle against the model. Return warnings; raise on anything that would break serving."""
    warnings: list[str] = []
    if not os.path.isdir(index_dir):
        raise IndexMissingError(f"no hay paquete de indice: INDEX_DIR={index_dir} no existe. {_HELP}")
    missing = missing_index_files(index_dir)
    if missing:
        raise IndexMissingError(
            f"faltan archivos del indice en INDEX_DIR={index_dir}: {', '.join(missing)}. {_HELP}"
        )
    _read_json(os.path.join(index_dir, "index_meta.json"))
    try:
        rows, shape = _index_shapes(index_dir)
    except Exception as exc:  # noqa: BLE001 - any read failure is a broken bundle
        raise IndexIncompatibleError(f"el indice en {index_dir} no se puede leer: {type(exc).__name__}: {exc}") from exc
    d_out = int(model_meta["d_out"])
    if len(shape) != 2 or shape[1] != d_out:
        raise IndexIncompatibleError(
            f"catastro_emb.npy tiene forma {shape}; el modelo produce vectores de {d_out} dimensiones")
    if shape[0] != rows:
        raise IndexIncompatibleError(
            f"catastro_emb.npy tiene {shape[0]} filas y catastro_docs.parquet {rows}: no son del mismo indice")
    if rows == 0:
        raise IndexIncompatibleError("el indice no tiene documentos")

    manifest_path = os.path.join(index_dir, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        warnings.append(
            f"el indice no trae {MANIFEST_FILENAME}: no se puede comprobar que sus embeddings sean de este "
            "modelo ni la integridad de los archivos (incluidos los .pkl, que se cargan con pickle). "
            "Generelo con scripts/preparar_indice.py."
        )
        return warnings
    manifest = _read_json(manifest_path)
    expected = model_meta.get("model_sha256")
    found = manifest.get("model_sha256")
    if expected and found and expected != found:
        raise IndexIncompatibleError(
            f"el indice se construyo para otro modelo (model_sha256 {found[:12]}..., "
            f"el modelo de la imagen es {expected[:12]}...). Reconstruya el indice con este modelo."
        )
    _verify_manifest_files(index_dir, manifest)
    return warnings


def build_manifest(index_dir: str, model_meta: dict) -> dict:
    """Manifest of an index bundle: model it belongs to, shapes and size/sha256 of every file."""
    rows, shape = _index_shapes(index_dir)
    files = {}
    for name in INDEX_REQUIRED + INDEX_OPTIONAL:
        path = os.path.join(index_dir, name)
        if os.path.isfile(path):
            files[name] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    return {
        "format": MANIFEST_FORMAT,
        "model_sha256": model_meta.get("model_sha256"),
        "d_out": int(model_meta["d_out"]),
        "n_docs": rows,
        "emb_shape": list(shape),
        "files": files,
    }


# ---------------------------------------------------------------------------------------------
# Download (INDEX_URL)
# ---------------------------------------------------------------------------------------------
def _safe_extract(archive: str, dest: str) -> None:
    dest_abs = os.path.abspath(dest)

    def inside(name: str) -> bool:
        target = os.path.abspath(os.path.join(dest_abs, name))
        return target == dest_abs or target.startswith(dest_abs + os.sep)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not inside(name) or os.path.isabs(name):
                    raise BundleError(f"el archivo descargado contiene una ruta insegura: {name!r}")
            zf.extractall(dest_abs)
        return
    try:
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                if not inside(m.name) or m.issym() or m.islnk() or m.isdev():
                    raise BundleError(f"el archivo descargado contiene una entrada insegura: {m.name!r}")
            try:
                tf.extractall(dest_abs, members=members, filter="data")
            except TypeError:  # Python without extraction filters (already vetted above)
                tf.extractall(dest_abs, members=members)
    except tarfile.TarError as exc:
        raise BundleError(f"el archivo descargado no es un .tar.gz ni un .zip valido: {exc}") from exc


def _locate_bundle_root(extracted: str) -> str:
    """The directory holding ``index_meta.json``: the archive root or a single top-level folder."""
    if os.path.isfile(os.path.join(extracted, "index_meta.json")):
        return extracted
    for root, _dirs, files in os.walk(extracted):
        if "index_meta.json" in files:
            return root
    raise BundleError("el archivo descargado no contiene index_meta.json")


DEFAULT_MAX_DOWNLOAD_BYTES = 4096 * 1024 * 1024


def download_index(url: str, sha256: str, index_dir: str, timeout_s: float = 600.0,
                   max_bytes: int | None = DEFAULT_MAX_DOWNLOAD_BYTES, allow_file_url: bool = False) -> None:
    """Download a ``.tar.gz``/``.zip`` bundle, verify its sha256 and install it into ``index_dir``.

    ``file://`` URLs are refused unless ``allow_file_url`` (a local-file read is only wanted in tests).
    The download is aborted once it exceeds ``max_bytes`` (``None`` disables the cap).
    """
    scheme = url.split(":", 1)[0].lower() if ":" in url else ""
    if scheme == "file" and not allow_file_url:
        raise BundleError("INDEX_URL con file:// esta deshabilitado; defina INDEX_ALLOW_FILE_URL=true solo en pruebas")
    if scheme not in ("https", "file"):
        raise BundleError("INDEX_URL debe empezar por https://")
    parent = os.path.dirname(os.path.abspath(index_dir)) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        work = tempfile.mkdtemp(prefix=".indice-", dir=parent)
    except OSError as exc:
        raise BundleError(
            f"no se puede escribir junto a INDEX_DIR={index_dir} para descargar el indice "
            f"({type(exc).__name__}: {exc}). Monte un volumen con escritura o el paquete ya descomprimido."
        ) from exc
    try:
        archive = os.path.join(work, "bundle")
        h = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp, open(archive, "wb") as out:  # noqa: S310
                received = 0
                for block in iter(lambda: resp.read(1 << 20), b""):
                    received += len(block)
                    if max_bytes is not None and received > max_bytes:
                        raise BundleError(
                            f"la descarga de INDEX_URL supera el limite de {max_bytes} bytes "
                            "(INDEX_MAX_DOWNLOAD_MB): se aborta")
                    h.update(block)
                    out.write(block)
        except OSError as exc:
            raise BundleError(f"no se pudo descargar INDEX_URL: {type(exc).__name__}: {exc}") from exc
        if h.hexdigest() != sha256.lower():
            raise BundleError(
                f"la descarga no coincide con INDEX_SHA256 (obtenido {h.hexdigest()[:12]}...): se descarta")
        extracted = os.path.join(work, "x")
        os.makedirs(extracted)
        _safe_extract(archive, extracted)
        root = _locate_bundle_root(extracted)
        try:
            os.makedirs(index_dir, exist_ok=True)
            for name in os.listdir(root):
                if name in INDEX_REQUIRED + INDEX_OPTIONAL + (MANIFEST_FILENAME,):
                    shutil.move(os.path.join(root, name), os.path.join(index_dir, name))
        except OSError as exc:
            raise BundleError(
                f"no se pudo instalar el indice descargado en INDEX_DIR={index_dir} "
                f"({type(exc).__name__}: {exc}). Si INDEX_DIR es un volumen de solo lectura, monte el "
                "paquete ya descomprimido o use un volumen con escritura para la descarga."
            ) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ensure_index(index_dir: str, url: str, sha256: str, timeout_s: float = 600.0,
                 max_bytes: int | None = DEFAULT_MAX_DOWNLOAD_BYTES, allow_file_url: bool = False) -> list[str]:
    """Download the bundle when files are missing and a URL is configured. Return log lines."""
    missing = missing_index_files(index_dir) if os.path.isdir(index_dir) else list(INDEX_REQUIRED)
    if not missing:
        return []
    if not url:
        return []  # check_index reports the missing files with the remedies
    download_index(url, sha256, index_dir, timeout_s, max_bytes, allow_file_url)
    return [f"indice descargado desde INDEX_URL en {index_dir}"]


# ---------------------------------------------------------------------------------------------
# Runtime layout
# ---------------------------------------------------------------------------------------------
def _link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def assemble_runtime(runtime_dir: str, model_dir: str, index_dir: str, basemaps_dir: str) -> tuple[str, str]:
    """Create ``<runtime>/artifacts`` and ``<runtime>/basemaps`` with links to the real files.

    Only the known file names are linked, so stray files in a mounted volume never reach the loader.
    The gazetteer writes its cache to ``<runtime>/artifacts/gazetteer.pkl`` (sibling of ``basemaps``).
    """
    artifacts = os.path.join(runtime_dir, "artifacts")
    basemaps = os.path.join(runtime_dir, "basemaps")
    for d in (artifacts, basemaps):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for name in MODEL_FILES:
        src = os.path.join(model_dir, name)
        if os.path.isfile(src):
            _link(os.path.abspath(src), os.path.join(artifacts, name))
    for name in INDEX_REQUIRED + INDEX_OPTIONAL:
        src = os.path.join(index_dir, name)
        if os.path.isfile(src):
            _link(os.path.abspath(src), os.path.join(artifacts, name))
    if os.path.isdir(basemaps_dir):
        for name in BASEMAP_FILES:
            src = os.path.join(basemaps_dir, name)
            if os.path.isfile(src):
                _link(os.path.abspath(src), os.path.join(basemaps, name))
    return artifacts, basemaps
