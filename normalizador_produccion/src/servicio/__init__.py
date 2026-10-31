"""Thin production entrypoint around ``cali_address`` (ONNX Runtime backend, no torch).

``python -m servicio`` validates the environment, makes sure the index bundle is present (or downloads
it), assembles the runtime artifacts directory and starts uvicorn on ``servicio.app:app``.
``python -m servicio check`` runs the same validation and exits without serving.
"""

__version__ = "0.2.0"
