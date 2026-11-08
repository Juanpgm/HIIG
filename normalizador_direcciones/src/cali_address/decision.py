"""Calibrated decision and reliability layer (serve time: numpy-free, stdlib only).

Two sign-constrained logistic heads turn the signals available when a row is decided into

* ``P_predio``  - probability that the delivered lot (``NPN[:21]``) is the cadastral identity of the typed address,
* ``P_manzana`` - the same for the delivered manzana, served as ``max(P_manzana, P_predio)``.

The decision policy (:func:`decide_level`) then emits a predial only when the pipeline produced one AND
``P_predio >= tau_p``, else a manzana-only match when ``P_manzana >= tau_m``, else ``SIN_MATCH``. Everything the policy
uses (``tau_p``, ``tau_m``, the floors, the band cut-offs, the neutral rule configuration) comes from ``decision.json``
in the artifact directory; the head parameters come from ``reliability_calibrated.json``. An artifact directory without
BOTH files runs the historical pipeline untouched (see :func:`calibrated_for`).

Files
-----
``reliability_calibrated.json``  ``kind: "logistic_ext"``; the legacy loader (``reliability.load_reliability``) rejects
                                 it on purpose, so the two formats can never be mixed up.
``decision.json``                thresholds and rule configuration, see :func:`validate_decision`.

Loaders never raise: a missing, malformed or inconsistent file gives ``None`` (calibrated mode off). When a
``decision.json`` exists but cannot be used, :func:`calibrated_for` warns once so the fallback is not silent.
"""

from __future__ import annotations

import copy
import json
import math
import os
import warnings
from dataclasses import dataclass, field

__all__ = [
    "BANDS", "CALIBRATED_FILENAME", "DECISION_FILENAME", "ENV_EXCLUDE_FUENTES", "FEATURE_NAMES", "MIN_FLOOR",
    "CATASTRO_FUENTE", "OPTIONAL_FEATURE_NAMES", "CalibratedReliability", "Head", "calibrated_for", "decide_level", "excluded_fuentes", "load_calibrated",
    "load_calibrated_pair", "load_decision", "parse_calibrated", "validate_decision",
]

DECISION_FILENAME = "decision.json"
#: Serving-time switch of the cadastre fallback tier (docs/respaldo-catastro.md): comma-separated ``fuente`` values whose
#: documents are dropped from retrieval, merged with the optional ``exclude_fuentes`` list of ``decision.json``.
ENV_EXCLUDE_FUENTES = "NORMALIZER_EXCLUDE_FUENTES"
CALIBRATED_FILENAME = "reliability_calibrated.json"
KIND = "logistic_ext"

#: Features the heads may use, in the canonical order (names are the JSON keys). Computed at serve time by
#: ``service._calibration_features``; a parameter file naming anything else is rejected.
FEATURE_NAMES = (
    "confidence", "margen_manzana", "margen_predio", "raw_margin_lot", "n_competidores", "log_n_predios", "soft_n",
    "relaxed_plate", "gate_fallback", "fidelity_medium", "fidelity_high", "plate_missing", "cross_missing",
    "src_pcer", "lot_conflict", "cat_verified", "cat_ambiguous", "cat_door_not_catalogued", "guard_one_sided",
    "guard_tramo", "predial_null", "nivel_manzana", "nivel_via", "nivel_esquina",
)

#: Opt-in features: a parameter file may list them (after the 24 above) when it was fitted with them; every file without
#: them keeps loading and predicting exactly as before. ``src_catastro`` is 1.0 when the selected document comes from
#: the cadastre fallback tier (``fuente == "catastro"``), see ``docs/respaldo-catastro.md``.
OPTIONAL_FEATURE_NAMES = ("src_catastro",)
#: ``fuente`` of the documents appended by the cadastre fallback tier (``cali_address.index_fallback``).
CATASTRO_FUENTE = "catastro"

#: Reliability bands of an OK row: ``alta`` and ``media`` carry a predial, ``baja`` is manzana only.
BANDS = ("alta", "media", "baja")

#: Absolute minimum of ``floor`` and ``confianza_floor``: every emitted decision is individually >= 70 % likely.
MIN_FLOOR = 0.70
#: Output probabilities stay strictly inside (0, 1).
P_MIN, P_MAX = 0.001, 0.999

#: ``motivos_confiabilidad`` lists at most this many codes, each with a contribution of at least ``MOTIVO_MIN_ABS`` logit.
MOTIVO_TOP = 3
MOTIVO_MIN_ABS = 0.10

_HEADS = {"predio": "predio", "manzana": "manzana"}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _finite(value) -> float:
    """``float(value)`` or ``ValueError`` when it is not a finite real number (bools and strings are refused)."""
    if isinstance(value, (bool, str)) or value is None:
        raise ValueError("not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def _vector(values, n: int) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != n:
        raise ValueError("wrong vector length")
    return tuple(_finite(v) for v in values)


@dataclass(frozen=True)
class Head:
    """One logistic head over the standardized feature vector (own means / stds, fitted on its own rows)."""

    coef: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    stds: tuple[float, ...]


@dataclass(frozen=True)
class CalibratedReliability:
    """The two heads plus the metadata the motivos need. Build it with :func:`load_calibrated`."""

    feature_names: tuple[str, ...]
    signs: dict
    predio: Head
    manzana: Head
    motivo_codes: dict
    motivo_reference: dict
    clip: float = 5.0
    version: str = ""
    provenance: dict = field(default_factory=dict)

    # -- standardization -------------------------------------------------------------------------------------------
    def _z(self, head: Head, j: int, value) -> float:
        """Standardized, clipped feature ``j``; a missing / ``None`` / NaN / infinite / non-numeric value is the mean."""
        mean, std = head.means[j], head.stds[j]
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = mean
        if not math.isfinite(number):
            number = mean
        z = (number - mean) / std if std > 0 else 0.0
        return max(-self.clip, min(self.clip, z))

    def _logit(self, head: Head, features: dict) -> float:
        total = head.intercept
        for j, name in enumerate(self.feature_names):
            if head.coef[j] != 0.0:
                total += head.coef[j] * self._z(head, j, features.get(name))
        return total

    def predict(self, features: dict) -> tuple[float, float]:
        """``(P_predio, P_manzana)`` in ``[0.001, 0.999]`` with ``P_manzana >= P_predio`` (a right lot implies a right manzana)."""
        pp = min(max(_sigmoid(self._logit(self.predio, features)), P_MIN), P_MAX)
        pm = min(max(_sigmoid(self._logit(self.manzana, features)), P_MIN), P_MAX)
        return pp, max(pm, pp)

    def motivos(self, head: str, features: dict) -> str:
        """The (at most 3) strongest reasons of the delivered level, e.g. ``-fuente_pcer;+margen_predio``.

        ``contribution_j = coef_j * (z(x_j) - z(ref_j))`` with ``ref`` the reference value of the artifact (absence for
        flags and counts, the training mean for ``confidence`` and the margins). Contributions below 0.10 logit and
        zero coefficients are dropped; ordering is by decreasing ``|contribution|``, ties by feature order.
        """
        if head not in _HEADS:
            raise ValueError(f"unknown head {head!r}; valid heads: {', '.join(_HEADS)}")
        block = self.predio if head == "predio" else self.manzana
        found = []
        for j, name in enumerate(self.feature_names):
            coef = block.coef[j]
            if coef == 0.0:
                continue
            contribution = coef * (self._z(block, j, features.get(name)) - self._z(block, j, self.motivo_reference[name]))
            if abs(contribution) >= MOTIVO_MIN_ABS:
                found.append((-abs(contribution), j, ("+" if contribution > 0 else "-") + self.motivo_codes[name]))
        return ";".join(code for _a, _j, code in sorted(found)[:MOTIVO_TOP])


# ---------------------------------------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------------------------------------
def _read_json(path) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_head(block, n: int, signs: list[int]) -> Head:
    if not isinstance(block, dict):
        raise ValueError("head must be an object")
    coef = _vector(block["coef"], n)
    for value, sign in zip(coef, signs):
        if (sign > 0 and value < 0) or (sign < 0 and value > 0):
            raise ValueError("coefficient violates its sign constraint")
    return Head(coef=coef, intercept=_finite(block["intercept"]), means=_vector(block["means"], n),
                stds=_vector(block["stds"], n))


def parse_calibrated(data) -> CalibratedReliability:
    """Validated model from an already-parsed ``reliability_calibrated.json`` document; ``ValueError`` when unusable."""
    if not isinstance(data, dict) or data.get("kind") != KIND:
        raise ValueError("not a logistic_ext artifact")
    names = data["feature_names"]
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not all(isinstance(n, str) and n in FEATURE_NAMES + OPTIONAL_FEATURE_NAMES for n in names)):
        raise ValueError("feature names differ from the serving contract")
    signs = data["signs"]
    if not isinstance(signs, dict) or any(signs.get(n) not in (-1, 0, 1) or isinstance(signs.get(n), bool) for n in names):
        raise ValueError("signs must be -1, 0 or +1 for every feature")
    codes, reference = data["motivo_codes"], data["motivo_reference"]
    if not isinstance(codes, dict) or any(not isinstance(codes.get(n), str) or not codes[n] for n in names):
        raise ValueError("motivo codes must cover every feature")
    if not isinstance(reference, dict):
        raise ValueError("motivo_reference must be an object")
    n = len(names)
    sign_list = [int(signs[name]) for name in names]
    clip = _finite(data.get("clip", 5.0))
    if clip <= 0:
        raise ValueError("clip must be positive")
    return CalibratedReliability(
        feature_names=tuple(names), signs={name: int(signs[name]) for name in names},
        predio=_parse_head(data["predio"], n, sign_list), manzana=_parse_head(data["manzana"], n, sign_list),
        motivo_codes={name: codes[name] for name in names},
        motivo_reference={name: _finite(reference[name]) for name in names},
        clip=clip, version=str(data.get("version", "")),
        provenance=copy.deepcopy(data.get("provenance")) if isinstance(data.get("provenance"), dict) else {},
    )


def load_calibrated(path) -> CalibratedReliability | None:
    """Read ``reliability_calibrated.json``; ``None`` when missing, malformed or inconsistent (never raises)."""
    try:
        return parse_calibrated(_read_json(path))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return None


def _string_leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _string_leaves(k)
            yield from _string_leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _string_leaves(v)


def validate_decision(data) -> dict:
    """Validated deep copy of a ``decision.json`` document; ``ValueError`` (with the reason) when it is not usable.

    Rules: ``version`` / ``reliability_version`` non-empty strings; ``floor`` and ``confianza_floor`` >= 0.70;
    ``floor <= tau_m``; ``floor <= tau_p <= bands.alta <= 1``; ``bands.media == tau_p`` (the band below ``alta`` that
    still carries a predial starts at ``tau_p``); ``neutral.soft_rules`` all known (``inference.SOFT_RULE_NAMES``),
    ``neutral.max_soft`` a non-negative integer, ``neutral.gate_fallback`` a bool, ``neutral.ambiguity_delta`` >= 0;
    optional ``exclude_fuentes`` a list of non-blank strings (``fuente`` values dropped from retrieval at serve time; the
    key is not added when absent, so existing files stay byte-compatible); no string anywhere may parse as an address.
    """
    from .inference import SOFT_RULE_NAMES
    from .parser import parse_address

    if not isinstance(data, dict):
        raise ValueError("decision must be a JSON object")
    out = copy.deepcopy(data)
    for key in ("version", "reliability_version"):
        if not isinstance(out.get(key), str) or not out[key].strip():
            raise ValueError(f"{key} must be a non-empty string")

    def number(key: str, source=out) -> float:
        try:
            return _finite(source[key])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc

    floor, conf_floor = number("floor"), number("confianza_floor")
    tau_p, tau_m = number("tau_p"), number("tau_m")
    bands = out.get("bands")
    if not isinstance(bands, dict):
        raise ValueError("bands must be an object with alta and media")
    alta, media = number("alta", bands), number("media", bands)
    if floor < MIN_FLOOR or conf_floor < MIN_FLOOR:
        raise ValueError(f"floor and confianza_floor must be >= {MIN_FLOOR}")
    if tau_m < floor:
        raise ValueError("tau_m must be >= floor")
    if not floor <= tau_p <= alta <= 1.0:
        raise ValueError("require floor <= tau_p <= bands.alta <= 1")
    if media != tau_p:
        raise ValueError("bands.media must equal tau_p")
    neutral = out.get("neutral")
    if not isinstance(neutral, dict):
        raise ValueError("neutral must be an object")
    rules = neutral.get("soft_rules")
    if not isinstance(rules, list) or any(not isinstance(r, str) for r in rules):
        raise ValueError("neutral.soft_rules must be a list of names")
    unknown = sorted(set(rules) - set(SOFT_RULE_NAMES))
    if unknown:
        raise ValueError(f"unknown soft rule(s) {unknown}; valid names: {', '.join(SOFT_RULE_NAMES)}")
    max_soft = neutral.get("max_soft")
    if isinstance(max_soft, bool) or not isinstance(max_soft, int) or max_soft < 0:
        raise ValueError("neutral.max_soft must be a non-negative integer")
    if not isinstance(neutral.get("gate_fallback"), bool):
        raise ValueError("neutral.gate_fallback must be a bool")
    delta = number("ambiguity_delta", neutral)
    if delta < 0:
        raise ValueError("neutral.ambiguity_delta must be >= 0")
    if "exclude_fuentes" in out:
        excluded = out["exclude_fuentes"]
        if not isinstance(excluded, list) or any(not isinstance(v, str) or not v.strip() for v in excluded):
            raise ValueError("exclude_fuentes must be a list of non-blank strings")
    for text in _string_leaves(out):
        if parse_address(text).parse_ok:
            raise ValueError("a string in the decision file reads as an address; the file must hold no address text")
    return out


def load_decision(path) -> dict | None:
    """Read and validate ``decision.json``; ``None`` when missing, malformed or invalid (never raises)."""
    try:
        return validate_decision(_read_json(path))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return None


def load_calibrated_pair(artifacts_dir) -> tuple[CalibratedReliability, dict] | None:
    """``(model, decision)`` of an artifact directory, or ``None`` (calibrated mode off).

    On only when BOTH files load and ``decision["reliability_version"] == model.version``. A missing directory, a
    missing file or any inconsistency gives ``None``; nothing raises.
    """
    if not isinstance(artifacts_dir, (str, os.PathLike)):
        return None
    model = load_calibrated(os.path.join(artifacts_dir, CALIBRATED_FILENAME))
    decision = load_decision(os.path.join(artifacts_dir, DECISION_FILENAME))
    if model is None or decision is None or decision["reliability_version"] != model.version:
        return None
    return model, decision


def excluded_fuentes(decision: dict | None, environ=None) -> tuple[str, ...]:
    """Sorted ``fuente`` values to drop at serve time: ``decision["exclude_fuentes"]`` plus the environment variable.

    ``environ`` defaults to ``os.environ``. Blank items are ignored; the default (nothing configured) is ``()``, i.e. the
    served index is the full one.
    """
    environ = os.environ if environ is None else environ
    found = {str(v).strip() for v in ((decision or {}).get("exclude_fuentes") or [])}
    found.update(part.strip() for part in str(environ.get(ENV_EXCLUDE_FUENTES) or "").split(","))
    found.discard("")
    return tuple(sorted(found))


_UNSET = object()


def calibrated_for(normalizer) -> tuple[CalibratedReliability, dict] | None:
    """The calibrated ``(model, decision)`` of ``normalizer`` (``None`` = historical behaviour), loaded once.

    An explicit ``normalizer.calibrated`` attribute wins; otherwise the files of ``normalizer.artifacts_dir`` are read
    and the answer is cached on the normalizer. A ``decision.json`` that exists but cannot be used (malformed,
    invalid, version mismatch, reliability file missing) turns calibrated mode off with a ``RuntimeWarning``.
    """
    cached = getattr(normalizer, "calibrated", _UNSET)
    if cached is not _UNSET:
        return cached
    directory = getattr(normalizer, "artifacts_dir", None)
    pair = load_calibrated_pair(directory)
    if pair is None and isinstance(directory, (str, os.PathLike)) and os.path.exists(
            os.path.join(directory, DECISION_FILENAME)):
        warnings.warn(
            f"{DECISION_FILENAME} found in {directory} but the calibrated decision layer cannot be used "
            f"(missing/invalid {CALIBRATED_FILENAME} or decision, or a version mismatch): running the historical pipeline",
            RuntimeWarning, stacklevel=2,
        )
    try:
        normalizer.calibrated = pair
    except (AttributeError, TypeError):
        pass
    return pair


# ---------------------------------------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------------------------------------
def decide_level(p_predio: float, p_manzana: float, emits_predial: bool, decision: dict) -> tuple[str, str | None]:
    """``(outcome, band)`` of one row that reached the decision.

    ``outcome``: ``"predio"`` (the pipeline's predial is kept: it emits one AND ``p_predio >= tau_p``), else
    ``"manzana"`` (predial removed, ``p_manzana >= tau_m``), else ``"sin_match"``. ``band`` is ``alta``
    (``p_predio >= bands.alta``) / ``media`` for ``predio``, ``baja`` for ``manzana`` and ``None`` for ``sin_match``.
    Every boundary is inclusive; a NaN probability fails every comparison, i.e. lands on the safest outcome.
    """
    if emits_predial and p_predio >= decision["tau_p"]:
        return "predio", "alta" if p_predio >= decision["bands"]["alta"] else "media"
    if p_manzana >= decision["tau_m"]:
        return "manzana", "baja"
    return "sin_match", None
