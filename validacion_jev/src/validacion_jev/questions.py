"""JEV question definitions. Instructions are data (Spanish) and are sent verbatim."""
from __future__ import annotations

import copy

NORMALIZER = "normalizador"
MATCHER = "matcher"
KINDS = (NORMALIZER, MATCHER)

PRIMARY_QUESTION = {NORMALIZER: "misma_ubicacion", MATCHER: "misma_puerta"}

_NORMALIZER_QUESTIONS = {
    "misma_ubicacion": {
        "type": "noul",
        "instructions": (
            "¿La `direccion_normalizada` designa la misma puerta o predio que la "
            "`direccion_original` (nomenclatura urbana colombiana)? Ignora diferencias de "
            "abreviaturas, tildes, mayúsculas y puntuación."
        ),
        "criteria": {
            "true": "Misma vía, mismos números de vía, de cruce y de placa; solo cambia el formato",
            "false": (
                "Cambia algún número, letra, sufijo o tipo de vía, o falta o sobra información "
                "que altera la ubicación"
            ),
        },
    },
}

_NORMALIZER_ESTADO_QUESTION = {
    "type": "choice",
    "instructions": (
        "El campo `estado` es el resultado que el sistema asignó al procesar la "
        "`direccion_original`. Evalúa si ese estado era el adecuado para esta entrada."
    ),
    "criteria": {
        "adecuado": "El estado asignado es coherente con la entrada.",
        "debio_resolver": (
            "El sistema respondió NO_MATCH o SIN_MATCH aunque la dirección de entrada era "
            "resoluble."
        ),
        "debio_rechazar": (
            "El sistema devolvió una coincidencia aunque la dirección de entrada estaba "
            "demasiado incompleta o era ambigua."
        ),
    },
}

_MATCHER_QUESTIONS = {
    "misma_puerta": {
        "type": "noul",
        "instructions": (
            "¿Las dos direcciones canónicas (`direccion_a` y `direccion_b`) designan la misma "
            "puerta o unidad (nomenclatura urbana colombiana)?"
        ),
        "criteria": {
            "true": "Misma vía, mismos números de cruce y de placa, y misma unidad si se indica",
            "false": "Difiere la vía, el cruce, la placa o la unidad",
        },
    },
    "relacion": {
        "type": "choice",
        "instructions": "Clasifica la relación entre `direccion_a` y `direccion_b`.",
        "criteria": {
            "misma_puerta": "Ambas designan exactamente la misma puerta o unidad.",
            "mismo_edificio_distinta_unidad": (
                "Mismo predio o edificio, pero distinta unidad (apartamento, local, oficina)."
            ),
            "misma_via_distinta_puerta": "Misma vía, pero otra puerta o placa.",
            "distinta_via": "Las direcciones están en vías distintas.",
        },
    },
}


def questions_for(kind: str, include_estado: bool = False) -> dict[str, dict]:
    """Return a fresh copy of the question set for ``kind``."""
    if kind == NORMALIZER:
        qs = copy.deepcopy(_NORMALIZER_QUESTIONS)
        if include_estado:
            qs["estado_adecuado"] = copy.deepcopy(_NORMALIZER_ESTADO_QUESTION)
        return qs
    if kind == MATCHER:
        return copy.deepcopy(_MATCHER_QUESTIONS)
    raise ValueError(f"Tipo de validacion desconocido: {kind!r}")
