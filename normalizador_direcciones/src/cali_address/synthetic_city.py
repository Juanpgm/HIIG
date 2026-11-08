"""An invented city to exercise the any-city training flow end to end (no real data).

``make_city`` builds a regular street grid that follows the Colombian urban nomenclature (``CL`` / ``KR`` /
``DG`` / ``TV``, ``#`` and ``-``, letters, ``BIS``, the ``SUR`` quadrant) but with names and numbers that
belong to no real city: calles are numbered from :data:`CALLE_BASE` and carreras from :data:`CARRERA_BASE`, above
the numbers used in Cali (99.9 % of its via and cross numbers are below 127), so an address whose via AND cross
are both above 200 does not correspond in practice to a real address of the reference city. Neighbourhood and district names are invented. Parcel codes have the 30-digit layout of the national
predial number with the non-existent municipality code ``99999``. Coordinates are planar metres of an invented
grid placed in EPSG:3116 (MAGNA-SIRGAS / Colombia Bogota zone) so the adapter's CRS handling is exercised.

The output frame deliberately uses column names that differ from the internal schema (``id_registro``,
``dir_texto``, ``coord_x`` ...): :func:`default_mapping` is the declarative mapping that
:mod:`cali_address.city_adapter` needs to read it. Everything is deterministic for a given ``seed``.
"""

from __future__ import annotations

import math
import random

import pandas as pd

__all__ = ["CALLE_BASE", "CARRERA_BASE", "SOURCE_CRS", "make_city", "default_mapping", "grid_for_size"]

#: First calle / carrera number of the invented grid (well above any number used in the reference city).
CALLE_BASE = 200
CARRERA_BASE = 300
#: CRS of the ``coord_x`` / ``coord_y`` columns.
SOURCE_CRS = "EPSG:3116"
#: Invented origin of the grid in :data:`SOURCE_CRS` (metres).
_ORIGIN_X, _ORIGIN_Y = 1_150_000.0, 980_000.0
_BLOCK_M = 100.0
_MUNICIPALITY = "99999"

_NAME_PREFIX = ("El", "La", "Los", "Las", "San", "Villa", "Alto", "Bajo")
_NAME_WORD = ("Mirador Ocre", "Bruma", "Arrecife", "Cedral", "Ventisca", "Alcor", "Pradera Gris", "Candil",
              "Sauce Rojo", "Olmeda", "Cuarzo", "Madrigal", "Nogal Azul", "Penumbra", "Jaral", "Albarda",
              "Ceibal", "Brezal", "Remanso", "Atalaya")
_DISTRICT_WORD = ("Norte", "Sur", "Oriente", "Occidente", "Centro", "Ribera", "Loma", "Llano", "Vega")


def _neighbourhood_name(index: int) -> str:
    prefix = _NAME_PREFIX[index % len(_NAME_PREFIX)]
    word = _NAME_WORD[(index // len(_NAME_PREFIX)) % len(_NAME_WORD)]
    cycle = index // (len(_NAME_PREFIX) * len(_NAME_WORD))
    return f"{prefix} {word}" + (f" {cycle + 1}" if cycle else "")


def _district_name(index: int) -> str:
    word = _DISTRICT_WORD[index % len(_DISTRICT_WORD)]
    cycle = index // len(_DISTRICT_WORD)
    return f"Distrito {word}" + (f" {cycle + 1}" if cycle else "")


def grid_for_size(n_addresses: int, plates_per_face: int = 3) -> tuple[int, int]:
    """``(n_rows, n_cols)`` of a roughly square grid that yields about ``n_addresses`` base addresses."""
    if n_addresses < 1:
        raise ValueError("n_addresses must be >= 1")
    if plates_per_face < 1:
        raise ValueError("plates_per_face must be >= 1")
    cells = max(1.0, n_addresses / (2.0 * plates_per_face))     # two street faces per grid cell
    side = max(2, int(math.ceil(math.sqrt(cells))))
    return side, side


def _street(kind: str, index: int, south_rows: int) -> tuple[str, str]:
    """``(type, number text)`` of grid line ``index`` (``kind`` is ``row`` for calles, ``col`` for carreras)."""
    if kind == "row":
        if index < south_rows:                       # south of the base calle: CL n SUR
            number = f"{CALLE_BASE + south_rows - index} SUR"
            return "CL", number
        n = CALLE_BASE + index - south_rows
        suffix = " A" if index % 4 == 3 else ""      # every fourth calle is an "A" street
        return ("TV" if index % 11 == 10 else "CL"), f"{n}{suffix}"
    n = CARRERA_BASE + index
    suffix = " BIS" if index % 5 == 4 else ""
    return ("DG" if index % 9 == 8 else "KR"), f"{n}{suffix}"


def make_city(
    n_addresses: int = 2000,
    seed: int = 7,
    plates_per_face: int = 3,
    unit_share: float = 0.08,
    provisional_share: float = 0.03,
    void_share: float = 0.01,
) -> pd.DataFrame:
    """The invented city's address records, one row per record (base addresses plus a few interior units).

    Columns: ``id_registro``, ``dir_texto``, ``coord_x``, ``coord_y`` (EPSG:3116), ``codigo_predial`` (30
    digits), ``manzana_cod``, ``barrio_nombre``, ``distrito_nombre``, ``estado_registro`` (``vigente`` /
    ``provisional`` / ``anulado``) and ``unidad`` (interior such as ``AP 201``, blank for base records).
    """
    if not 0.0 <= unit_share <= 1.0 or not 0.0 <= provisional_share <= 1.0 or not 0.0 <= void_share <= 1.0:
        raise ValueError("shares must be in [0, 1]")
    rng = random.Random(seed)
    n_rows, n_cols = grid_for_size(n_addresses, plates_per_face)
    south_rows = max(1, n_rows // 5)
    hood_cells = 3                                   # a neighbourhood covers 3 x 3 grid cells
    hoods_per_row = (n_cols + hood_cells - 1) // hood_cells
    district_hoods = 3                               # a district covers 3 x 3 neighbourhoods

    records: list[dict] = []
    terrain: dict[str, int] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            hr, hc = r // hood_cells, c // hood_cells
            hood = hr * hoods_per_row + hc
            district = (hr // district_hoods) * ((hoods_per_row + district_hoods - 1) // district_hoods) + hc // district_hoods
            block = f"{r:03d}{c:03d}"
            comuna_code = f"{district % 100:02d}"
            barrio_code = f"{hood % 100:02d}"
            block_code = f"{(r * n_cols + c) % 10000:04d}"
            for face in ("row", "col"):
                for k in range(plates_per_face):
                    if face == "row":                # on the calle of row r, between carrera c and c + 1
                        vtype, vnum = _street("row", r, south_rows)
                        _, cross = _street("col", c, south_rows)
                        along = (k + 1) * _BLOCK_M / (plates_per_face + 1) + rng.uniform(-6, 6)
                        x, y = c * _BLOCK_M + along, r * _BLOCK_M + rng.choice((-6.0, 6.0))
                    else:                            # on the carrera of column c, between calle r and r + 1
                        vtype, vnum = _street("col", c, south_rows)
                        _, cross = _street("row", r, south_rows)
                        along = (k + 1) * _BLOCK_M / (plates_per_face + 1) + rng.uniform(-6, 6)
                        x, y = c * _BLOCK_M + rng.choice((-6.0, 6.0)), r * _BLOCK_M + along
                    plate = max(2, min(98, int(round(along))))
                    text = f"{vtype} {vnum} # {cross} - {plate:02d}"
                    key = f"{comuna_code}{barrio_code}{block_code}"
                    terrain[key] = terrain.get(key, 0) + 1
                    npn = f"{_MUNICIPALITY}0100{key}{terrain[key] % 10000:04d}000000000"
                    roll = rng.random()
                    estado = "anulado" if roll < void_share else (
                        "provisional" if roll < void_share + provisional_share else "vigente")
                    base = {
                        "dir_texto": text, "coord_x": _ORIGIN_X + x, "coord_y": _ORIGIN_Y + y,
                        "codigo_predial": npn, "manzana_cod": block, "barrio_nombre": _neighbourhood_name(hood),
                        "distrito_nombre": _district_name(district), "estado_registro": estado, "unidad": "",
                    }
                    records.append(base)
                    if rng.random() < unit_share:
                        for floor in (1, 2):
                            unit = f"AP {floor}0{1 + k}"
                            records.append({**base, "dir_texto": f"{text} {unit}", "unidad": unit,
                                            "codigo_predial": npn[:21] + "9" + f"{floor:02d}{floor:02d}{k + 1:04d}",
                                            "coord_x": base["coord_x"] + rng.uniform(-1, 1),
                                            "coord_y": base["coord_y"] + rng.uniform(-1, 1)})
    frame = pd.DataFrame(records)
    frame.insert(0, "id_registro", [f"SC-{i:07d}" for i in range(1, len(frame) + 1)])
    return frame


def default_mapping(input_path: str, city: str = "Ciudad Sintetica") -> dict:
    """The declarative mapping (see :mod:`cali_address.city_adapter`) for a file written from :func:`make_city`."""
    return {
        "city": city,
        "input": input_path,
        "crs": SOURCE_CRS,
        "columns": {
            "address": "dir_texto", "record_id": "id_registro", "x": "coord_x", "y": "coord_y",
            "parcel_id": "codigo_predial", "block": "manzana_cod", "neighborhood": "barrio_nombre",
            "district": "distrito_nombre", "status": "estado_registro", "unit": "unidad",
        },
        "status_values": {"cert": ["vigente"], "pcer": ["provisional"]},
        "max_km_from_median": 30,
    }
