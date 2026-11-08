"""Synthetic nomenclature rows (cleaned-WFS layout) for the index-union tests.

Every address is invented with improbable numbers (900+); no real citizen data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

#: Columns a cleaned WFS frame carries that the index builder reads.
NOM_COLUMNS = [
    "gid", "ndid", "ndcomuna", "ndidbarrio", "ndmanzana", "ndnpn", "nddirecc", "ndinterior", "nddircons2",
    "ndidestado", "ndidtipzon", "lon", "lat", "npn_valid", "npn_len",
]


def npn(manzana: str = "0956", lot: str = "0917", unit: str = "000000001", comuna: str = "08", barrio: str = "10") -> str:
    """A 30-character NPN: 76001 01 00 <comuna> <barrio> <manzana> <lot> <unit>."""
    value = f"76001010 0{comuna}{barrio}{manzana}{lot}{unit}".replace(" ", "")
    assert len(value) == 30, value
    return value


def row(gid: int, address: str, *, interior=None, estado: str = "cert", npn_value: str | None = None,
        lon: float | None | str = "auto", lat: float | None | str = "auto", cons2=None, comuna=None, barrio=None,
        manzana=None, zona: str = "URB", ndid: int | None = None) -> dict:
    """One cleaned-WFS-style row. ``npn_value=None`` derives a valid NPN from the gid."""
    value = npn(unit=f"{gid:09d}") if npn_value is None else npn_value
    lon = -76.5 - gid * 1e-5 if isinstance(lon, str) else lon
    lat = 3.4 + gid * 1e-5 if isinstance(lat, str) else lat
    derived = value if len(value) >= 17 else ""
    return {
        "gid": gid,
        "ndid": gid if ndid is None else ndid,
        "ndcomuna": derived[9:11] if comuna is None else comuna,
        "ndidbarrio": derived[9:13] if barrio is None else barrio,
        "ndmanzana": derived[13:17] if manzana is None else manzana,
        "ndnpn": value,
        "nddirecc": address,
        "ndinterior": interior,
        "nddircons2": cons2 if cons2 is not None else address.replace("#", " ").replace(" - ", " "),
        "ndidestado": estado,
        "ndidtipzon": zona,
        "lon": lon,
        "lat": lat,
        "npn_valid": len(value) == 30 and value.startswith("76001"),
        "npn_len": len(value),
    }


def nom_frame(*rows: dict) -> pd.DataFrame:
    """Frame with the nullable dtypes of ``artifacts/nomenclatura_clean.parquet``."""
    df = pd.DataFrame(list(rows), columns=NOM_COLUMNS)
    df["gid"] = df["gid"].astype("Int64")
    df["ndid"] = df["ndid"].astype("Int64")
    for col in ("lon", "lat"):
        df[col] = df[col].astype("float64")
    df["npn_valid"] = df["npn_valid"].astype(bool)
    df["npn_len"] = df["npn_len"].astype("int64")
    return df


def catastro_docs_frame(entries: list[tuple[str, str]], start_oid: int = 70001) -> pd.DataFrame:
    """Catastro-style docs from ``(direccion, npn)`` pairs, columns/dtypes of ``catastro_docs.parquet``."""
    rows = []
    for i, (address, value) in enumerate(entries):
        rows.append({
            "direccion": address,
            "numero_predial_nacional": value,
            "manzana": value[:17],
            "comuna": value[9:11],
            "barrio": value[11:13],
            "barrio_code": value[9:13],
            "centroid_lon": -76.4 - i * 1e-4,
            "centroid_lat": 3.3 + i * 1e-4,
            "n_predios": 1,
            "manzana_nunique": 1,
            "objectid": start_oid + i,
            "x_m": 0.0,
            "y_m": 0.0,
            "doc_id": i,
        })
    df = pd.DataFrame(rows, columns=["direccion", "numero_predial_nacional", "manzana", "comuna", "barrio",
                                     "barrio_code", "centroid_lon", "centroid_lat", "n_predios", "manzana_nunique",
                                     "objectid", "x_m", "y_m", "doc_id"])
    for col in ("direccion", "numero_predial_nacional", "manzana", "comuna", "barrio", "barrio_code"):
        df[col] = df[col].astype(str)
    return df


def catastro_parcels_frame(docs: pd.DataFrame) -> pd.DataFrame:
    """A ``catastro.parquet``-style table: one parcel per doc, own centroid slightly off the doc one."""
    return pd.DataFrame({
        "OBJECTID": docs["objectid"].to_numpy(),
        "numero_predial_nacional": docs["numero_predial_nacional"].to_numpy(),
        "centroid_lon": docs["centroid_lon"].to_numpy() + 1e-5,
        "centroid_lat": docs["centroid_lat"].to_numpy() + 1e-5,
    })


def nan_if_none(values):
    return np.array([np.nan if v is None else v for v in values], dtype="float64")
