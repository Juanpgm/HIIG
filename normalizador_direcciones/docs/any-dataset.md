# Normalizing any dataset

`python -m cali_address` reads addresses from almost any table (file, URL or
database), normalizes them against the Cali cadastre with the same engine the API
uses, and writes the result to almost any format. Your original columns are kept;
the normalization columns are appended.

```
python -m cali_address setup                  # first run: regenerate catastro_emb.pt, then self-check
python -m cali_address normalize INPUT [-o OUTPUT] [options]
python -m cali_address web                    # browser upload page (needs the "api" extra)
python -m cali_address inspect INPUT          # what is in this file? (does not normalize)
python -m cali_address formats                # supported input / output formats
```

Install with `pip install -e .` (see the README) and use the `cali-address` command,
or run `python -m cali_address` from a checkout with `PYTHONPATH=src`. The examples
below use `python -m cali_address`; `cali-address` is an exact alias.
User errors print one `error:` line with the fix, never a traceback.

**Default output.** Without `-o`, `normalize` writes `<input stem>_normalizado<ext>` next to a local input, in
the same format (`.xls` -> `.xlsx`, `.txt` -> `.csv`, shapefile / zip / GeoPackage -> `.geojson`). The input is
never overwritten (the suffix always differs); an existing output file is replaced atomically. URL and SQL
inputs have no local name and need `-o` (exit 2). `--dry-run N` still writes nothing.

| Exit code | Meaning |
|-----------|---------|
| `0` | ok (rows with `ERROR` are allowed as long as at least one row was processed normally, and an input with zero rows is fine) |
| `1` | aborted: `--on-error raise` hit a failing row, or the run ran out of memory (nothing is written) |
| `2` | usage / column-mapping / config error, including an invalid or unparseable database URL |
| `3` | I/O error: missing or corrupt input, unreachable database, unwritable output, missing model artifacts |
| `4` | the run finished but **every** row is `ERROR` (typically a broken model); the output and the summary are still written |
| `130` | interrupted (Ctrl+C) |

## Quickstart by source

Always start with `inspect` when you do not know the file: it shows the detected
format, delimiter, encoding, header row, the columns, the guessed address column
and the first 5 rows.

```
python -m cali_address inspect data.csv
```

**CSV with an odd delimiter or encoding.** Delimiter (`, ; TAB |`) and encoding
(BOM, UTF-8, cp1252/latin-1, UTF-16) are sniffed. Override when the sniffing is wrong:

```
python -m cali_address normalize data.csv -o out.csv
python -m cali_address normalize data.txt -o out.csv --delimiter "|" --encoding cp1252
python -m cali_address normalize list.txt -o out.csv     # .txt with no delimiter: one address per line
```

**Excel with the header on row 3.** `--header-row` is 0-based (row 3 is `2`). When omitted
the header row is detected (works when the header is the densest of the first 15
rows). Only the first sheet is read unless you pass `--sheet`.

```
python -m cali_address normalize report.xlsx -o out.xlsx --sheet Hoja1 --header-row 2
```

**Address split across columns.** Parts are joined in the given order with
`--parts-sep` (default one space); blank parts are skipped, integral numbers lose the
`.0`.

```
python -m cali_address normalize data.csv -o out.csv --address-parts via,numero,complemento
python -m cali_address normalize data.csv -o out.csv --address-parts via,numero --parts-sep " "
```

**Database (SQLAlchemy URL).** Use `--table [schema.]name`; the table can be a real table or a **view**.
There is deliberately no free-form SQL option: a text filter cannot be a security boundary for SQL, so
for joins or filters create a read-only `VIEW` in the database and read it with `--table`. Rows are
fetched in chunks through a server-side cursor. Table and schema names (letters, digits and `_`,
not starting with a digit, at most one `schema.` prefix) are validated with a strict pattern and rendered by
SQLAlchemy with identifier quoting. Identifiers are quoted exactly as given, so the case matters: on
PostgreSQL use the lower-case name unless the table was created with quotes (an unquoted `CREATE TABLE
Predios` is stored as `predios`). A `sqlite:///path` file that does not exist is an I/O error (exit 3); a
read never creates it.
The session is opened read-only where the driver supports it (SQLite: `PRAGMA query_only=ON` on every
pooled connection; PostgreSQL through `psycopg2`, `psycopg` and `pg8000`: a read-only transaction) and
every run ends with a rollback, never a commit. Other databases and drivers get no read-only session (no
error either), so connect with a **read-only database account** whatever the database. Passwords and secret query parameters (`token`,
`password`, `api_key`, `key`, `secret`, `sig`, `auth`, ...), raw and URL-encoded, are masked in every message, with one residual: a 1-2 character password is masked only
inside a URL (`user:pw@host`), not in free text, because masking it there would shred the message. A driver
message that echoes such a password outside a URL is therefore NOT redacted; use a password of at least 3
characters, ideally a proper secret. An unparseable URL is a usage error (exit 2) and an unreachable
database is an I/O error (exit 3). Drivers are optional: `pip install sqlalchemy` plus the driver
(`psycopg2-binary` for Postgres); a missing one gives an install hint.

```
python -m cali_address normalize sqlite:///local.db -o out.parquet --table direcciones
# joins / filters: CREATE VIEW predios_cali AS SELECT id, via, numero FROM predios WHERE municipio = 'Cali';
python -m cali_address normalize postgresql://user:pw@host/db -o out.csv \n    --table public.predios_cali --address-parts via,numero
```

**GeoJSON, shapefile, GeoPackage.** GeoJSON (`FeatureCollection`) and shapefile
(`.shp` beside its `.shx/.dbf/.prj`, or a `.zip`) use the existing dependencies
(`shapely`, `pyshp`, `pyproj`); each feature's centroid becomes `_lon` / `_lat` in
the passthrough columns (informational). `.gpkg` needs `pip install pyogrio geopandas`
(`--table` selects the layer).

```
python -m cali_address normalize points.geojson -o out.geojson --address-col direccion
```

Other inputs: `.tsv`, `.parquet`, `.json` (list of objects, `{"records": [...]}` or
a FeatureCollection), `.jsonl`, `.xls` (needs `xlrd`), http(s) URLs (downloaded to a
temp file first), and `--format` to override the extension. Outputs: `.csv` (UTF-8
with BOM), `.tsv`, `.xlsx` (sheets `normalizado` / `revisar` / `resumen`), `.parquet`,
`.json`, `.jsonl`, `.geojson` (a Point per located row, `null` geometry otherwise).

## Column mapping

| Option | Meaning |
|---|---|
| `--address-col NAME` | Address column. Auto-detected (Spanish/English synonyms) when omitted. If two columns are equally plausible (`direccion` and `domicilio`) the command stops and lists the candidates: it never guesses. |
| `--address-parts A,B,C` | Join several columns into one address. Mutually exclusive with `--address-col`. |
| `--id-col NAME` | Always carried to the output, even with `--keep-columns none`. |
| `--municipality-col NAME` | Out-of-area guard (see below). Off unless given. |
| `--lat-col` / `--lon-col` | Validated and reported only. Coordinates never influence matching. |
| `--keep-columns` | `all` (default), `none`, or a comma list of input columns to carry. |

A missing or unknown column fails before the model is loaded and lists the detected
columns.

## Output columns

Your columns first (all of them by default), then these, always in this order.
A source column whose name equals an output column (or a coordinate-like name such as
`lat`, `lon`, `latitud`, `longitud`, which are renamed as a pair) is carried as
`<name>_original`. When the address is a single kept column, `direccion_entrada` is
omitted because it would duplicate that column.

| Column | Type | Meaning | Null / empty when |
|---|---|---|---|
| `direccion_entrada` | text | The address text received (joined text for `--address-parts`). | never; omitted with a single kept address column |
| `estado` | text | `OK`, `SIN_MATCH`, `NO_PARSEABLE`, plus `FUERA_DE_AREA` and `ERROR` (below). | never |
| `motivo` | text | Why the row was not OK. On OK rows it is empty or a note: `aproximado: ...` and / or `direccion compartida por N predios`. A row rejected only because a numeric street suffix is present on one side (`KR 31 F` vs `KR 31 F 3`) ends with the stable token `regla: sufijo unilateral`. | empty on a plain OK row |
| `direccion_normalizada` | text | The matched cadastral base address (OK) or the deterministic canonical form (`reglas`), followed by the input's canonical units (`complemento_normalizado`, same spelling on both). | blank/unparseable input, `ERROR`, `FUERA_DE_AREA` |
| `fuente_normalizacion` | text | `catastro` (real match), `reglas` (grammar only) or empty. | when `direccion_normalizada` is null |
| `numero_predial_nacional` | text | Predial number of the matched predio; null (never the text `None`) when the cadastre has none, flagged in `motivo`. A digit-led code that is not 30 digits is flagged (`NPN anomalo en catastro (...)`), not dropped. | non-OK rows |
| `manzana` | text | Cadastral block code. | non-OK rows |
| `lat`, `lon` | float | Centroid of the predio whose NPN is reported (EPSG:4326). On a shared address that is the first predio by OBJECTID, not the median of all of them. The parcel's own point comes from `catastro_own_centroids.parquet` (shipped in the image) or `catastro.parquet`; with neither, the address centroid (the median over its parcels) is kept and `motivo` says `coordenada: centroide de la direccion compartida por N predios`. | non-OK rows |
| `confianza` | float 0-1 | Fused match score, uncalibrated. | non-OK rows |
| `confiabilidad` | float 0-1 | Calibrated probability the predio is right. Informational; never changes `estado`. | non-OK rows, or if `reliability.json` is absent |
| `confiabilidad_manzana` | float 0-1 | Same for the manzana (always >= `confiabilidad`). | same as above |
| `margen` | float | Score gap to the best rule-passing candidate in ANOTHER manzana; small = ambiguous. | non-OK rows, or no competitor |
| `nivel_precision` | text | `predio` (via, cross and plate confirmed by the text), `via` (no cross number), `esquina` (no plate), `manzana` (plate off by 1-2, a street letter or a quadrant present on one side only, or a shared address within a block), `direccion` (shared across blocks, or null NPN). A row with any of these approximations is never `predio`; `motivo` says which (`aproximado: ...; nivel maximo <level>: <reason>`). | non-OK rows |
| `barrio_vereda` | text | Official IDESC barrio / vereda. For OK rows from the matched predio; otherwise from a place detected in the text. | nothing known |
| `comuna_corregimiento` | text | `Comuna 19` style name or corregimiento. Same sourcing. | nothing known |
| `complemento_normalizado` | text | The unit part of `direccion_normalizada` (e.g. `TO F AP 101`), taken from the input and written canonically; identical on OK and non-OK rows. | the input has no units |
| `position_tier` | text | How well the row is located: `predio` (parcel centroid, radius 15 m), `lote` (lot centroid; a site unit differs from the cadastre, predial withheld, 44 m), `tramo` (coordinates interpolated between the neighbouring parcels of the same street face, predial withheld, 15 to 23 m depending on the gap), `manzana` (best candidate point, approximate, 56 m), `via` (plate beyond the last known parcel, truncated parse or high fidelity risk: predial, manzana and coordinates withheld unless the service enables `EXTRAPOLATED_COORDS_ENABLED`; 21 m) or `none`. | never (`none` when there is no position) |
| `position_radius_m` | float | Geometric error radius of the tier in metres (p90, derived from the cadastre itself, no GPS). | tier `none` |
| `position_source` | text | Where the coordinate comes from: `parcel`, `interp` (interpolated / extrapolated) or `none`. | never |
| `selected_doc_id` | int | Index of the chosen cadastral document (audit). | non-OK rows |
| `override_rejected` | bool | True when the retrieved candidate was rejected by the structural guard because its base differs from the input address. | never (False by default) |
| `direccion_catastral` | text | The matched parcel's address exactly as the cadastre writes it (OK rows only, for audit). | non-OK rows |
| `fidelity_risk` | text | Risk that `direccion_normalizada` is a DIFFERENT address from the input: `none`, `low`, `medium` or `high`. `high` caps the tier at `via` and blanks the address; `medium` caps it at `manzana` (no predial). | never |
| `fidelity_flags` | text | `;`-joined flags behind `fidelity_risk` (e.g. `plate_changed;letters_dropped`); `no_address` when there is no output address. | no findings |
| `catalog_status` | text | `verified`, `ambiguous`, `door_not_catalogued`, `rejected_mismatch`, `unparsed` or `below_cutoff` (decided by precedence; see `documentacion/06-predicciones.md`, section 10.4). | never |
| `address_verified` | bool | True exactly when `catalog_status` is `verified`. A plate mismatch is never a verified door. | never |

`ERROR` and `FUERA_DE_AREA` rows carry `direccion_entrada` and `motivo` only; every
other result column is empty. `direccion_normalizada` (and `fuente_normalizacion`) is also left empty
on rows whose rule text is not trustworthy: unparsed rows, below-cutoff or rejected rows with an unsafe parse,
and every row with `fidelity_risk = high`; the file pipeline then falls back to the raw text.

## Error semantics

* **One bad row never aborts the run.** If the normalizer raises for a row, that row
  becomes `estado='ERROR'`, `motivo='<ExceptionType>: <short message>'`, and the rest
  of its chunk is still scored normally (the failing batch is bisected to isolate the
  offending rows). The summary counts them and prints up to 5 samples. `--on-error raise`
  aborts instead.
* **Blank, placeholder and non-text values** (`None`, `NaN`, `N/A`, dates, numbers)
  never crash: they become `SIN_MATCH` / `NO_PARSEABLE` like any other unusable address.
  Text formats are read as text, so ids like `007` and addresses like `NA` are preserved.
* **Out-of-area guard.** With `--municipality-col`, a non-empty municipality that is not
  Cali is reported as `FUERA_DE_AREA` without matching. Accepted as Cali (accent and case
  insensitive, whole words only): `Cali`, `Santiago de Cali`, `Santiago de Cali D.E.`,
  `Distrito Especial de Santiago de Cali`, `Cali - Valle`, `Cali, Valle del Cauca`, and the DANE
  code as text or number (`76001`, `76001.0`, `76001000`). Words such as `Calima` or `Caliente`
  are other municipalities. Empty or placeholder values (and a bare `Valle del Cauca`) are not guarded.
* **Order and count** of the input are preserved for any `--chunk-size`; output is
  deterministic. The output file is written to a temporary file next to it and moved into
  place only on success, so a failed run never leaves a truncated file or clobbers a previous
  one. **An existing output file is overwritten** on success (no prompt, no backup), even when
  `-o` is the input path itself: the input is closed before the replacement.
* Matching results are identical to the legacy `scripts/normalizar.py` and the API for the
  same rows and options (the run is `normalize_strict` on every chunk).

## Run summary

Printed to stderr and, with `--summary-json PATH`, saved: `rows`, `ok`, `sin_match`,
`no_parseable`, `fuera_de_area`, `error`, `by_estado`, `by_nivel_precision`, `chunks`,
`seconds`, `rows_per_sec`, `error_samples`, the gazetteer gate counters, reader warnings
and the resolved column mapping. `--dry-run N` processes only the first N rows, prints
them and writes nothing.

## Config file

`--config run.toml` accepts the same options (`-` or `_`), flat or grouped in tables.
Precedence: defaults < config file < command line. Unknown keys and malformed TOML are
errors (exit 2). Relative paths inside the file (`input`, `output`, `summary_json`,
`artifacts_dir`, `basemaps`) are resolved against the **config file's directory**, so the file
works from any working directory; URLs and absolute paths are left as written. Paths given on the
command line stay relative to the current directory and win over the file.

```toml
input = "reporte.xlsx"
output = "salida.csv"
sheet = "Hoja1"
header_row = 2
address_parts = ["via", "numero", "complemento"]
id_col = "id"
municipality_col = "municipio"
chunk_size = 5000

[tunables]
min_struct = 0.6
ambiguity_delta = 0.02
```

The tunables are the same as the legacy CLI: `--min-struct`, `--plate-tolerance`,
`--ambiguity-delta`, `--barrio-buffer`, `--zone-buffer`, `--gate-escalate`,
`--gazetteer/--no-gazetteer`, `--soft-rules`, `--max-soft`, `--gate-fallback`,
`--threshold`, `--device`, `--artifacts-dir`, `--basemaps`. Defaults equal today's
CLI and API behaviour.

Environment: the CLI reads `CALI_ARTIFACTS_DIR` for the default `--artifacts-dir`. The HTTP API
and the Docker image use a different variable, `ARTIFACTS_DIR`; setting one does not affect the
other.

`--artifacts-dir` is trusted local input: `model.pt` is loaded with
`torch.load(weights_only=False)`, which can execute code from the file. Never point it at
artifacts from an untrusted source.

## Performance

* `--chunk-size` (default 20 000) bounds memory: CSV/TSV/TXT, XLSX (read-only mode),
  Parquet, JSONL, `.shp` and SQL stream, so files larger than RAM work. `.xls`, `.json`,
  GeoJSON, zipped shapefiles and GPKG are read whole and then sliced. The `.xlsx` output is
  buffered (Excel caps a sheet at 1 048 575 rows); use csv or parquet for huge results.
* Chunk size never changes the results. Larger chunks amortize model batching (better on
  GPU); smaller ones give steadier progress. If a chunk fails, only the bisection retries
  cost extra.
* Encoding detection scans the file once for UTF-8 validity before reading.

## Limits

* The reference catalog is the **Cali cadastre** (IDESC / catastro parquet) and the model
  is trained on its address grammar. Addresses from another city will be matched against
  Cali predios, which is wrong. Another city needs its own catalog, gazetteer and a
  retrained model (see `docs/model-artifacts.md`); this layer only removes the file-format
  and column-mapping obstacles.
* Coordinates in your data are never used to decide a match.
* Header auto-detection is a heuristic; when `inspect` shows a wrong header, pass
  `--header-row`.
* A CSV row with more non-empty fields than the header is an error (exit 3, with the line number), never silently truncated; short rows are padded and trailing separators ignored.
* Zipped shapefiles and other archives are read whole, with **no cap on the decompressed size**
  (a zip bomb is not detected); only feed the CLI files you trust. Known limitation.
* A `.txt` with no delimiter is treated as one address per line without a header.
* The HTTP API keeps its own upload path (three statuses only); it does not emit `ERROR` /
  `FUERA_DE_AREA`.
