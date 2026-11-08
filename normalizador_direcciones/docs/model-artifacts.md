# Model artifacts

## The model

A character-level Transformer dual encoder (4 layers, `d_model` 256, 8 heads, 256-d output, 3,375,601
parameters, max length 40). Address strings and cadastral documents are embedded into the same space and matched by
cosine similarity, then re-ranked by structural and geographic rules (`cali_address.service.normalize_strict`).
It is trained on a **synthetic cadastre**: each cadastral address is corrupted by augmentation operators to simulate
field text. No real citizen data is used by the production recipe.

Held-out synthetic validation recall (`artifacts/train_report.json`, 8 epochs, about 16 min on one GPU):

| recall@1 | recall@5 | recall@20 |
|----------|----------|-----------|
| 0.9352 | 0.9637 | 0.9869 |

## Tracked in git

| File | Size | Why tracked |
|------|------|-------------|
| `artifacts/model.pt` | 14 MB | The trained weights |
| `artifacts/tuning.json` | 5 KB | Fusion weights, threshold, decision options |
| `artifacts/reliability.json` | 4 KB | Calibrated confidence model |
| `artifacts/train_report.json` | 5 KB | Training history and validation recall |
| `artifacts/gazetteer.pkl` | 5 MB | Parsed public basemaps (cache) |
| `artifacts/catastro_docs.parquet` | 21 MB | Public cadastre: address, parcel id, block, barrio, centroid (14 columns, 330,387 rows, no personal data) |
| `artifacts/catastro_own_centroids.parquet` | 8 MB | Centroid of the parcel each document reports (`objectid`, `numero_predial_nacional`, `own_lon`, `own_lat`; one row per document, same order). Public data. Built by `scripts/build_own_centroids.py` |
| `artifacts/experiments/leaderboard.csv` | 3 KB | One metrics row per experiment |
| `artifacts/experiments/<tag>/{experiment,tuning,reliability}.json` | small | Per-experiment config and calibration |
| `basemaps/*.geojson` | 12 MB | Public administrative boundaries |

## Not tracked, and why

| File | Reason |
|------|--------|
| `artifacts/catastro_emb.pt` (161 MB) | Above GitHub's 100 MB per-file limit. Regenerable, see below. |
| `artifacts/experiments/*/model.pt`, `catastro_emb.pt`, `*.npy`, `*.npz` | Large, regenerable |
| `artifacts/train_data.npz`, `spatial_negatives.npy` | Large, regenerable |
| `artifacts/splits/`, `strict_eval_*`, `eval_*.parquet` and row-level CSVs | Contain raw addresses from private sources (privacy) |
| `artifacts/catastro.parquet`, `catastro_parts/` | Raw download with geometry (87 MB), regenerable |
| `context/`, `outputs/` | Real citizen data (Fasecolda, RUD with names and ID numbers, inspections) |
| `deploy/artifacts/` | A copy of the seven serving files |
| `direcciones_ANN.ipynb`, `.atl/` | Large or local tooling |

## Regenerating `catastro_emb.pt` (one command)

`catastro_emb.pt` depends only on `model.pt` and `catastro_docs.parquet`, both tracked in git, so a fresh clone
only needs `cali-address setup` (seconds on a GPU, about 3-10 minutes on a CPU). It embeds `direccion.upper()`
of every document with `train.embed_documents` (the same function training and
`scripts/finish_training_artifacts.py` use) and the result is identical to the file produced by training.
Deployment: `deploy/prepare_artifacts.py` and the `Dockerfile` still expect a ready `catastro_emb.pt` in
`deploy/artifacts/`; running `setup` before `prepare_artifacts.py` (or as a Docker build step that has
`model.pt` and `catastro_docs.parquet`) would replace copying it by hand. See README_DEPLOY.md.

## Regenerating the untracked files

Run from the repository root. Steps 1-2 only need network access; the data is public.

```bash
# 1. Download the cadastral layer (resumable) -> artifacts/catastro.parquet
python scripts/download_catastro.py

# 2. Build docs, spatial negatives and training tensors
#    -> catastro_docs.parquet, catastro_own_centroids.parquet, spatial_negatives.npy, train_data.npz
python scripts/prepare_training.py

# 2b. Only if you already have catastro_docs.parquet / catastro_emb.pt and want the parcel's own point
#     without rebuilding anything: writes the NEW file catastro_own_centroids.parquet next to the docs
#     (the docs and the embeddings are untouched)
python scripts/build_own_centroids.py

# 3. Train (writes model.pt, train_report.json, catastro_emb.pt; needs a GPU for reasonable time)
PYTHONPATH=src python scripts/train.py

# 3b. If model.pt exists but catastro_emb.pt is missing, embed without retraining
PYTHONPATH=src python scripts/finish_training_artifacts.py

# 4. Fusion weights and threshold -> tuning.json
PYTHONPATH=src python scripts/tune_threshold.py

# 5. Calibrated confidence -> reliability.json (needs dev rows, see below)
PYTHONPATH=src python scripts/fit_reliability.py --rows <dev rows parquet>

# 6. Assemble the seven serving files into deploy/artifacts/
python deploy/prepare_artifacts.py
```

`gazetteer.pkl` is rebuilt automatically from `basemaps/` the first time `Gazetteer.load` runs.
The seven serving files are `model.pt`, `catastro_emb.pt`, `catastro_docs.parquet`, `tuning.json`,
`gazetteer.pkl`, `reliability.json` and `catastro_own_centroids.parquet` (see `README_DEPLOY.md`). The last one is
optional at runtime: without it the service reads `catastro.parquet` when present, else it keeps the address
centroid and says so in `motivo`; the image build (`Dockerfile`) and `deploy/prepare_artifacts.py` do require it.

Experiments write to `artifacts/experiments/<tag>/` and never touch production; see [experiments.md](experiments.md).

### Backend ONNX opcional (sin torch en producción)

Los archivos del backend `onnx` / `onnx-int8` se generan a partir de `model.pt` y `catastro_emb.pt`; no reemplazan
a ninguno de los dos (el backend por defecto sigue siendo `torch`). Detalle, resultados y benchmark en [onnx.md](onnx.md).

Medición final en silencio (Ryzen 9 9950X, Windows, Python 3.14, 16 hilos, candidato `v2g_cert_pcer_catfb_s42_r3_basecal`
con capa de decisión calibrada; datos en `artifacts/experiments/onnx_bench_final/bench_final.json`):

| Backend | p50 (ms) | 2.000 filas/s | RSS pico (MB) | Paridad frente a torch (2.000 cadenas públicas) |
|---|---|---|---|---|
| torch | 23,5 | 120 | 1.827 | referencia |
| onnx fp32 | 21,5 | 182 | 1.962 | coseno mín. 0,99999988; top-1 99,95 % |
| onnx-int8 | 20,9 | 190 | 2.077 | coseno mín. 0,9995; no invariante al lote |

Con 4 hilos, ONNX fp32 rinde 3,9 veces lo de torch (155 frente a 40 filas/s). Recomendación: `onnx` fp32 por defecto;
`onnx-int8` es opcional (gana 3 a 4 % con 16 hilos y 8 a 14 % con 4). `NORMALIZER_ONNX_INDEX=fp16` baja el pico unos
220 MB (11 %) a cambio de multiplicar por 8,5 la latencia de una dirección. Imagen sin torch estimada en aprox. 1,45 GB
frente a 2,33 GB (por confirmar con `docker build`). Los números de un contenedor Linux serán distintos; solo la
comparación relativa es válida.

```bash
# Dependencias opcionales: pip install -e ".[onnx]"
# 7. Exporta model.onnx, model.int8.onnx, onnx_meta.json y catastro_emb.npy (fp16) junto a model.pt
PYTHONPATH=src python scripts/export_onnx.py --artifacts-dir <directorio con model.pt>

# 7b. Paridad contra torch (coseno, top-1) y benchmark en CPU; escriben en artifacts/experiments/onnx_bench/
PYTHONPATH=src python scripts/onnx_parity.py --artifacts-dir <directorio de servicio>
PYTHONPATH=src python scripts/bench_backends.py --set prod=<directorio de servicio>   # la medición final usa artifacts/experiments/onnx_bench_final/

# 7c. Servir con el backend elegido (torch | onnx | onnx-int8)
NORMALIZER_BACKEND=onnx-int8 uvicorn cali_address.api.main:app
```

## Index v2: built from the official nomenclature

The production index (`catastro_docs.parquet`, 330,387 documents) comes from the cadastral layer. Index v2 is built
from the official domiciliary nomenclature instead, and **only** from it: the cadastre is read-only audit input (the
remap of `train_pairs`, the lot-level oracle gate, coverage figures) and never a source of documents, NPN,
`n_predios`, block or centroids. Cadastral addresses with no nomenclature record are therefore not in a v2 index by
construction; the cost is measured, not assumed (`index_meta.json["oracle_ceiling"]`, report section 4).

### Sources

| Source | Rows | Role |
|--------|------|------|
| IDESC WFS `dapm:pdt_nmc_nomenclatura_domiciliaria` | 692,389 (437,945 `cert`, 253,627 `pcer`, 808 `lote`, 9 `prov`), 40 properties | Primary, reproducible |
| Shapefile `train_data/nomenclatura_certificada_20261030/*.shp` | 437,825 `cert` points, EPSG:6249 | Validation snapshot only (not reproducible) |

`lote` and `prov` are quarantined and never indexed. Coordinates of every indexed row come from reprojecting the
projected `ndcoordx` / `ndcoordy` (EPSG:6249) to EPSG:4326 locally with the shapefile `.prj`; the WFS lon/lat are kept
as `lon_wfs` / `lat_wfs` for audit (rows whose projected coordinates are 0/0, 7,541, fall back to the WFS point and are
counted). `x_m` / `y_m` still come from `geo.to_metric` (EPSG:3116).

### Build, in order

Run from the repository root (`train_data/` and the full `artifacts/` are untracked; see the table above).

```bash
# 1. Reproject the shapefile snapshot -> train_data/.../*_4326.parquet + reprojection_report.json
python scripts/reproject_nomenclatura.py
# 2. Download the WFS layer (resumable) -> artifacts/nomenclatura_wfs.parquet + nomenclatura_meta.json
#    (optionally validate against the snapshot -> nomenclatura_wfs_vs_shp.json)
python scripts/download_nomenclatura.py
# 3. Clean it (per-rule counts, quarantine < 0.5 %) -> nomenclatura_clean.parquet, nomenclatura_quarantine.parquet,
#    nomenclatura_cleaning_report.json
python scripts/clean_nomenclatura.py
# 4. Build an index into its OWN directory (never artifacts/ itself)
python scripts/build_index.py --tiers cert      --out-dir artifacts/index_v2c_cert_r2
python scripts/build_index.py --tiers cert,pcer --out-dir artifacts/index_v2d_cert_pcer_r2
# 5. Compare against production (aggregates only) -> <index dir>/compare_indexes.json
python scripts/compare_indexes.py --old artifacts --new artifacts/index_v2d_cert_pcer_r2
# 6. Train on it (GPU venv), then score / compare / report; see experiments.md
.venv-train/Scripts/python scripts/run_experiment.py --tag v2d_cert_pcer_s42 --index-dir artifacts/index_v2d_cert_pcer_r2
python scripts/compare_runs.py --old artifacts/experiments/baseline_prod_2026_10 --new artifacts/experiments/v2d_cert_pcer_s42
python scripts/write_retrain_report.py --out docs/retrain_2026-10_nomenclatura.md --final-tag v2d_cert_pcer_s42 ...
```

`--tiers cert` builds index v2c, `cert,pcer` index v2d (priority `cert` > `pcer` on a shared address). `build_index.py`
exits 2 on a usage error or a failed gate (parser failure rate >= 1 %, accounting mismatch, dropped share > 0.5 %,
lot-level oracle ceiling more than 2 points below production) and then writes no marker.

### Directory layout

```text
artifacts/index_v2d_cert_pcer_r2/
  catastro_docs.parquet           documents (production columns first, new columns after)
  catastro_own_centroids.parquet  point of the row each document reports, same row order as the docs
  nomenclatura_units.parquet      units sidecar: rows with an interior (AP 302, TO 2...) keyed to their base document
  spatial_negatives.npy           hard negatives for training
  train_data.npz                  augmented queries + official strings (q_source 0 synthetic / 1 official) + doc tokens
  train_pairs.parquet             real pairs remapped to the new doc ids
  index_dropped.parquet           quarantined records with the reason
  index_meta.json                 written LAST: the completion marker
```

`index_meta.json` (schema 2) holds `n_docs`, `tiers`, `counts_by_fuente` / `counts_by_estado`, `accounting`
(`input`, per-bucket counts `base_doc_new`, `base_doc_merged`, `unit`, `fallback_doc`, `dropped`, `excluded_tier`,
`incomplete_address`, `spread_outlier`; the buckets must sum to the input), `npn_choice`, `disagreements` (WFS
comuna/barrio/manzana vs the NPN-derived values), `queries` (synthetic vs official counts and share),
`train_pairs` remap counts, `oracle_ceiling` (with its `gate`), `crs`, `objectid_offset`,
`pre_normalize_separates_letters`, `files` (sha256 and size of every file), `sources` (sha256 of the inputs),
`code_fingerprint` and `seed`. A directory without it is not an index: `paths.shared_path` raises
`IndexFileMissingError` for any index file instead of silently falling back to production, and
`run_experiment.py --index-dir` refuses it.

### Document columns

Production columns first (`direccion`, `numero_predial_nacional`, `manzana`, `comuna`, `barrio`, `barrio_code`,
`centroid_lon`, `centroid_lat`, `n_predios`, `manzana_nunique`, `objectid`, `x_m`, `y_m`, `doc_id`), then:

| Column | Meaning |
|--------|---------|
| `fuente` | `nomenclatura_cert` or `nomenclatura_pcer`; appended to the strict output as the last column when present |
| `estado_nomenclatura` | `cert` or `pcer` |
| `npn_catastro`, `direccion_catastro` | reserved for cadastre cross-references; null in new-sources-only builds |
| `npn_choice` | how the document's NPN was chosen among several (`single`, `base_row`, `lowest`) |
| `n_units` | distinct NPN behind the document (units); `n_predios` stays the number of distinct lots (`NPN[:21]`), as in production |
| `source_objectid` | the WFS `gid` the document's point came from |
| `parse_ok`, `zona` | parser success flag; `URB` / `EXP` (expansion zone) |
| `address_key`, `doc_key` | join keys (`labels.address_key`; whitespace-free canonical text) |
| `spread_m` | distance (m) from the document's points to their median; last column |

`objectid` of a nomenclature document is `gid + 10_000_000`, so it never collides with a cadastral id. The units sidecar
(`address_key_base`, `interior`, `numero_predial_nacional`, `own_lon`, `own_lat`, `objectid`, `direccion_original`,
`estado`, `doc_ord`) keeps interiors out of the documents: exploding them would add about 190k near-duplicate documents
for an encoder that only reads 40 characters.

### Per-artifact parser flag

`pre_normalize_separates_letters` (`KR 32B` read as `KR 32 B` by `parser._pre_normalize`) is **off** by default so
production is byte-identical. An index declares `"pre_normalize_separates_letters": true` in `index_meta.json` and
`AddressNormalizer`, `train.py` and `build_index.py` read it from there; the production artifacts carry no marker and
keep the old view. Never evaluate production weights with the flag on: they were trained on the old view.

### Why `catastro.parquet` is not part of v2 indexes

`catastro.parquet` (338k polygons with geometry, 87 MB) is a raw cadastral download. A v2 index is nomenclature-only, so
it neither needs nor ships it: the units sidecar and `catastro_own_centroids.parquet` already hold the points. It is
still listed in `paths.INDEX_FILES` (optional): in an index directory the loader resolves it to a local, non-existent
path so optional consumers get `None` rather than the production polygons.

### Promotion (explicit approval only)

Nothing above touches `artifacts/`. Promoting a v2 index changes the serving files `model.pt`, `catastro_emb.pt`,
`catastro_docs.parquet`, `catastro_own_centroids.parquet`, `tuning.json` and `reliability.json`, and adds
`train_data.npz`, `spatial_negatives.npy`, the remapped `train_pairs`, `nomenclatura_units.parquet` and
`index_meta.json`. `gazetteer.pkl` is unchanged. Delete the caches `catalog_keys_v1.{pkl,json}` and
`interp_index_v1.{pkl,json}` (they are keyed to the documents) and rerun `python deploy/prepare_artifacts.py`; the
image grows because `catastro_emb.pt` is `n_docs` x 512 bytes. Back up `artifacts/` first with a sha256 manifest;
rollback is copying it back and rerunning `prepare_artifacts.py`.

### Respaldo de catastro (opcional y reversible)

El índice puede llevar, **después de todos los documentos de nomenclatura**, los documentos del catastro cuyos lotes no
tiene ningún registro de nomenclatura (`fuente = catastro`). Se construye con `scripts/index_fallback.py` (`add`,
`experiment`, `remove`) en directorios nuevos: el índice base, sus archivos y su `catastro_emb.pt` no se modifican, el modelo
no se reentrena y las primeras `N` filas de los embeddings son las del índice base. Se revierte de tres formas: (1) servir
el índice base, que sigue intacto; (2) `index_fallback.py remove`, que corta las filas agregadas de documentos, centroides,
pares y embeddings y reproduce los archivos del índice base byte a byte (verificado con sha256); (3) sin reconstruir, con
`"exclude_fuentes": ["catastro"]` en `decision.json` o `NORMALIZER_EXCLUDE_FUENTES=catastro`, que deja el servicio idéntico
al de un índice sin esos documentos. La regla, la construcción paso a paso, los riesgos y el efecto en las métricas están en
[respaldo-catastro.md](respaldo-catastro.md).

## Splits

`scripts/build_splits.py` builds `artifacts/splits/` (`frozen_test`, `dev`, `train_pairs`, `unlabeled_addresses`) and
needs the private `context/` workbooks, so it cannot run from a fresh clone. Rules and rationale are in
[splits.md](splits.md). Step 5 and the strict evaluation (`scripts/eval_strict.py`) also depend on those splits.

## Experiment status

- Stage 2b/3: retraining variants (real pairs, plate negatives, realistic noise, fine-tuning, seed repeats) did not
  beat the seed-noise floor, so the production weights are unchanged.
- Decision-layer candidate `s3_final` (weights 0.7/0/0.2/0.1, threshold 0.70, soft rules `letra_via`,
  `letra_cruce_cardinal`, `bis_via`, `bis_cruce`, `cuadrante_compl`, `max_soft` 1, `gate_fallback`) on `frozen_test`:
  - all-rows OK: 1036 -> 1090
  - gold OK: 398 -> 405
  - gold predial precision: 0.927 -> 0.9259
  - all-rows manzana precision: 0.694 -> 0.6807 (-0.0133, mostly noisy-label rows)
- **Not promoted.** Production serving files are unchanged. Details and selection rules: [experiments.md](experiments.md).
