# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - Unreleased

First packaged release. The version moves from the unreleased 0.1.0 of the initial import to 0.2.0 because it adds the
reliability work below (position tiers, fidelity risk, catalog status, override guard, parser rounds 5 to 8). The package,
`cali_address.__version__` and the API (`/openapi.json`) now report the same version.

### Added

- Parser / fidelity fixes for the families behind true high fidelity-risk corruptions (round 7). Parser: letter-digit
  designators without `#` read like the explicit-hash form when unambiguous (`CRA 987 5B1 321 APT 201`,
  `KR 987G6 654S1 321`, `Cra 987C2 654-321`; a bare two-number run or a chain `B5B2 321` keeps the historical reading);
  `NO654` / `NO654-321` is the number sign, not the quadrant N + OESTE (a `NO #` quadrant and `CASA NO18` are
  untouched); a street letter glued to a spelled quadrant survives (`3BNORTE`, `3AOESTE`; a doubled letter such as
  `OOESTE` stays a typo); `entre` / `con` followed by two or more numbers is read as via + first cross with no plate
  and `truncated_parse` (tier via at most) instead of inventing a plate from a street number; `ET 2` / `ETAPA 2` is a
  unit, never the plate; ordinal indicators (`2ª`, `6º`) are not letters, `NÂ°` / `NÂº` is the number sign,
  `987AN°654` is letter + number sign; coordinates (4+ decimals) and 7+ digit runs glued to an address are dropped;
  a cardinal letter after the street letter (`KR 987A N 654 B 321`), a short quadrant (`OE`) and `BNORTE` stay in the
  via of a `#`-less address. `parser.rule_canonical` (used by `service`) keeps a spelled NORTE / SUR / OESTE that
  sits right after the plate (`# 654 - 321 OESTE SAN FERNANDO`; `ESTE` is not kept: no cadastral occurrence).
  Fidelity: decimal numbers with 3+ decimals, runs of 7+ digits and a `//` coordinate tail are not door digits;
  `ª` / `º` / `NÂ°` are read like the parser. Emission: a non-`OK` row with `fidelity_risk == 'high'` never emits rule
  text (`catalog.keeps_rule_text`), whatever its `catalog_status`; `OK` rows are untouched.
  Rows previously flagged high that were only a false alarm are now none, fidelity-none rows are unchanged, and no
  non-OK high row emits text.
- Fidelity hardening (round 6). A stray `;` glued to a number sign / dash (`#;58-65`) no longer ends the door (the
  dropped plate is `door_digits_dropped`, high); a number group that trails the door and has no trace in the output
  (`- 106`, `/10`, `al 4-10`, `-4D 42`, a second `# 4-10`) raises the new `trailing_number_dropped` flag (medium) or
  `multi_address_truncated` for ranges; groups behind a unit marker (AP, TO, INT, LC, CS, BL, CASA, `P 1`) and
  numbers that start place text (`3 de Julio`) are exempt, and a door letter lost before a trailing number
  (`40A 41`) is `letters_dropped` (high).
  Fidelity-none rows are unchanged.
- Emission rule of `direccion_normalizada` by `catalog_status` (`catalog.keeps_rule_text`, applied in
  `service._apply_emission_rule`): for non-`OK` rows, `unparsed` no longer emits text, and `below_cutoff` /
  `rejected_mismatch` emit only when the fidelity risk is none/low and the parse is complete (via type + number,
  cross number, plate, not truncated). A blanked row has `direccion_normalizada` = null and
  `fuente_normalizacion` = null (documented: "empty when `direccion_normalizada` is null"); `estado`, `motivo`,
  `catalog_status`, `position_tier` and the schema are unchanged. The pipeline treats an empty normalized address as
  not normalized and falls back to the raw text (`fuente_direccion == 'origen'`). `OK` rows are never blanked.

- Position tiers in `normalize_strict` (output columns appended last: `position_tier`, `position_radius_m`,
  `position_source`, `selected_doc_id`, `override_rejected`). Tiers and empirical radii: `predio` 65 m, `lote` 75 m
  (a site unit both sides state differs: predial withheld), `tramo` 110 m (plate +-1/+-2 or plate letters/BIS:
  coordinates interpolated between the two nearest parcels of the same side of the street pair, predial withheld),
  `manzana` 135 m (one-sided street letters or quadrants, shared addresses, soft findings), `via` 260 m (plate
  beyond the last known parcel: coordinates withheld unless `service.EXTRAPOLATED_COORDS_ENABLED` is True) and `none`.
  A quadrant that moved to another position (via / cross / trailing) is now rejected (`base distinta al catastro:
  quadrant_moved`). `nivel_precision` keeps its vocabulary and derives from the tier (`predio`->predio, `lote` /
  `tramo` / `manzana`->manzana, `via`->via); a one-sided via letter is now capped at `manzana` (was `esquina`).
  `inference.base_mismatches(..., mode="position")` is the shared guard; the interpolation index
  (`cali_address.interp`) is built lazily and cached to `artifacts/interp_index_v1.pkl`.
  `scripts/eval_strict.py` reports a per-tier calibration table (`strict_eval_<tag>_tiers.csv`).
- Three-step local use: `cali-address setup` checks the artifacts, regenerates `catastro_emb.pt` from
  `model.pt` + `catastro_docs.parquet` (atomic write, progress, `--force`, `--device`, `--batch-size`) and
  self-checks with 3 invented addresses; `cali-address web` starts the existing FastAPI upload page
  (`--host`, `--port`, `--no-browser`). `train.embed_documents` gained an optional `progress` callback.
- Any-dataset input and output: `cali-address normalize | inspect | formats` reads CSV, TSV, Excel,
  Parquet, JSON/JSONL, GeoJSON, shapefile, GeoPackage, URLs and SQL databases, maps the address column
  (auto-detected, single column, or several parts), streams in chunks, and writes CSV, Excel, Parquet,
  JSON, JSONL or GeoJSON atomically. TOML `--config` files, `--dry-run`, run summaries and exit codes
  (0 ok, 2 usage, 3 I/O). See `docs/any-dataset.md`.
- Friendly error when the model artifacts are missing: the CLI lists the exact files and points to
  `docs/model-artifacts.md`. `CALI_ARTIFACTS_DIR` sets the default artifacts directory.
- Decision layer with soft-rule parameters (`--soft-rules`, `--max-soft`, `--gate-fallback`) that can also
  be carried by `tuning.json`.
- Experiment infrastructure: isolated experiment directories, leaderboard, seed repeats, real-pair
  training, plate negatives and realistic-noise augmentation (`docs/experiments.md`).
- Gold-quality labels and leakage-free data splits (`docs/splits.md`).
- Packaging: `pyproject.toml` with optional extras (`api`, `sql`, `excel`, `geo`, `train`, `dev`, `all`),
  the `cali-address` console script, `__version__`, and a GitHub Actions workflow.

### Changed (round 5: GPS-free reliability, 2026-10-18)

- Truncated parses can never be verified. A comma that separates a bare via from ``<cross> - <plate>`` (``CL 987, # 654 - 12``,
  ``KR 987, 8B-20``, ``AV 987 B 6, 654 NORTE - 12``, ``CL 987 OESTE, # AV 654 OESTE - 12``) is now parsed as one address
  (explicit ``#``); when a dropped tail with digits leaves a head without cross or plate the parse carries the note
  ``truncated_parse`` (``parser.is_truncated``; ``dropped_trailing_segment`` alone is kept for a complete address followed
  by a neighbourhood). Research guard: ``base_mismatches`` returns ``["truncated_parse"]`` for every candidate rank
  (``override_rejected``, rule text kept). Production: tier ``via`` (predial cleared, coordinates withheld), motivo
  ``posicion maxima via: parse truncado``.
- Position radii are geometric (cadastre, no GPS): ``predio`` 15 m, ``lote`` 44 m, ``manzana`` 56 m, ``via`` 21 m;
  ``tramo`` reports a step function of the gap between the flanking parcels (``interp.TRAMO_RADIUS_STEPS``: 15 / 23 m inside
  the cap; floor = predio) and an extrapolated ``via`` row a step function of its distance from the last parcel
  (``interp.EXTRAP_RADIUS_STEPS``). ``position_radius_m`` therefore varies inside ``tramo`` / ``via``. Caps derived with the
  rule "largest bin edge whose bin p90 stays <= 30 m": ``INTERP_MAX_GAP_M`` 150 -> 50, ``INTERP_MAX_PLATE_GAP`` 40 -> 60,
  new ``INTERP_MAX_EXTRAP_M`` 25 (extrapolation used to share the 150 m cap). Tables and derivation: ``interp.py``.

### Added (round 5)

- Output columns ``catalog_status`` (``verified`` / ``ambiguous`` / ``door_not_catalogued`` / ``rejected_mismatch`` /
  ``unparsed`` / ``below_cutoff``) and ``address_verified`` (True exactly for ``verified``), appended after the fidelity
  columns; ``ResultRow`` carries them. The exact-base key set behind ``door_not_catalogued`` (``cali_address.catalog``) is
  cached next to the interpolation index (``catalog_keys_v1.pkl`` + sidecar ``catalog_keys_v1.json``, same hash
  discipline) and warmed by ``deploy/prepare_artifacts.py`` and the API start-up. ``confiabilidad`` is unchanged.

### Added (reliability re-verification 2026-10-17)

- `scripts/fit_reliability.py --kind logistic --gold-only`: fits the logistic calibrator on text-consistent rows only
  (`gt_quality == "gold"`) and writes a `reliability-v5-gold-YYYYMMDD` artifact with `label_quality`,
  `code_fingerprint`, `cv_protocol` and every source file in `source_rows`. `--rows` takes several files.
  Same serving format as v2 (`kind: logistic`): the current loader serves it unchanged.
- `scripts/eval_strict.py` copies `gt_quality` into the rows output, fills `has_gt` / `gt_lat` / `gt_lon` for files
  that lack them, and takes `--label-quality gold` for `train_pairs.parquet` (gold by construction).
- New artifact `artifacts/reliability_v2.json` (v5-gold, fitted on gold OK rows of dev + train_pairs). `artifacts/reliability.json` (hybrid v4) is unchanged and still the one served.
  Why: the stored v4 calibrator was fitted on rows whose predio label is mostly GPS noise, so it barely separates
  correct from incorrect rows. The v5 calibrator, fitted on text-consistent rows and checked once on the frozen gold
  rows (never used for fitting or choosing), orders them far better. Promotion changes the scale of `confiabilidad`,
  so the `>= 0.58` rule does not carry over (dev-chosen equivalent: `>= 0.81`). Manzana: no measurable discrimination
  either way (the gold sets have almost no wrong manzanas). Run results are not published. Details:
  `../docs/CHANGELOG_confiabilidad.md` (repository root `docs/`).

### Fixed (audit 2026-10)

Findings of the October 2026 audit of the normalizer (the audit report itself is not part of this repository); each fix has its regression
test in `tests/test_auditoria_2026_10.py` (round 1) and `tests/test_auditoria_2026_10_r2.py` (round 2, after the
independent review). Run results (before/after tables on the SIG reports layer and field GPS) are not published.

- A numeric via / cross suffix that differs or is present on one side only (`KR 26 H` vs `KR 26 H 3`) is a hard
  violation (`numero de sufijo de via - != 3`); it used to come back `OK` at level `predio` with no note. The rule stays
  hard on purpose, but it has a cost: some rows that were `OK` before and are now rejected for a suffix present on
  one side only were the right lot. Those
  rejections now end with the stable token `regla: sufijo unilateral` in `motivo` so downstream can audit or recover them.
  A candidate that fails ONLY that one-sided check still counts as an ambiguity competitor (`margen`, `ambiguo: ...`); a
  candidate with different suffixes on both sides, or with any other hard violation, does not. Round 1 had dropped such
  competitors, which turned genuinely ambiguous rows into `OK`; the remaining boundary cases have a margin very close
  to the 0.02 threshold and moved with the exact fp32 scoring.
- A quadrant that conflicts across fields (`CL 5 NORTE # 38-10` vs `CL 5 # 38 OESTE - 10`: each side names one, none in
  common) is a hard rejection (`cuadrante NORTE != OESTE`).
  It was not seen in the evaluated rows, so it costs nothing there.
- A street letter on one side only (`KR 26` vs `KR 26 H`) stays an accepted soft match but `nivel_precision` is capped
  at `esquina` (was `manzana`) and `motivo` adds `nivel maximo esquina: via con letra en un solo lado`.
  `confiabilidad` is unchanged (the calibrated model keeps its previous level input).
- A quadrant on one side only is accepted but capped at `manzana` (was `predio`), with `aproximado: cuadrante de via
  NORTE != -` in `motivo`. A quadrant that moved between via and cross, sits in the complement, or is the cardinal
  street letter (`CALLE 6N`) is not one-sided. It does not count against `max_soft`.
- Plate off by 1-2 is pinned by tests: always `manzana` at most, always `aproximado: placa X != Y`.
- Tower / block / stage / interior after a comma or semicolon (`Calle 5 # 38-10, Torre B`) is parsed as a complement
  (city, barrio and free-text tails are still ignored); `TORRE O` and a bare `T B` are recognised. Only
  `complement_text` changes: the scoring complement, `confianza`, `estado` and the NPN are untouched.
- Shared addresses: the reported `lat` / `lon` is now the centroid of the parcel whose NPN is reported; it used to be the
  median of all parcels at the address, which could be more than 50 m away from the returned parcel. The own centroid comes from, in order: the new sidecar `catastro_own_centroids.parquet` (one row
  per document in document order: `objectid`, `numero_predial_nacional`, `own_lon`, `own_lat`; 8 MB; built by
  `scripts/build_own_centroids.py` or by `scripts/prepare_training.py`; accepted only when its rows match the documents),
  else `catastro.parquet` (`inference.load_own_centroids`, validated against the NPN). Round 1 only read
  `catastro.parquet`, which the Docker image does not ship, so the fix was inactive when deployed: the sidecar is now in
  `deploy/prepare_artifacts.py` `SERVING_ARTIFACTS`, `paths.SERVING_FILES` / `SHARED_INPUTS` and the Dockerfile `test -f`
  (the build fails without it), and it is optional at runtime (not in `REQUIRED_ARTIFACTS`). With neither file the
  address centroid is kept and `motivo` adds `coordenada: centroide de la direccion compartida por N predios`. The
  sidecar is a NEW file: `catastro_docs.parquet` and `catastro_emb.pt` are untouched and nothing is re-embedded.
- A null NPN is returned as `null` (it was the text `None`, also in `manzana`), flagged `NPN nulo en catastro` at level
  `direccion`; digit-led NPNs that are not 30 digits (28 / 29 digits, trailing `*`) are flagged
  `NPN anomalo en catastro (...)` and kept.
- Results no longer depend on the batch size: the query encoder and the coordinate head run in fp32 (no bf16
  autocast), candidates are picked with the fp16 matmul but their similarity is recomputed in fp32
  (`AddressNormalizer.encode_queries_exact`, `search_exact`). On GPU, one batch vs batches of 64 used to flip some
  statuses and shift `confianza`; both now agree. `score_components([])` returns empty
  arrays instead of raising. Caveat: `normalize_batch` (used by `scripts/ablation.py`, `scripts/smoke_eval.py` and
  `scripts/build_notebook.py`) still runs the bf16 `encode_queries` / `search`, so the batch-size independence holds for
  the `score_components` / `normalize_strict` path (CLI, API, pipeline), not for those scripts. Same input twice: no
  differences.
- `esquina` now covers two situations: no plate, **or** a street letter on one side only. Consumers must read `motivo`
  (`nivel maximo esquina: via con letra en un solo lado`) to tell them apart.
- Parser: `TORRE O 2` keeps the 2 (`TO O2`, like `TORRE B 2`); a glued `T2` is a tower (`T2 APTO 301` -> `TO 2`, `AP 301`);
  the bare `T` alias is not applied after a place word or a connector (`ESQUINA T 2`), a marker word (`NO T 2`) or a
  number, and takes at most 3 digits. A second address typed after a unit without a comma or semicolon stays in
  `complement_text` (documented limit; recognised kinds are the only units read from it).

### Recommended rule for a trustworthy NPN

`estado == "OK"` and `nivel_precision == "predio"` and `confiabilidad >= 0.58` (CLI configuration). Run results are not
published; validate the cut-off with your own sample.

Do not raise `confiabilidad` above ~0.62: the relation is not monotonic. GPS lands on the facade, so a same-lot figure
measured against GPS is a floor.

### Changed

- `normalize_strict` defaults are now the CLI / API / pipeline ones: `barrio_buffer_m=CLI_BARRIO_BUFFER_M` (1000 m) and
  `gate_escalate=DEFAULT_GATE_ESCALATE` (`True`), one constant for every entry point. Library callers that relied on the
  old 200 m / no escalation must pass them explicitly.
- `normalize INPUT` without `-o` now writes `<input stem>_normalizado<ext>` next to a local input (same format;
  xls -> xlsx, txt -> csv, shp/zip/gpkg -> geojson) instead of failing with exit 2. URL and database inputs still
  require `-o`.
- SQL input is table/view based by design (`--table [schema.]name`, session read-only where the driver supports it): there is no free-form SQL option because a text filter cannot be a security boundary for SQL, so create a database view for joins or filters.
- `--help`, `inspect` and `formats` no longer import torch (table helpers moved to `cali_address.tables`;
  every name is still importable from `cali_address.service`).
- Tests that need the real serving artifacts carry the `requires_artifacts` marker and are skipped when
  the artifacts are absent.

### Notes

- Model weights and embeddings are not shipped in the wheel.
