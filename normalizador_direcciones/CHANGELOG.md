# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First packaged release.

### Added (grammar rescue, 2026-11)

- `cali_address.grammar_rescue`: a second, conservative chance for addresses the rule grammar rejects (`NO_PARSEABLE`).
  It runs only for rows `parse_address` already failed on, so an address that parses today (and everything downstream of
  it: query, model score, decision) is byte-identical. It tries, in order: the verified street-name table, variants and
  abbreviations (`CAR`, `AVD`, leading `RA`, ordinal words such as `primera` / `sexta`, a comma after the street type
  `AV, <n>`, `Av.` with a dot, `OE`), and free text before the first street-type word (building / hotel names) when the
  remainder parses. A tower / block in that prefix (`TORRE D CALLE ...`) is never dropped: it moves after the
  address as a complement. A prefix that contains digits is never stripped.
- `cali_address.via_names` (generated) + `src/cali_address/data/via_names_review.csv` + `scripts/via_names_evidence.py` /
  `scripts/build_via_names.py`: a reviewable street-NAME table. A name ships only when the cadastre evidence resolves it to
  one street without ambiguity (status `verified`); `ambiguous` and `rejected` names stay in the CSV with the reason and are
  not used. Today no name passes (see `docs/grammar-rescue.md`), so the shipped table is empty.
- `normalize_strict(stats=...)` gains the counter `rescatadas_gramatica`.
- Tests: `tests/test_grammar_rescue.py`.

Expected effect and rollout: no retraining, ONNX weights and artifacts untouched. The engine fingerprint (sha256 of every
`cali_address/**/*.py`) changes, so the ETL will create a NEW normalization profile and re-normalize every address when this
engine is switched in; that switch is a manual step (unit N2), not part of this change.

### Added

- Retrain tooling for an index built from the official nomenclature (new sources only; production artifacts and
  default behaviour unchanged):
  - `scripts/reproject_nomenclatura.py` (shapefile EPSG:6249 to 4326 with the `.prj`; bbox and cadastre-sanity gates)
    and `scripts/download_nomenclatura.py` (resumable, paginated IDESC WFS download with `nomenclatura_meta.json` and a
    WFS-vs-shapefile validation; the NPN gate is at lot level, coordinates median <= 1 m / p99 <= 2 m).
  - `cali_address.nomenclatura_clean` + `scripts/clean_nomenclatura.py`: per-rule cleaning report, quarantine with a
    reason (gate: < 0.5 %), `lote` / `prov` states excluded, coordinates from the local reprojection of
    `ndcoordx` / `ndcoordy` (WFS point only as fallback).
  - `parser.normalize_spacing` (`KR 32B` = `KR 32 B`), wired into `_pre_normalize` behind
    `PRE_NORMALIZE_SEPARATES_LETTERS`, which is **off** by default and enabled per artifact by
    `"pre_normalize_separates_letters": true` in `index_meta.json`.
  - `cali_address.index_union` + `scripts/build_index.py`: index v2c (`--tiers cert`) / v2d (`cert,pcer`) written to its
    own directory, `index_meta.json` last as the completion marker (sha256 per file), record accounting (every record
    in exactly one bucket, `dropped` = 0), units sidecar `nomenclatura_units.parquet`, official strings as training
    queries (`dataset.build_training_arrays(extra_queries=...)`, `q_source`), lot-level oracle-ceiling gate.
    New document columns: `fuente`, `estado_nomenclatura`, `npn_catastro`, `direccion_catastro`, `npn_choice`,
    `n_units`, `source_objectid`, `parse_ok`, `zona`, `address_key`, `doc_key`, `spread_m`; `normalize_strict` appends
    `fuente` / `estado_nomenclatura` to its output only when the index has them.
  - Index guard in `paths.py` (`INDEX_MARKER`, `INDEX_FILES`, `IndexFileMissingError`): a directory with
    `index_meta.json` never falls back to production for index files. `scripts/run_experiment.py --index-dir` validates
    the marker and sha256, links the index files into the experiment dir, defaults `max_minutes` to 120, fails loudly
    (exit 4) when the time budget cuts training, and adds `stopped_early` / `index_id` to the leaderboard.
  - Lot-level scoring (`NPN[:21]`): `correct_predial_lot`, `lot_present_in_index`, `predial_lot_precision` columns and
    leaderboard fields; `eval_strict.py` writes `strict_eval_<tag>_by_fuente.csv` and `--validate-outputs`.
  - `cali_address.retrain_compare` + `scripts/compare_indexes.py` / `scripts/compare_runs.py`: index profile, gold lot
    coverage, coordinate shift, row agreement between runs, confidence curve and the A1-A12 acceptance checks
    (`docs/retrain_2026-10_acceptance.md`).
  - `cali_address.retrain_report` + `scripts/write_retrain_report.py`: the aggregates-only Markdown report
    (`docs/retrain_2026-10_nomenclatura.md`) with a computed PASS / FAIL / INCONCLUSIVE verdict; missing inputs render as
    `n/a (pending)`; deterministic output.
  - Docs: `docs/model-artifacts.md` (v2 build from the new sources, layout, columns, promotion set) and
    `docs/experiments.md`.
- Output validation (D12a): `cali_address.output_schema` (pydantic v2 `NormalizedRecord`, `Correction`,
  `validate_and_correct(row_or_frame, mode)`), driven by `chosen.decision.validate_outputs` in `tuning.json` or the
  `validate_outputs=` argument of `service.normalize_strict` (`off` default = output byte-identical; `report` counts
  violations per rule into `stats["output_validation"]` without touching the output; `correct` applies only
  conservative corrections). `scripts/eval_strict.py --validate-outputs {off,report,correct}` writes
  `strict_eval_<tag>_validation.csv`. `pydantic` is now a base dependency.
- JEV audit (D12b): `cali_address.jev_check` + `scripts/audit_with_jev.py`, a thin layer over the sibling
  `models/validacion_jev` (lazy import, `VALIDACION_JEV_SRC` override). Offline auditor, never in the serving path.
  Privacy guard: only origins `nomenclatura_public` and `synthetic` can be sent; `dev`, `frozen_test`, `train_pairs`,
  `unlabeled` and unknown origins are refused unless `--allow-private` (no pipeline passes it). Outputs
  `jev_audit.json` (no address text) and `jev_audit_rows.parquet` stay under the untracked `artifacts/`.

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
  text (`catalog.keeps_rule_text`), whatever its `catalog_status`; `OK` rows are untouched. On the fresh 10,343-row
  sample: high 484 -> 333, 130 of them now none, fidelity-none rows unchanged (0 became medium/high), and no non-OK
  high row emits text (350 did).
- Fidelity hardening (round 6). A stray `;` glued to a number sign / dash (`#;58-65`) no longer ends the door (the
  dropped plate is `door_digits_dropped`, high); a number group that trails the door and has no trace in the output
  (`- 106`, `/10`, `al 4-10`, `-4D 42`, a second `# 4-10`) raises the new `trailing_number_dropped` flag (medium) or
  `multi_address_truncated` for ranges; groups behind a unit marker (AP, TO, INT, LC, CS, BL, CASA, `P 1`) and
  numbers that start place text (`3 de Julio`) are exempt, and a door letter lost before a trailing number
  (`40A 41`) is `letters_dropped` (high). On the fresh 10,343-row sample all 48 JEV-"no" rows are now medium/high
  (42 before) with 1 new medium flag on 5,980 JEV-"yes" rows; fidelity-none rows are unchanged.
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
- New artifact `artifacts/reliability_v2.json` (v5-gold, 5,256 gold OK rows of dev + train_pairs, current code
  `852ca89c4360`). `artifacts/reliability.json` (hybrid v4) is unchanged and still the one served.
  Why: the stored v4 numbers (predio AUC 0.496, n = 529) come from 2026-09-25 rows whose predio label is mostly GPS
  noise (noisy rows 23 % "correct" vs 99 % on gold rows). Measured once on the frozen gold rows (393 OK, never used
  for fitting or choosing): predio AUC 0.709 -> 0.973, Brier 0.158 -> 0.035, ECE 0.306 -> 0.009 (bootstrap CIs of
  the gains exclude zero). Promotion changes the scale of `confiabilidad` (mean ~0.63 -> ~0.94), so the `>= 0.58`
  rule does not carry over (dev-chosen equivalent: `>= 0.81` for 98 % predio precision). Manzana: no measurable
  discrimination either way (2 to 20 wrong manzanas in the gold sets). Details:
  `../docs/CHANGELOG_confiabilidad.md`.

### Fixed (audit 2026-10)

Findings of `../../docs/auditorias/AUDITORIA_NORMALIZADOR.md` (relative to this package); each fix has its regression
test in `tests/test_auditoria_2026_10.py` (round 1) and `tests/test_auditoria_2026_10_r2.py` (round 2, after the
independent review) and was measured before/after on two gold sets: the SIG reports layer and field GPS (stickers and
inspections with a device / firebase / manual GPS only, derived GPS excluded), 26,287 real rows, CLI configuration.

Measured effect of the whole audit (original code -> after round 2; clean references):

| Gold | OK rows | Same lot | More than 100 m away |
| --- | --- | --- | --- |
| SIG (6,013 rows) | 4,328 -> 4,280 | 89.3 % -> 90.1 % | 2.9 % -> 2.3 % |
| GPS (4,988 rows) | 2,951 -> 2,896 | 58.1 % -> 58.7 % | 5.6 % -> 5.1 % |

Round 1 alone gave 4,282 OK rows (SIG) / 2,897 (GPS) with the same percentages: round 2 turned 6 OK rows (2 SIG, 1 GPS,
1 derived GPS, 2 without reference) back into ambiguity abstentions and changed no other row (see below).

- A numeric via / cross suffix that differs or is present on one side only (`KR 26 H` vs `KR 26 H 3`) is a hard
  violation (`numero de sufijo de via - != 3`); it used to come back `OK` at level `predio` with no note. The rule stays
  hard on purpose, but it has a measured cost: of the 79 gold rows that were `OK` before and are now rejected for a
  suffix present on one side only, **24 were the right lot** (with different suffixes on both sides: 1 of 27). Those
  rejections now end with the stable token `regla: sufijo unilateral` in `motivo` so downstream can audit or recover them.
  A candidate that fails ONLY that one-sided check still counts as an ambiguity competitor (`margen`, `ambiguo: ...`); a
  candidate with different suffixes on both sides, or with any other hard violation, does not. Round 1 had dropped such
  competitors, which turned 6 genuinely ambiguous rows into `OK` (one a correct SIG lot, five wrong lots 63-224 m from the reference) (the other 21 of the 27 rows that went from
  `ambiguo` to `OK` are boundary cases: their margin is 0.0202-0.0218 against the 0.02 threshold and moved with the exact
  fp32 scoring). Same-lot accuracy unchanged (SIG 90.1 %, GPS 58.7 %).
- A quadrant that conflicts across fields (`CL 5 NORTE # 38-10` vs `CL 5 # 38 OESTE - 10`: each side names one, none in
  common) is a hard rejection (`cuadrante NORTE != OESTE`). Measured: 0 of the 26,287 rows were in that situation, so it
  costs nothing.
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
  median of all parcels at the address, more than 50 m away from the returned parcel in 432 documents (13 OK rows of the
  measured set; 0 now). The own centroid comes from, in order: the new sidecar `catastro_own_centroids.parquet` (one row
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
  (`AddressNormalizer.encode_queries_exact`, `search_exact`). On GPU, 8,435 rows in one batch vs batches of 64: 10 status
  flips and `confianza` differences up to 0.0101 before; 0 flips and 1e-4 after. `score_components([])` returns empty
  arrays instead of raising. Caveat: `normalize_batch` (used by `scripts/ablation.py`, `scripts/smoke_eval.py` and
  `scripts/build_notebook.py`) still runs the bf16 `encode_queries` / `search`, so the batch-size independence holds for
  the `score_components` / `normalize_strict` path (CLI, API, pipeline), not for those scripts. Same input twice: 0
  differences on 26,287 rows.
- `esquina` now covers two situations: no plate, **or** a street letter on one side only. Consumers must read `motivo`
  (`nivel maximo esquina: via con letra en un solo lado`) to tell them apart.
- Parser: `TORRE O 2` keeps the 2 (`TO O2`, like `TORRE B 2`); a glued `T2` is a tower (`T2 APTO 301` -> `TO 2`, `AP 301`);
  the bare `T` alias is not applied after a place word or a connector (`ESQUINA T 2`), a marker word (`NO T 2`) or a
  number, and takes at most 3 digits. A second address typed after a unit without a comma or semicolon stays in
  `complement_text` (documented limit; recognised kinds are the only units read from it).

### Recommended rule for a trustworthy NPN

`estado == "OK"` and `nivel_precision == "predio"` and `confiabilidad >= 0.58` (CLI configuration, 26,287 real rows,
clean references):

| Gold | Coverage of the OK rows | Same lot | More than 100 m away |
| --- | --- | --- | --- |
| SIG, all OK | 100 % | 90.1 % | 2.27 % |
| SIG, rule | 87.4 % | 94.1 % | 1.20 % |
| GPS, all OK | 100 % | 58.7 % | 5.08 % |
| GPS, rule | 85.2 % | 61.0 % | 3.28 % |

Do not raise `confiabilidad` above ~0.62: the relation is not monotonic (GPS at >= 0.64: 50.9 % same lot, 3.87 % beyond
100 m). GPS lands on the facade, so its same-lot figure is a floor.

### Changed

- `normalize_strict` defaults are now the CLI / API / pipeline ones: `barrio_buffer_m=CLI_BARRIO_BUFFER_M` (1000 m) and
  `gate_escalate=DEFAULT_GATE_ESCALATE` (`True`), one constant for every entry point. Library callers that relied on the
  old 200 m / no escalation must pass them explicitly. Measured on the SIG layer and field GPS: +619 / +202 `OK` rows,
  same share of lot and >100 m errors.
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
