# Training experiments

Experiment infrastructure (Stage 2a). Every experiment lives in its own directory
`artifacts/experiments/<tag>/`; production artifacts are never written.

## Run one

```bash
PYTHONPATH=src python scripts/run_experiment.py --tag mixed_r30 \
    --set real_frac=0.3 --set plate_neg_weight=0.3 --set epochs=6
PYTHONPATH=src python scripts/run_experiment.py --tag ft_lowlr --mode finetune \
    --set lr=2e-4 --set epochs=3 --set freeze_aux=true
PYTHONPATH=src python scripts/run_experiment.py --tag baseline_prod --eval-only --from-dir artifacts
```

Pipeline: `train.py` -> (`finish_training_artifacts.py` if embeddings are missing) ->
`tune_threshold.py` -> `eval_strict.py` on **dev** (all rows and `--gold-only`) -> one row in
`artifacts/experiments/leaderboard.csv`. The experiment dir holds only what differs from production
(`model.pt`, `catastro_emb.pt`, `tuning.json`, `experiment.json`, `train_report.json`, eval files, `run.log`).
`catastro_docs.parquet`, `train_data.npz`, `spatial_negatives.npy` and plate-negative tables fall back to `artifacts/`.
`--smoke` runs a tiny pass (about 1 minute on the GPU) to prove the pipeline; its row has `smoke=True` and must be
ignored when selecting.

The runner refuses to start when `--tag` is unsafe, an override key is unknown, the experiment dir is non-empty
(`--overwrite` to reuse) or `frozen_test` is requested without `--final`. It hashes the serving files before and
after and exits non-zero (and records `production_untouched` in `experiment.json`) if they changed.

## Index runs (`--index-dir`)

```bash
.venv-train/Scripts/python scripts/run_experiment.py --tag v2a_cert_s42 --index-dir artifacts/index_v2a_cert
.venv-train/Scripts/python scripts/run_experiment.py --tag v2a_cert_s43 --index-dir artifacts/index_v2a_cert --set seed=43
```

`--index-dir DIR` trains on a built index (a directory whose `index_meta.json` was written last by
`scripts/build_index.py`). Before anything is created the runner checks that the marker parses, that `files`
lists the training inputs and that every listed file exists with the recorded sha256; otherwise it prints the
reason and exits **2**. It then hardlinks (`os.link`, `shutil.copy2` as fallback) every listed file into the
experiment dir and copies `index_meta.json` last, so the experiment dir is self-contained: no index file falls
back to `artifacts/`, and a later `--eval-only --from-dir artifacts/experiments/<tag>` works. Re-using a tag
needs `--overwrite`, which replaces the links (it never writes through them, so the source index cannot change;
`train.py --realistic-noise` also removes the linked `train_data.npz` before rewriting it). `experiment.json`
gets an `index_meta` block (id, `n_docs`, tiers, sha256 of the meta, whether the artifact opts in to letter
separation). `--index-dir` is for training runs and cannot be combined with `--eval-only`.

Index runs default `max_minutes` to **120** (an explicit `--set max_minutes=...` wins; other runs keep the
script's 24).

**Early stop fails loudly.** After training the runner reads `train_report.json` (`stopped_early`,
`epochs_configured`, `epochs_completed`; older reports fall back to the length of `history`). If the time budget
cut the run, or the report is missing, it prints the reason, records `training_check` in `experiment.json`, appends
the leaderboard row with `stopped_early=True`, skips tuning/scoring and exits **4**. Such a run is not a candidate.
Exit codes: 2 invalid index, 3 production artifacts changed, 4 training did not finish.

`leaderboard.csv` has two more trailing columns: `stopped_early` and `index_id` (always last). An older file keeps
every existing byte: only its header line is extended, and only when a row that fills the new columns is appended
(older rows then read as missing in those columns; `csv.DictReader` and `pandas.read_csv` both cope).

Lot-level predial (`NPN[:21]`). `gt_predial` is cadastral and lot-level, while a nomenclature NPN may carry a unit
suffix, so every evaluation also scores the lot. New LAST columns (existing ones unchanged):
`strict_eval_*_rows.parquet` gets `correct_predial_lot` and `lot_present_in_index` (ground-truth lot among the index
documents' lots); `strict_eval_*_summary.csv` gets `predial_lot_precision`, `lot_present_in_index_share`,
`n_lot_missing_from_index`; `leaderboard.csv` gets `dev_predial_lot_precision`, `dev_gold_predial_lot_precision`,
`dev_gold_lot_missing_from_index` and, for `--final` runs, `final_gold_ok`, `final_gold_predial_precision`,
`final_gold_predial_lot_precision`. A code is a lot when it is all digits and at least 21 characters long (28, 29, 30
included); None/NaN/blank/short/non-digit never match. `correct_predial_lot` is NaN where `correct_predial` is NaN.
Acceptance A2/A3/A7 use these (see `docs/retrain_2026-10_acceptance.md`, amendment 2026-10-31).

### Validation recall by source

When `train_data.npz` has official queries (`q_source == 1`), each epoch's history row and `train_report.json`
also carry `val_official_*` and `val_synthetic_*` (`recall@1/5/20` and `n`; `None` recall for an empty group) and
`final_val_recall` (last epoch). With no official query these keys are omitted, so old data dirs produce the
history they always did. The overall `recall@k` mixes both groups; neither is comparable across different
indexes. `--smoke` now slices `q_source` together with the other per-query arrays.

### Threshold tuning sample

`tune_threshold.py` tunes on VALIDATION-split queries only (`q_is_val`; training queries are never used). On data
with official queries the synthetic variants are replayed from the leading synthetic block and the official
validation texts are decoded from the stored raw-view tokens; the sample is a stratified mix, half synthetic and
half official (the remainder goes to whichever pool is short), recorded in `tuning.json["method"]`
(`positives_synthetic`, `positives_official`). Old data dirs draw exactly the sample they always did.

## Comparing indexes and runs

```bash
.venv/Scripts/python scripts/compare_indexes.py --old artifacts --new artifacts/index_v2a_cert [--out PATH]
.venv/Scripts/python scripts/compare_runs.py --old artifacts/experiments/baseline_prod_2026_10 --new artifacts/experiments/v2_final
```

`compare_indexes.py` writes `compare_indexes.json` (default: inside the index dir; `--out` leaves the index
untouched): docs by `fuente`, key overlap, `npn_choice`, `n_predios` shift on shared keys, per-comuna counts,
gold `gt_predial` coverage of dev + frozen in each index (gate A7) and the centroid shift of shared NPNs in
metres. Aggregates only. `compare_runs.py` joins the two runs' gold rows by position (it refuses misaligned
`raw_address`) and writes `probe_agreement.json` (counts), `probe_rows.parquet` (row level, keep untracked) and,
when both runs have an `experiment.json`, `acceptance.json`. The checklist and thresholds are in
`docs/retrain_2026-10_acceptance.md`.

### Per-source (`fuente`) outputs

When the served index carries `fuente` / `estado_nomenclatura` (nomenclature indexes), `normalize_strict` appends both
as the LAST output columns (the selected document's values, null without one); with a production-style index nothing
is appended. `eval_strict.py` then adds `fuente` as the last column of `strict_eval_<tag>_rows.parquet` and writes
`strict_eval_<tag>_by_fuente.csv` (per `fuente`, scopes `all` and `gold`: rows, OK rows, manzana / exact predial / lot
predial precision, `lost_correct`; non-OK rows have no document and sit under `(none)`). `compare_runs.py` reports
`by_fuente` in `probe_agreement.json` from that column.

### Retrain report

```bash
.venv/Scripts/python scripts/write_retrain_report.py --out docs/retrain_2026-10_nomenclatura.md \
    --baseline-tag baseline_prod_2026_10 --candidates v2c_cert_s42 v2d_cert_pcer_s42 \
    --final-tag v2d_cert_pcer_s42 --jev-dir artifacts/jev_audit artifacts/jev_audit_synthetic
```

Read-only over `artifacts/` (CPU only; no training stack is imported). Every missing input prints `n/a (pending)`, so it
can run at any point. The verdict line (`PASS` / `FAIL` / `INCONCLUSIVE`) is computed from the final run's
`acceptance.json` (any failed criterion is `FAIL`; otherwise any criterion not evaluated is `INCONCLUSIVE`) and promotion
is always reported as pending explicit approval. The document carries aggregates only (counts, rates, rule names);
labels that do not look like identifiers are masked. Output is deterministic and is not rewritten when unchanged; a
date appears only through `--generated-at`. Exit codes: 0 written, 2 usage error (including an `--out` inside
`artifacts/`, an experiment directory or an index directory).

### Optional source feature in reliability (default OFF)

`fit_reliability.py --kind logistic --source-feature` adds `src_pcer` (1.0 when the selected document's `fuente` is
`nomenclatura_pcer`, else 0.0; needs the `fuente` column in the rows parquet). The artifact lists it in
`feature_names` and carries `"source_feature": true`; only such an artifact makes serving compute the feature.
Artifacts without the flag (production's included) give exactly the same confidences as before.

## Add a variant

1. Prefer a `--set key=value` override; no code needed for any hyper-parameter below.
2. A new behaviour (loss term, sampler, augmentation) goes behind a `TrainConfig` field with a default that
   reproduces today's recipe, plus unit tests (tiny CPU fixtures in `tests/tiny.py`).
3. Add the key to `experiment.py::EXTRA_DEFAULTS` if it is not a `TrainConfig` field, so `--set` accepts it.
4. Run a `--smoke` first, then the real run with a descriptive tag.

## Hyper-parameters (defaults reproduce production)

| Group | Keys (default) |
|-------|----------------|
| Optimisation | `lr` 3e-3, `weight_decay` 0.01, `epochs` 8, `batch_size` 2048, `pct_start` 0.15, `div_factor` 25, `final_div_factor` 100, `grad_clip` 1.0, `seed` 42, `max_minutes` 24 (script) |
| Loss | `label_smoothing` 0.02, `init_temperature` 0.05, `w_coord` 0.5, `w_comuna` 0.2, `w_barrio` 0.2 |
| Architecture | `d_model` 256, `n_layers` 4, `n_heads` 8, `d_out` 256, `dropout` 0.1, `max_len` 40 |
| Views / negatives | `p_raw_view` 0.25, `n_hard_neg` 1, `plate_neg_weight` 0.0, `plate_neg_count` 8, `plate_neg_max_delta` 20 |
| Real pairs | `real_frac` 0.0 (share of each batch), `real_oversample` 1, `real_val_frac` 0.10, `real_split_seed` 42 |
| Mode | `mode` mixed / finetune, `freeze_aux` false, `init_from` (finetune start, default `artifacts/model.pt`), `steps_per_epoch` (override) |
| Data | `realistic_noise` false, `n_variants` 3, `real_pairs` (path override) |

Finetune presets (only for keys you did not set): `lr=3e-4`, `real_frac=0.5`. Architecture always comes from the checkpoint.
`plate_neg_weight` is the probability that a hard-negative slot is a plate neighbour (same block face, plate off by
1..`plate_neg_max_delta`, or same plate on the swapped via/cross) instead of a spatial neighbour. It needs `n_hard_neg >= 1`.
`--realistic-noise` measures structural pattern frequencies on real strings (train pairs + unlabeled RUD strings),
rescales existing augmentation operators (clamped 0.25..4x, none removed) and rebuilds `train_data.npz` inside the
experiment dir. The measured table (numbers only, no text) is stored in `experiment.json["realistic_noise"]`.

## Reading the leaderboard

`syn_recall@k`: held-out synthetic queries (cadastral docs never seen as positives). `real_val_recall@k`: recall on
the group-split real validation fold (10% of `train_pairs` by `address_key`, fixed by `real_split_seed`, so every run is
measured on the same fold; the real training pairs of a run never include it). `dev_*`: the production path
(`normalize_strict`) on `dev.parquet`; `dev_gold_*` is the same on gold rows only. `dev_gold_ok_coverage` = OK gold rows /
gold rows. `dev_lost_correct` = rows the model found but a rule/gate/threshold discarded. `stopped_early` /
`index_id`: see "Index runs" above.
Baseline row: `baseline_prod` (dev gold OK-coverage 0.958, gold predial precision 0.982).

## Selection and leakage rules

- Primary: highest `dev_gold_predial_lot_precision` (lot level, `NPN[:21]`; amended 2026-10-31) among runs with `dev_gold_ok_coverage` (gold rows answered OK / gold rows) >= the MEASURED baseline's (`baseline_prod_2026_10`, current code; the stored `baseline_prod` row is historical).
  Tie-break (within one gold row): `dev_manzana_precision` on all dev rows. Recalls are diagnostics only.
- `frozen_test` is scored only with `--final`, once, on the already-selected candidate. Never select, tune or
  compare variants on it. `eval_strict.py` itself also refuses `frozen_test` without `--final`.
- Real pairs come only from `train_pairs.parquet`; dev/frozen rows are never read for training. The validation fold is
  split by `address_key`, so no address appears on both sides. Note the split notes in `docs/splits.md` (units of one
  building can share a predio across keys).
- Dev is small (289 gold rows): one row is 0.35 points of coverage. Treat differences under ~1-2 rows as noise.

## Retrain / promotion order

1. Experiment runs (never touches production) -> pick by the rule above.
2. `--final` on the single selected candidate (frozen_test).
3. Fit reliability for the winner: `python scripts/fit_reliability.py --artifacts-dir artifacts/experiments/<tag> --rows <dev rows parquet>`
   (writes `reliability.json` there).
4. Only after explicit user approval: back up `artifacts/`, copy the winner's `model.pt`, `catastro_emb.pt`, `tuning.json`,
   `reliability.json` over the production ones (`catastro_docs.parquet` and `gazetteer.pkl` are unchanged), then
   `python deploy/prepare_artifacts.py`.
5. The training tensors (`train_data.npz`) only change with `--realistic-noise`; the production one stays as is.

Standalone scripts keep working with their old defaults and gained `--artifacts-dir`
(`train.py --out-dir`, `finish_training_artifacts.py`, `tune_threshold.py`, `fit_reliability.py`, `eval_strict.py --out-dir`).
The env var `CALI_ARTIFACTS_DIR` sets the default for all of them; `AddressNormalizer(artifacts_dir)` accepts an experiment dir.

## Decision-layer options (Stage 3)

The strict decision layer (`service.normalize_strict`) has three options that default to the historical behaviour and are
read from an optional `chosen.decision` block of the artifact's `tuning.json` when the call does not pass them:

| Key | Meaning | Default |
|-----|---------|---------|
| `soft_rules` | rule classes downgraded from hard to soft (row stays OK, `nivel_precision` capped at `manzana`; a one-sided via letter is capped at `esquina`): `letra_via`, `letra_cruce_cardinal`, `letra_cruce_una_cara`, `letra_cruce`, `bis_via`, `bis_cruce`, `cuadrante_compl` | none |
| `max_soft` | soft violations an OK row may carry | 1 |
| `gate_fallback` | when the geographic gate would leave no candidate, keep the text-ranked ones (note + `manzana` cap) | false |

Identity rules (via type/number, cross number, cross type, plate beyond the soft window, structural floor) are never
softened. `eval_strict.py` takes `--soft-rules/--max-soft/--gate-fallback` overrides; without them it scores the
artifact's own block. A decision-only experiment has no model files: its `experiment.json` declares
`"inherits_model": true` and `AddressNormalizer` reads `model.pt`/`catastro_emb.pt` from `artifacts/` (`paths.model_path`).
