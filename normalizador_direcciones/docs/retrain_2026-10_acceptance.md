# Retrain 2026-10 (nomenclature index): acceptance criteria

Written: 2026-10-30T22:40:04 (before any model was trained on the new indexes).
Machine-readable twin: `artifacts/experiments/retrain_2026_10_acceptance.json` (same thresholds, written
by `cali_address.retrain_compare.build_acceptance_thresholds` from `leaderboard.csv`). The checklist is
evaluated by `cali_address.retrain_compare.check_acceptance` and `scripts/compare_runs.py`; this document
and the code must agree, and `tests/test_retrain_compare.py` pins every boundary.

These criteria were fixed BEFORE running anything. Do not edit them after seeing a candidate's numbers.

## Baseline

The production model scored by the current code on the frozen splits: run `baseline_prod_2026_10`
(`run_experiment.py --eval-only --from-dir artifacts --eval-set both --final`). The reference numbers below
are the `baseline_prod` row of `artifacts/experiments/leaderboard.csv` (dev only; frozen, reliability and
confidence numbers are MEASURED on that run later and read from it, never copied here).

| Metric (dev) | Baseline |
|---|---|
| gold rows (`dev_gold_n`) | 289 |
| gold rows answered OK (`dev_gold_ok`) | 277 (coverage 0.9585) |
| gold predial precision (`dev_gold_predial_precision`) | 0.9819494584837545 |
| gold manzana precision | 0.9963898916967509 |
| all-rows manzana precision (`dev_manzana_precision`) | 0.8431001890359168 |
| all-rows `lost_correct` (`dev_lost_correct`) | 96 |

Noise band, measured with the two production-recipe retrains `s2b_retrain_s42` / `s2b_retrain_s43`:
`dev_lost_correct` 107 and 112, `dev_gold_ok` 277 and 278. That is why A4 allows up to 112.

## Criteria

All comparisons are inclusive unless a strict `<` is shown (equal passes, one unit below fails). A NaN
metric FAILS with a reason. A metric that is missing is reported as NOT EVALUATED and blocks "all passed".

| ID | Criterion |
|---|---|
| A1 | `dev_gold_ok` >= the MEASURED baseline's - 1 (baseline `baseline_prod_2026_10`: 278, so >= 277; the stored `baseline_prod` 277 is historical) |
| A2 | `dev_gold_predial_lot_precision` (lot level, `NPN[:21]`) >= the baseline's, measured on `baseline_prod_2026_10`; the exact `dev_gold_predial_precision` (baseline 0.9819494584837545) is still reported but no longer gates (amended 2026-10-31) |
| A3 | frozen gold OK >= baseline frozen gold OK - 2 rows; frozen gold LOT-level predial precision >= baseline frozen gold lot-level predial precision - 2 / baseline frozen gold OK (amended 2026-10-31) |
| A4 | `dev_lost_correct` <= max(measured baseline, `s2b_retrain_s42`, `s2b_retrain_s43`) (the noise band, 107 and 112 in `leaderboard.csv`; 112 when the band is not recorded) |
| A5 | `dev_manzana_precision` >= the measured baseline's - 0.005 |
| A6 | Reliability refitted on the candidate's dev rows with the SAME `kind` as production's `reliability.json`: for BOTH heads (predio, manzana) the pooled CV calibration error (ECE) <= the refitted baseline's + 0.01, and the ship rule against the refit on the same rows passes (the rule against the v2 reference is NOT required; amended 2026-10-31) |
| A7 | Parser failure rate on certified rows < 1 %; CRS gates pass; (c1) CONSTRUCTION: over the gold rows (dev + frozen) whose lot exists in the cleaned source records, the new index's matched share >= the production index's - 2.0 pp; (c2) COVERAGE LOSS: (production lot share - new lot share) on gold <= the inherent loss (`lot_absent_from_source` / rows) + 1.0 pp; headline coverage and missing-lot counts are reported; `accounting.dropped` is 0 or every drop is explained (amended 2026-10-31: lot level, no dependence on the audit-only catastral NPN) |
| A8 | `production_untouched` is true in every run |
| A9 | Full `pytest` green in `.venv`; `scripts/check_no_real_addresses.py` exits 0 |
| A10 | Relative to the measured baseline, at `confianza >= 0.7` on dev gold rows: (a) share of gold rows with `estado == OK` and `confianza >= 0.7` >= baseline - 1.0 pp; (b) lot-level predial precision among OK rows that EMIT a predial (and have `confianza >= 0.7`) >= baseline - 0.5 pp, same function for both runs, `n` of emitting rows reported; (c) `correct_manzana = False` rate among OK rows with `confianza >= 0.7` <= baseline + 0.5 pp. The full curve (0.50 to 0.95 step 0.05) is reported for both models (amended 2026-10-31) |
| A11 | Cleaning quarantine < 0.5 %; `build_index.py` and `download_nomenclatura.py` exit 0 on the real data and on the anomaly fixtures (row without coordinates, short NPN, unknown state, truncated JSON, missing column gives a clear error, not a traceback) |
| A12 | With `validate_outputs=correct`: A1-A5 and A10 still hold against the baseline AND no metric is worse than the same candidate without correction; 0 violations remain on dev |

### Relative criteria (A1-A6, A10, A12)

They compare against the baseline's MEASURED metrics and are encoded in `check_acceptance`, not as fixed
numbers: the baseline frozen scores (A3), the baseline's refitted reliability (A6), the baseline's confidence
curve at 0.7 (A10) and the uncorrected candidate (A12) are read from the runs at check time. Details of
the implementation choices that the one-line statements above leave open:

* A3: "baseline - 2 / ok_n" uses the BASELINE's frozen gold OK count as `ok_n`, so the floor does not move
  with the candidate.
* A6: both heads are always checked; the error is the pooled cross-validation ECE of the chosen `kind`
  (`cv.pooled.<head>.<kind>.ece`). The baseline must be refitted inside its own experiment directory; the
  production `reliability.json` is never used as the baseline refit.
* A10(a): the denominator is ALL dev gold rows, so a candidate cannot look better by answering fewer rows.
  The share over OK rows is reported next to it. "Gold OK" means `estado == "OK"`.
* A10(b)/(c): a candidate with no row at `confianza >= 0.7`, or none of them emitting a predial, fails (b); (c)
  then counts as 0. A baseline with no row (or no emitting row) at the threshold makes (b) and (c) NOT EVALUATED.
* Unknown `correct_predial` counts as not correct; unknown `correct_manzana` counts as not wrong.

## Amendment 2026-10-31 (before any training run)

Written before any model was trained on the new indexes (the only run so far is the production baseline
re-score, which is not a training run). Two facts found in Phase 0 and the user's scope change ("base the
index only on the new sources") make three criteria measure the wrong thing:

* `gt_predial` is cadastral and lot-level, while a nomenclature NPN may carry a unit suffix (30 % of the
  `ndid` shared by the WFS and the shapefile differ only in the last two digits; the lot, `NPN[:21]`, agrees in
  98.96 %). Scoring the exact code would count a correct lot with another unit as an error.
* The index is built from the nomenclature alone, so documents that exist only in the cadastre are gone by
  construction, and `npn_catastro` is audit-only: it is no longer an index key, so A7 cannot lean on it.

What changes (everything else, including all numeric constants, is as written above):

| Criterion | Before | After |
|---|---|---|
| A2 | `dev_gold_predial_precision` >= baseline | `dev_gold_predial_lot_precision` >= baseline; the baseline is the lot-level value MEASURED on `baseline_prod_2026_10` (production NPNs are already lots, so re-scoring at lot level does not lower it) |
| A3 | frozen gold predial precision (exact) | frozen gold lot-level predial precision, same `- 2 / ok_n` margin |
| A7 | gold `gt_predial` present in the new index as `numero_predial_nacional` or `npn_catastro` >= the old index's | (a) certified parse-failure < 1 %; (b) CRS gates passed; (c1) construction and (c2) coverage loss as in the table above, read from the `oracle_ceiling` block of the new index's `index_meta.json`; `accounting.dropped` rule unchanged |

A7(c) was split in two (coordinator amendment, same date, still before any training run), because losing lots
that do not exist in the new sources is the coverage cost the user explicitly accepted and must not be confused
with a construction defect. The builder writes `oracle_ceiling` in `index_meta.json` (`rows`, `prod_lot_matched`,
`new_lot_matched`, `lot_absent_from_source`, `lot_lost_incomplete`, `lot_not_chosen`); with
`rows_with_lot_in_source = rows - lot_absent_from_source`:

* (c1) CONSTRUCTION: `new_lot_matched / rows_with_lot_in_source` >= `prod_lot_matched / rows_with_lot_in_source` - 2.0 pp.
* (c2) COVERAGE LOSS: `(prod_lot_matched - new_lot_matched) / rows` <= `lot_absent_from_source / rows` + 1.0 pp.
* Reported, never gates: the headline lot shares of both indexes, `gold_lot_missing_new` (rows whose lot the new
  index lacks), `lot_absent_from_source`, `lot_lost_incomplete`, `lot_not_chosen`.

The fields are read defensively: if any is absent, not a non-negative integer, or no gold row has its lot in the
source, A7 is NOT EVALUATED with a reason naming the missing metric (equal passes, one point below fails, NaN
fails).

Definitions. A lot is the first 21 characters of a code that is made only of digits and has at least 21
characters (28, 29 and 30 character codes included); anything else (None, NaN, blank, short, letters) is not a
lot and never matches. Two codes agree at lot level when both are valid lots and their first 21 characters
are equal; the unit suffix is ignored, the lot is not (same lot with another unit is a hit, another lot with
the same unit is a miss). `correct_predial_lot` is NaN where `correct_predial` is NaN (no ground truth, or the
answer carries no predial), so both precisions share one denominator. Without cadastre-only documents the coverage falls by construction; the loss is quantified, never hidden:
`scripts/compare_indexes.py` reports gold and all-with-ground-truth coverage by lot per split, the share of the
production documents still reached by lot and by `doc_key`, and the absolute number of lost rows.

Still reported and unchanged in meaning: `correct_predial`, `predial_precision`, `dev_gold_predial_precision`
and every other existing column (the lot-level columns are appended last in the summaries and in
`leaderboard.csv`). A10(b) is lot-level (second round below).

### Second round (same date, still before any candidate training), from the baseline re-score

The re-score of the production model (`baseline_prod_2026_10`, current code) exposed four problems; each decision
below has a test in `tests/test_retrain_compare.py`, `tests/test_compare_scripts.py` or `tests/test_acceptance_artifacts.py`.

1. **The measured baseline replaces the stored reference.** The current code gives dev gold OK 278 (stored 277),
   `dev_manzana_precision` 0.8630 (0.8431) and `dev_lost_correct` 108 (96), so the stored `baseline_prod` row is
   historical. A1, A4, A5, A2/A3 and the selection rule use the measured run whenever it is provided; the stored
   numbers are only the fallback. A4 keeps the existing noise-band logic but reads the measured value:
   `dev_lost_correct` <= max(measured baseline, `s2b_retrain_s42`, `s2b_retrain_s43`), the two retrains being
   recorded in the thresholds JSON (`noise_band`).
2. **A6 can be evaluated for the baseline.** The baseline refit failed the ship rule against the v2 reference (it
   was fitted on dev rows only, not on the older, larger set the production file used), so `fit_reliability.py`
   wrote `reliability_hybrid_candidate.json` instead of `reliability.json`. `collect_run_metrics` now falls back to
   `reliability_<kind>_candidate.json` when `reliability.json` is absent. A6 compares BOTH heads' pooled CV ECE (baseline
   refit: predio 0.0975, manzana 0.0338) and requires the ship rule against the refit on the same rows (recomputed
   from `cv.pooled`: Brier <= refit + 0.002 and AUC >= refit, both heads); the rule against the v2 reference is
   not a candidate gate because the baseline itself fails it.
3. **A10 is relative and lot level, with the right denominator.** The old curve counted OK rows that emit no
   predial (shared addresses) as wrong, so its precision at 0.7 was 0.8885 for the baseline and the absolute 0.98
   floor was unreachable even for it. The curve now exposes `predial_lot_precision_emitted` and
   `n_emitting_predial` next to the old keys (kept). Margins: share -1.0 pp, precision -0.5 pp, manzana error
   rate +0.5 pp.
4. **Selection rule**: see below; it now ranks by `dev_gold_predial_lot_precision` against the measured baseline's OK coverage.

`accounting.dropped` stays in A7. The `oracle_ceiling` block is read as the builder writes it (per split plus
`total`; `prod_lot_matched_in_source` for the production share on the in-source population).

## Selection rule

Amended 2026-10-31 (see `docs/experiments.md`):

1. Among runs whose `dev_gold_ok_coverage` (gold rows answered OK / gold rows) is at least the MEASURED baseline's
   (`baseline_prod_2026_10`), pick the highest `dev_gold_predial_lot_precision`.
2. Ties (within one gold row) are broken by `dev_manzana_precision` on all dev rows (unchanged).

`real_val_recall@k` and `syn_recall@k` are diagnostics, never selectors, and the synthetic recall is NOT
comparable across different indexes (the validation queries are drawn from different documents). Never select
by `frozen_test`; it is scored once, with `--final`, on the selected candidate. A run whose training stopped
before its configured epochs (`stopped_early` in `train_report.json` and in the leaderboard) is not a
candidate.

## Diagnostics (never selectors)

Row agreement on dev + frozen gold between the baseline and the winner (both OK / only old / only new / both
wrong, joined by position with a `raw_address` equality check, plus the breakdown by `fuente`); "predial
correct against either source" (output NPN or the matched document's `npn_catastro` equals `gt_predial`);
`ambiguous` rate before and after; share of expansion-zone rows among the errors. Only aggregates are
versioned; the row-level tables stay in `artifacts/` (untracked).
