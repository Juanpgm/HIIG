# Ground-truth pair set

## Purpose

Measure how the matcher behaves at threshold 0.90 with pairs whose truth does
not come from the matcher itself. The primary unit of analysis is the
BUILDING / PREDIO (the door), not the unit inside it.

## Why the previous eval is circular

`tools/eval_catastro.py` builds its positives by rewriting addresses with the
matcher's own grammar (parse, then re-render), so it mostly confirms what the
parser already accepts. Its negatives are one-field mutations, which the hard
vetoes reject by construction. Neither tier can reveal a wrong policy (for
example a unit vs. its bare building) or a spelling the grammar never saw.

## What is generated

`python tools/build_groundtruth.py --catastro <parquet> [--out-dir data/groundtruth] [--private-dir data/groundtruth_private] [--seed N]`

| File | Content |
|---|---|
| `data/groundtruth/synthetic_positives.jsonl` | `{pair_id, addr_a, addr_b, same_door:true, tier:"synthetic", transform, source_direccion}`. addr_b is a mutated spelling of a real address, made with rules inside the builder. Every rule keeps all digits (checked). Rationale per tag: `MANIFEST.json`. Tags are either known-vocabulary (spellings the parser was built to know) or `novel_*` (plausible spellings that were outside the parser tables when the set was designed, now learned by the parser and kept as regression coverage: `APT`/`APART` for AP, `LOC` for LC, `OFIC`, `EDIF`/`EDF`, `BLOQ`, `CARR`/`CRRA`, `AVDA`/`AVEN`, `DIAG`, `NUMERO` instead of `#`). `hash_absent` is only applied to streets with one digit run; the parser accepts the no-`#` form only for the strictest shape (`TYPE N N - N`), so the multi-token variants (`KR 17 B 41 - 78`) remain deliberately unparseable. |
| `data/groundtruth/candidates_to_label.jsonl` | `{pair_id, addr_a, addr_b, stratum, npn_a, npn_b, comuna_a, comuna_b, barrio_a, barrio_b, same_manzana}`, about 60 per stratum. NO scores. The stratum lives only here: it is for the analyst and must NOT be given to labelers. |
| `data/groundtruth/random_negatives.jsonl` | `{..., same_door:false, tier:"sanity"}` random pairs from different comunas. |
| `data/groundtruth/label_sheet.csv` | The sheet to fill in. Columns: `pair_id, addr_a, addr_b, query_a, query_b, comuna_a, comuna_b, barrio_a, barrio_b, npn_a, npn_b, same_door, same_unit, evidence_source, notes`. Rows are shuffled (seeded, deterministic) and the stratum column is deliberately absent so the labeler is not anchored; join back to the strata through `pair_id`. `query_a`/`query_b` are the address plus `, Cali, Valle del Cauca, Colombia`. `barrio_*` is whatever the cadastre stores (a code in the current parquet). UTF-8, no BOM. |
| `data/groundtruth/MANIFEST.json` | seed, counts per stratum/transform, quota shortfalls, score-bin counts per stratum/tier, sha256 of each file. |
| `data/groundtruth_private/matcher_scores_sidecar.jsonl` | `{pair_id, score, decision}` for every pair. Lives in a SIBLING directory on purpose: do not hand it (or the manifest bin counts per pair) to labelers. |

Strata: `complement_one_sided`, `cross_type_presence`, `plate_letter`
(same NUMERIC plate, zero padding ignored, letter present vs absent or two
different letters), `adjacent_or_swap` (plates +/-1..2, via/cross number swap),
`same_base_diff_complement`, `opaque_or_degenerate` (missing plate, opaque
tail, unparseable). A stratum that cannot reach its quota is reported in
`MANIFEST.json` (`shortfalls`), never padded with off-definition pairs.

Same seed gives byte-identical files (verify by building twice into two
scratch directories and comparing).

## Score distribution

The matcher's scores are coarse, so the oversampling weights near the
threshold may have nothing to work with for some strata. The manifest lists
the real bin counts of each build. Do not read the bins as a smooth
distribution.

## Labeling protocol

Two columns, filled independently. Values `yes`, `no`, `unsure` (any case,
surrounding blanks are ignored).

### `same_door` (PRIMARY)

`yes` = both addresses refer to the same building / predio: the same street
(via) and the same plate. Ignore any unit or complement (apartment, block,
house, local, interior...). `no` = different building. `unsure` = cannot tell.
This is the label that drives the headline precision/recall.

### `same_unit` (SECONDARY)

Only asked when `same_door=yes`.

- `yes` / `no` / `unsure`: both sides name a unit and it is / is not the same one.
- `n/a`: REQUIRED when `same_door=no`, and also when either side has no
  unit/complement (nothing to compare). Blank is read as `n/a`.

Reported in its own section; it never touches the headline numbers.

### Worked examples

| addr_a | addr_b | same_door | same_unit |
|---|---|---|---|
| `KR 1 # 67 - 42 BLQ 2 AP 302` | `KR 1 # 67 - 42` | yes | n/a (one side has no unit) |
| `KR 1 # 67 - 42 AP 101` | `KR 1 # 67 - 42 AP 102` | yes | no |
| `KR 1 # 67 - 42 AP 101` | `KR 1 # 67 - 42 APTO 101` | yes | yes |
| `KR 1 # 67 - 42` | `KR 1 # 67 - 44` | no | n/a |
| `KR 1 # 67 - 42 AP 101` | `KR 2 # 67 - 42 AP 101` | no | n/a (different street) |

`evidence_source`: `cadastre_viewer`, `npn`, `field`, `owner`, `google_maps`,
... free text. `notes`: anything worth remembering.

### Evidence: what each tool can and cannot show

Google Maps places pins by interpolating along the street, so units, plate
letters and +/-1-2 plates resolve to the SAME spot. It is fine for street-level
sanity (right street, right area) but it cannot separate those strata. For
`plate_letter`, `adjacent_or_swap`, `complement_*` and `same_base_diff_*` use
the cadastre viewer with `npn_a` / `npn_b` (each NPN is one predio), or field
evidence. Some rows cannot be looked up at all (for example `K 50 # - T`, no
usable plate); mark them `unsure`.

## Merging external labels

Any CSV or JSONL with at least `pair_id` and `same_door` (and optionally
`same_unit`) works. BOM, CRLF, mixed case and stray blanks are fine; extra
columns are ignored. In JSONL, the booleans `true` / `false` count as `yes` /
`no`. The old single `label` column is rejected with a clear error (it mixed
door and unit). Duplicate `pair_id`s are an error; unknown ids are warnings;
blank labels count as not labeled (coverage is reported). A labels file
without a `same_door` column (for example the typo `same-door`) is an error
(exit code 2), like a missing `pair_id`; in JSONL every row needs the
`same_door` key (`null` means blank).

Validation: `same_unit` of `yes`/`no`/`unsure` on a pair with `same_door=no`
is inconsistent; an unrecognised `same_unit` value (for example `maybe`) is
counted as invalid and named in a WARNING when `same_door` is `yes` or `no`. The evaluator prints a WARNING naming the pairs, keeps the
door label and ignores that unit value. `same_unit` on a pair whose
`same_door` is unsure or blank is ignored too (the pair is not scored).

## Evaluating

Hand labelers ONLY `data/groundtruth/label_sheet.csv` (a copy of it). Never
give them `candidates_to_label.jsonl` (it holds the strata) or the private
sidecar. Evaluate with the filled copy:

`python tools/eval_groundtruth.py --labels label_sheet_filled.csv --pairs-dir data/groundtruth [--scores-dir data/groundtruth_private] [--threshold 0.90] [--json-out r.json]`

`--labels` is optional; without it only the synthetic positives and sanity
negatives are scored. By default the pairs are scored live with the matcher,
so a changed matcher is measured as it is; `--scores-dir` reads the frozen
sidecar instead (the report warns when it does). A pair is a MATCH when
`score >= threshold`, as in the matcher. Undefined ratios (for example recall
with no positives) print `n/a`; `unsure` and blank labels are excluded and
counted.

Report sections: coverage, same_door metrics (overall, and the human-labeled
row), per stratum/tier, synthetic recall split into known-vocabulary vs
`novel_*` tags and per transform, a separate same_unit section, and the top
false positives/negatives.

### How to read the numbers (honesty notes)

- The candidate sample is stratified and oversampled with no weights, so the
  human-labeled row gives conditional rates per stratum. That is why it is
  titled "per-stratum rates (stratified sample, not population precision)":
  it is not the matcher's precision or recall on the real cadastre.
- "Overall" mixes synthetic positives and sanity negatives with human labels
  and is not a precision estimate.
- Known-vocabulary synthetic recall is not independent evidence: those
  spellings are the ones the parser was built around. Only `novel_*` recall
  says something about unseen input, and even that is limited to the alias
  families listed above.
- Sanity negatives are random cross-comuna pairs: trivially different, a
  smoke check only.
