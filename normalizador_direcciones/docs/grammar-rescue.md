# Grammar rescue and the street-name table

Unit N1 (2026-11). No retraining; ONNX weights and `artifacts/` are untouched.

## Why

The ETL reported ordinary-looking addresses as "La dirección no se pudo normalizar". Of the 1.427 EDE cases, 228 were
`NO_PARSEABLE`; about 25 of them failed only because of grammar gaps (leading free text such as `EDIFICIO ...`, a tower
prefix, `CAR` / `AVD` / `OE` / `RA`, ordinal words, a comma after `AV`) and about 20 use street NAMES (Roosevelt,
Autopista, Américas). The rest are truly unparseable (no digits, rural / km) and stay `NO_PARSEABLE`.

## What changed

`cali_address.grammar_rescue` runs inside `service.normalize_strict` **only for rows `parse_address` already rejected**
(after place detection and the existing "raw text fallback"). A row that parses today never reaches it, so its query, model
score and decision are identical. Candidates are tried in this order and the first one that parses wins:

1. verified street-name substitution (`via_names.VIA_NAMES`, see below);
2. variants and abbreviations: `CAR` -> `KR` (before a number), `AVD` -> `AV`, a leading `RA <n>` -> `KR` (never when
   another street word is present), `OE` -> `OESTE`, ordinal words after a street type (`Carrera primera B1` ->
   `KR 1 B1`, `Avenida sexta oeste` -> `AV 6 OESTE`), the comma after the street type (`AV, 3H`), `Av.` with a dot,
   accents;
3. free text before the first street-type word (building / hotel names). The remainder must parse; the prefix must not
   contain digits; a tower / block code found in the prefix (`TORRE D`, `BLOQUE B`, `TO 2`) is moved after the address and
   stays a complement (decision S26: the tower is never dropped). `TORRE DE ...` (a connector) is a name, not tower D.

Nothing guesses digits. Counter: `stats["rescatadas_gramatica"]`.

## Street-name table (S47)

The cadastre carries numbered streets only, so a name is verified indirectly (`scripts/via_names_evidence.py`): for each
distinct door `cross - plate` written with the name in the real raw addresses (ETL silver: EDE, RUD, reportes, geo reports,
stickers), the cadastre says which street owns that exact door. `verified` requires >= 10 distinct doors, >= 50 % of them
present in the cadastre, one street owning >= 70 % of those, and the runner-up <= 50 % of the top. Otherwise `ambiguous`
(thin or contradictory evidence) or `rejected` (not a street name). The reviewable file is
`src/cali_address/data/via_names_review.csv` (variants, canonical street, evidence counts, status, reason).

The parser reads `src/cali_address/via_names.py`, generated from the CSV by `scripts/build_via_names.py` (only `verified`
rows). It is a `.py` on purpose: the ETL engine fingerprint hashes `*.py` only, so a table change must change a `.py`.
To accept a name after review: set its `status` to `verified` and `canonical_via`, run `python scripts/build_via_names.py`.
A substitution only applies when the name is followed by a door (a digit), is not preceded by a place word (`BARRIO`,
`EDIFICIO`, ...), and is not the start of a longer name (`AUTOPISTA SUR NUEVA`); the longest listed name wins.

Result of the first evidence run: **no name is verified** (Roosevelt: CL 6 explains only 30 of 57 doors; Autopista Sur, Estación,
Guadalupe, Colombia: fewer than 10 doors; Américas: only 14 of 31 doors exist in the cadastre). The shipped table is empty
until the owner reviews the evidence file.

## Rollout

The engine fingerprint changes (new `.py` files, `service.py`), so the ETL creates a NEW profile and re-normalizes every
address when this engine is switched in. That is a manual step (unit N2); it is not done by this change.
