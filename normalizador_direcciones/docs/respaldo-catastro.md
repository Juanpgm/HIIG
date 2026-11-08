# Respaldo de catastro (nivel de último recurso, reversible)

## Qué es

El índice v2 se construye solo con la nomenclatura oficial (WFS). Una parte pequeña de los lotes reales existe en el
catastro pero en ningún registro de nomenclatura, de modo que ninguna consulta puede llegar a ellos. En la evaluación final
del candidato (`v2e_cert_pcer_s42_r3_cal`) la única falla significativa fue de cobertura: 30 lotes de la referencia `gold`
del conjunto congelado ausentes del índice (la línea base de producción: 7); 17 de ellos no están en ninguna fuente de
nomenclatura y 10 sí están, pero no fueron elegidos.

El respaldo de catastro agrega, **después de todos los documentos del índice base**, los documentos del catastro cuyos lotes
no tiene la nomenclatura. Es una decisión explícita y reversible: si más adelante la cobertura se mejora de otra forma, se
retira sin reconstruir nada (sección «Cómo revertirlo»).

## Regla (determinista, sin parámetros)

Un documento del catastro (`artifacts/catastro_docs.parquet`) se agrega si y solo si:

1. su lote válido (`NPN[:21]`: 21 dígitos con prefijo `76001`) **no está** en el índice base, entendido como la unión de
   los lotes de los documentos, de la tabla de unidades (`nomenclatura_units.parquet`) y de **todos los registros de
   nomenclatura conservados** (así, un lote que existe pero no fue elegido **no** se vuelve a agregar), y
2. su `doc_key` no colisiona con el de un documento existente (ni con uno ya agregado; gana el primero). Las colisiones
   omitidas se cuentan.

La prueba de lote se hace contra el índice base y nunca crece durante la selección: varias direcciones de catastro de un
mismo lote ausente se agregan todas.

Qué conservan los documentos agregados: el texto `direccion` del catastro sin modificar, NPN, manzana, comuna, barrio,
centroide, `n_predios` y `manzana_nunique`. Además:

| Columna | Valor |
| --- | --- |
| `fuente` | `catastro` |
| `estado_nomenclatura`, `npn_catastro`, `direccion_catastro`, `zona` | nulos |
| `objectid` | `objectid` del catastro + `CATASTRO_OBJECTID_OFFSET` (20 000 000); el original queda en `source_objectid` |
| `n_units`, `spread_m`, `n_lots_cert`, `n_lots_pcer` | 0 (valores neutros: no hay registros de nomenclatura) |
| `lot_conflict` | `False` |
| `npn_choice` | `catastro` |
| `doc_id` | continúa desde el último documento base |

Los identificadores y el orden de los documentos base no cambian, de modo que los embeddings del índice base siguen siendo
válidos como las primeras `N` filas de la matriz nueva. El modelo **no se reentrena**: `train_data.npz`,
`spatial_negatives.npy`, `nomenclatura_units.parquet` e `index_dropped.parquet` son los archivos base sin cambios.
`train_pairs.parquet` conserva los pares base y agrega, al final, los pares que el índice base había descartado y cuyo lote
de `gt_predial` corresponde a un documento agregado.

## Cómo construirlo

Siempre en directorios nuevos; nada de lo existente se modifica (CPU; el entorno de entrenamiento no es necesario).

```bash
# 1. Índice con respaldo (index_meta.json se escribe al final; bloque `fallback` con regla, hashes y contadores)
.venv/Scripts/python.exe scripts/index_fallback.py add \
    --base-index artifacts/index_v2e_cert_pcer_r3 --splits-dir artifacts/splits \
    --out-dir artifacts/index_v2g_cert_pcer_catfb_r3

# 2. Experimento: modelo del base + embeddings (filas base copiadas y verificadas bit a bit; filas nuevas con el mismo modelo)
.venv/Scripts/python.exe scripts/index_fallback.py experiment \
    --index-dir artifacts/index_v2g_cert_pcer_catfb_r3 --model-dir artifacts/experiments/v2e_cert_pcer_s42_r3 \
    --out-dir artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 --device cpu

# 3. Calibración (mismo procedimiento que el candidato; `--catastro-feature` es opcional, ver más abajo)
.venv/Scripts/python.exe scripts/fit_decision.py --artifacts-dir artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 \
    --pairs artifacts/splits/train_pairs.parquet --out-dir artifacts/experiments/calib_catfb/params --tag catfb \
    --device cpu --catastro-feature
.venv/Scripts/python.exe scripts/make_decision_experiment.py --source artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 \
    --out artifacts/experiments/v2g_cert_pcer_catfb_s42_r3_cal \
    --decision artifacts/experiments/calib_catfb/params/decision_catfb.json \
    --reliability artifacts/experiments/calib_catfb/params/reliability_calibrated_catfb.json
```

`add` rechaza: un directorio de salida no vacío o igual al índice base, un índice base que ya tiene respaldo, un índice base
cuyo manifiesto sha256 no coincide, archivos de catastro o de pares que no son los que usó el índice base, y cualquier
retroceso del techo de lote (el número de filas `gold` cuyo lote queda cubierto no puede bajar respecto del índice base).
`--no-kept-lots` omite la lectura de la nomenclatura y deja solo los lotes de documentos y unidades como «cubiertos»
(más permisivo: agrega más documentos).

## Cómo revertirlo

Hay tres mecanismos independientes; cada uno tiene pruebas automáticas.

1. **El índice base y los experimentos base nunca se modifican.** El respaldo vive en directorios nuevos
   (`index_v2g_...`, `v2g_...`); el modelo (`model.pt`, `tuning.json`) se enlaza sin cambios. Servir el índice base es
   simplemente seguir usando `index_v2e_cert_pcer_r3` y `v2e_cert_pcer_s42_r3`.
2. **Retirar las filas agregadas** (sirve para el índice y para el directorio de experimento):

   ```bash
   .venv/Scripts/python.exe scripts/index_fallback.py remove \
       --index-dir artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 --out-dir <directorio nuevo>
   ```

   Corta la cola de `catastro_docs.parquet`, del archivo de centroides, de `train_pairs.parquet` y de `catastro_emb.pt`,
   restituye `index_meta.json` desde la copia del base guardada en `fallback.base_meta` y **verifica cada archivo contra el
   manifiesto del índice base** (y contra el sha256 del `catastro_emb.pt` base registrado en `experiment.json`); si algo no
   es idéntico byte a byte, borra la salida y falla. Prueba con los datos reales: ida y vuelta `add` → `remove`, los ocho
   archivos del índice y `catastro_emb.pt` tienen exactamente el sha256 de los archivos del índice base.
3. **Interruptor en tiempo de servicio, sin reconstruir**: la clave `exclude_fuentes` (lista) de `decision.json` y/o la
   variable de entorno `NORMALIZER_EXCLUDE_FUENTES` (valores separados por comas), por ejemplo
   `"exclude_fuentes": ["catastro"]` o `NORMALIZER_EXCLUDE_FUENTES=catastro`. Se combinan (unión). Con el interruptor
   activo el servicio atiende **exactamente** el índice sin esos documentos:

   - no se filtran candidatos después de recuperar (eso reduciría el `top-k` efectivo): los documentos excluidos **dejan de
     existir** para la búsqueda (se cortan todos los arreglos por documento y se reinician los cachés derivados de los
     documentos: índice de interpolación y llaves de catálogo), así que el `top-k` se llena con los documentos restantes
     igual que en un índice que nunca los tuvo;
   - los documentos excluidos deben ser una **cola contigua** del índice (así quedan los de catastro); si no, falla con un
     mensaje claro, porque excluirlos correría los identificadores de los demás. Excluir todos los documentos también
     falla;
   - se aplica una vez por objeto normalizador (en la primera llamada a `normalize_strict`) y no se deshace: pedir otro
     conjunto después lanza `RuntimeError`; para cambiarlo se crea un normalizador nuevo;
   - sin la clave y sin la variable no cambia nada (la clave no se agrega a los `decision.json` existentes);
   - al reconstruir los cachés de disco (`interp_index_v1.*`, `catalog_keys_v1.*`) del directorio, se escriben los del
     índice reducido: son regenerables y se reemplazan de forma atómica.

   Prueba con los datos reales: el experimento con respaldo y la calibración del candidato base, con
   `"exclude_fuentes": ["catastro"]` en `decision.json`, produce sobre `dev` (completo y `gold`) filas **idénticas** (las 55
   columnas, 2 078 y 289 filas) a las del candidato base `v2e_cert_pcer_s42_r3_cal`.

## Calibración y la característica `src_catastro`

Los documentos agregados son de catastro, que es la fuente de la verdad del puntaje a nivel de lote, y su fiabilidad puede
ser distinta a la de los documentos de nomenclatura. `scripts/fit_decision.py --catastro-feature` (por omisión apagado) agrega
una 25.ª característica binaria, `src_catastro` (1,0 cuando el documento elegido tiene `fuente == "catastro"`, signo libre),
**solo si** al menos 30 filas del ajuste llevan cada uno de sus dos valores; en otro caso el artefacto es el de 24
características, idéntico al de siempre. Todos los archivos de parámetros existentes cargan y predicen igual: las
características opcionales viven aparte (`decision.OPTIONAL_FEATURE_NAMES`), el modelo solo recibe `src_catastro` si su
lista de características la nombra, y el ajuste por omisión del índice base reproduce los parámetros del candidato con
diferencia 0,0 (verificado).

## Efecto en las métricas (solo `dev`; `frozen_test` no se evaluó)

Evaluación estricta sobre `dev`, CPU, opciones por omisión más `--validate-outputs report`. «Candidato» es
`v2e_cert_pcer_s42_r3_cal`; «respaldo, calibración nueva» es `v2g_cert_pcer_catfb_s42_r3_cal` (reajuste con
`--catastro-feature`); «respaldo, calibración del candidato» es `v2g_cert_pcer_catfb_s42_r3_basecal` (los mismos parámetros del
candidato sobre el índice con respaldo, para aislar el efecto de los documentos).

| `dev` | Candidato | Respaldo, calibración del candidato | Respaldo, calibración nueva | Línea base producción |
| --- | --- | --- | --- | --- |
| `gold`: filas OK (de 289) | 281 | 283 | 283 | 278 |
| `gold`: precisión de lote | 0,9880 | 0,9881 | 0,9885 | 0,9960 |
| `gold`: precisión de manzana | 0,9928 | 0,9929 | 0,9929 | 0,9964 |
| `gold`: `lost_correct` | 5 | 6 | 6 | 9 |
| `gold`: filas con predial | 251 | 252 | 261 | 250 |
| `gold`: lotes ausentes del índice | 4 | 1 | 1 | 0 |
| completo: filas OK (de 2 078) | 1 217 | 1 231 | 1 323 | 1 084 |
| completo: precisión de lote | 0,6667 | 0,6631 | 0,6593 | 0,6566 |
| completo: precisión de manzana | 0,8614 | 0,8609 | 0,8485 | 0,8630 |
| completo: `lost_correct` | 115 | 121 | 90 | 108 |
| completo: filas con predial | 943 | 956 | 1 015 | 837 |
| completo: lotes ausentes del índice | 76 | 16 | 16 | 14 |

Cobertura (solo conteos, sin evaluar el modelo): se agregan 6 941 documentos (6 602 lotes que el índice base no tenía) y se
omiten 2 245 colisiones de `doc_key`. Filas `gold` cuyo lote está en el índice: `dev` 285 → 288 de 289; `frozen_test`
387 → 402 de 417 (lotes ausentes 30 → 15). Los 15 restantes del conjunto congelado incluyen los 10 que existen en la
nomenclatura pero no fueron elegidos; la regla no los toca por diseño. El techo de lote del constructor pasa de 671 a 689
filas `gold` coincidentes en total.

Lectura. Con la calibración del candidato, el respaldo recupera lotes sin costo visible en `dev` (precisión de manzana
0,8609 frente a 0,8614). El reajuste de la calibración (umbral `tau_p` de 0,94 a 0,89) responde más filas, pero baja la
precisión de manzana del conjunto completo a 0,8485, por debajo del piso 0,858 que se usó para el candidato; la
característica `src_catastro` queda con coeficiente −0,44 (estandarizado) en la cabeza de lote y ≈ 0 en la de manzana. Un
control sin la característica (reajuste de 24 características sobre el índice con respaldo, `tau_p` 0,885) da los mismos
resultados (completo: 1 323 OK, precisión de manzana 0,8485; `gold`: 283 OK), de modo que el cambio viene de reajustar con
las 95 filas de ajuste que el respaldo vuelve respondibles y no de `src_catastro`. La decisión entre las dos calibraciones
corresponde a quien selecciona el candidato.

Precisión de lote por fuente (filas OK con predial y lote de referencia, intervalo de Wilson al 95 %):

| `dev` | `nomenclatura_cert` | `nomenclatura_pcer` | `catastro` |
| --- | --- | --- | --- |
| `gold`, respaldo (calibración nueva) | 196/197 = 0,995 [0,972; 0,999] | 60/62 = 0,968 [0,890; 0,991] | 2/2 = 1,000 [0,342; 1,000] |
| completo, respaldo (calibración nueva) | 225/342 = 0,658 [0,606; 0,706] | 100/149 = 0,671 [0,592; 0,741] | 2/5 = 0,400 [0,118; 0,769] |

Calibración fuera de muestra (OOF, 5 pliegues por grupo): cabeza de lote, candidato `n=5087`, ECE 0,0044, Brier 0,0271, AUC
0,895; con respaldo `n=5182`, ECE 0,0055, Brier 0,0256, AUC 0,919. Cabeza de manzana, candidato `n=5014`, ECE 0,0022, Brier
0,0040, AUC 0,915; con respaldo `n=5107`, ECE 0,0011, Brier 0,0039, AUC 0,850. Las tablas por banda están en
`strict_eval_dev_*_validate_by_band.csv` de cada directorio.

## Riesgos y límites

- **Calidad del documento de catastro.** Sus direcciones llevan complementos y no están normalizadas como las de la
  nomenclatura; en `dev` completo solo 2 de 5 filas con lote de referencia servidas desde `catastro` acertaron el lote (la
  muestra es pequeña: ver el intervalo). El interruptor existe precisamente para apagarlo.
- **Etiquetas de los pares.** Un par cuyo documento antiguo tiene la llave de un documento agregado pero cuyo `gt_predial`
  es de un lote de nomenclatura se queda en su documento base (se prefiere el predial); solo los pares que el índice base
  había descartado pasan al documento agregado, por lote.
- **Embeddings.** Las filas agregadas se calculan en CPU (fp32) y las base provienen del dispositivo de entrenamiento: una
  recomputación en GPU puede diferir en los últimos dígitos de fp16. Las filas base se copian y se verifican bit a bit.
- **Interruptor de servicio.** Es por proceso/objeto y destructivo sobre el normalizador (ver arriba); con ONNX corta además
  la matriz fp16 del índice documental (`_doc_index._docs`), un detalle interno que debe revisarse si el backend ONNX cambia.
- **Lotes presentes pero no elegidos.** El respaldo no los recupera (decisión de diseño); requieren otra solución en la
  construcción del índice de nomenclatura.
- **Datos.** El respaldo usa solo datos públicos de catastro; los informes contienen únicamente conteos y tasas.
