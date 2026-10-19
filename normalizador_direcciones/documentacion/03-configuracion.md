# 03. Configuración

Referencia de todo lo configurable: variables de entorno, opciones de línea de comandos, archivos de configuración, campos de `tuning.json` y `reliability.json`, y parámetros de Docker y Railway. Para entender el efecto conceptual de cada umbral, véase [01-teoria.md](01-teoria.md); para usos concretos, [05-uso-e-implementacion.md](05-uso-e-implementacion.md).

## Contenido

1. [Variables de entorno](#1-variables-de-entorno)
2. [Opciones de la CLI](#2-opciones-de-la-cli)
3. [Archivo de configuración TOML](#3-archivo-de-configuración-toml)
4. [Parámetros del motor de decisión](#4-parámetros-del-motor-de-decisión)
5. [`artifacts/tuning.json`](#5-artifactstuningjson)
6. [`artifacts/reliability.json`](#6-artifactsreliabilityjson) y [`reliability_v2.json`](#61-artifactsreliability_v2json-calibrador-alternativo-no-se-sirve-por-defecto)
7. [Hiperparámetros de entrenamiento](#7-hiperparámetros-de-entrenamiento)
8. [Docker y Railway](#8-docker-y-railway)
9. [Archivos de proyecto](#9-archivos-de-proyecto)

## 1. Variables de entorno

Solo se listan nombres. Ninguna variable contiene secretos en este repositorio.

| Variable | Ámbito | Valor por defecto | Efecto |
|----------|--------|-------------------|--------|
| `NORMALIZER_DEVICE` | API | `cpu` | Dispositivo de inferencia (`cpu` o `cuda`). La CLI `web` la define a partir de `--device`. |
| `ARTIFACTS_DIR` | API y Docker | `<raíz>/artifacts` | Directorio con los archivos de servicio. La CLI `web` la define a partir de `--artifacts-dir`. |
| `BASEMAPS_DIR` | API y Docker | `<raíz>/basemaps` | Directorio con los dos GeoJSON de límites. |
| `DOCS_DIR` | API y Docker | `<raíz>/docs` | Directorio desde el que se sirve `/docs-integracion` (busca `API.md`). |
| `MAX_UPLOAD_MB` | API | `25` | Tope del cuerpo de la petición en megabytes. Por encima responde 413 `payload_too_large`. |
| `MAX_ROWS` | API | `20000` | Tope de filas por petición de normalización. Por encima responde 422. |
| `WARMUP_TIMEOUT_S` | API | `300` | Segundos que una petición espera a que el modelo termine de cargar antes de responder 503. |
| `PORT` | Docker y Railway | `8000` | Puerto de escucha de uvicorn (lo inyecta Railway). |
| `CALI_ARTIFACTS_DIR` | CLI y scripts | `<raíz>/artifacts` | Directorio de artefactos por defecto de la CLI y de los scripts de entrenamiento, ajuste y evaluación. |
| `PYTHONPATH` | Scripts, Docker | - | Los scripts de `scripts/` se ejecutan con `PYTHONPATH=src`; la imagen define `PYTHONPATH=/app/src`. |
| `EPOCHS` | `scripts/train.py` | 8 | Número de épocas (heredado; equivale a `--epochs`). |
| `MAX_MINUTES` | `scripts/train.py` | 24 | Presupuesto de tiempo de entrenamiento en minutos. |
| `SMOKE` | `scripts/train.py` | `0` | Con `1`, ejecuta una corrida mínima (una época, 3 minutos) sin escribir en el directorio de producción. |
| `N_TUNE` | `scripts/tune_threshold.py` | `10000` | Tamaño de cada submuestra de positivos para ajustar pesos y umbral. |

La CLI y la API usan **variables distintas** para el directorio de artefactos (`CALI_ARTIFACTS_DIR` frente a `ARTIFACTS_DIR`); definir una no afecta a la otra, salvo que `cali-address web` copia su `--artifacts-dir` a `ARTIFACTS_DIR` para el proceso del servicio.

Variables adicionales fijadas en el `Dockerfile` (sin efecto sobre el modelo): `PYTHONUNBUFFERED`, `PYTHONDONTWRITEBYTECODE`, `PIP_NO_CACHE_DIR`.

## 2. Opciones de la CLI

Se invoca como `cali-address <subcomando>` o `python -m cali_address <subcomando>`.

### 2.1 `normalize`

`cali-address normalize [INPUT] [opciones]`. `INPUT` es un archivo, una URL http(s) o una URL de SQLAlchemy (puede venir en el TOML). Sin `-o`, escribe `<nombre de entrada>_normalizado<ext>` junto a la entrada local.

| Grupo | Opción | Valor por defecto | Efecto |
|-------|--------|-------------------|--------|
| Salida | `-o`, `--output` | derivado de la entrada | Archivo de salida (`.csv .tsv .xlsx .parquet .json .jsonl .geojson`). |
| Salida | `--output-format` | por extensión | Fuerza el formato de salida. |
| Entrada | `--format` | por extensión o esquema | Fuerza el formato de entrada. |
| Entrada | `--sheet` | primera hoja | Hoja de Excel. |
| Entrada | `--header-row`, `--header` | detección automática | Fila de encabezado (base 0). |
| Entrada | `--delimiter` | detección (`, ; TAB \|`) | Delimitador de CSV. |
| Entrada | `--encoding` | detección (BOM, UTF-8, cp1252) | Codificación del texto. |
| Entrada | `--table` | - | Tabla o vista (SQL) o capa (GeoPackage). |
| Mapeo | `--address-col`, `--column`, `-c` | detección automática | Columna de dirección. Si dos columnas son igual de plausibles, la CLI se detiene y lista candidatas. |
| Mapeo | `--address-parts` | - | Columnas a unir en orden (excluyente con `--address-col`). |
| Mapeo | `--parts-sep` | espacio | Separador al unir partes. |
| Mapeo | `--id-col` | - | Columna identificadora que siempre se conserva. |
| Mapeo | `--municipality-col` | desactivado | Valores no vacíos distintos de Cali producen `FUERA_DE_AREA` sin buscar. |
| Mapeo | `--lat-col`, `--lon-col` | - | Solo se validan e informan; nunca influyen en la coincidencia. |
| Mapeo | `--keep-columns` | `all` | `all`, `none` o lista de columnas a conservar. |
| Ejecución | `--chunk-size` | `20000` | Filas por bloque (limita la memoria; no cambia los resultados). |
| Ejecución | `--on-error` | `mark` | `mark` convierte una fila fallida en `ERROR`; `raise` aborta. |
| Ejecución | `--device` | `cuda` si hay, si no `cpu` | Dispositivo de inferencia. |
| Ejecución | `--dry-run N` | - | Procesa solo las primeras N filas, las imprime y no escribe. |
| Ejecución | `--summary-json` | - | Guarda el resumen de la corrida. |
| Ejecución | `--config` | - | Archivo TOML con las mismas opciones; la línea de comandos tiene prioridad. |
| Rutas | `--artifacts-dir` | `CALI_ARTIFACTS_DIR` o `artifacts/` | Directorio de artefactos. Debe ser de confianza (ver [05](05-uso-e-implementacion.md), seguridad). |
| Rutas | `--basemaps` | `basemaps/` | Directorio de límites geográficos. |
| Motor | `--threshold` | el de `tuning.json` (0.73) | Sustituye el umbral de confianza (0 a 1). |
| Motor | `--min-struct` | `0.6` | Acuerdo estructural mínimo. |
| Motor | `--plate-tolerance` | `0` | Diferencia de placa aceptada sin degradar (0 a 2). |
| Motor | `--ambiguity-delta` | `0.02` | Abstención por ambigüedad (0 a 0.2; 0 la desactiva). |
| Motor | `--barrio-buffer` | `1000` | Holgura en metros alrededor del barrio detectado. |
| Motor | `--zone-buffer` | `300` | Holgura en metros alrededor de comuna o corregimiento. |
| Motor | `--gate-escalate` / `--no-gate-escalate` | activado | Si el barrio no deja candidatos, amplía a comuna y luego corregimiento. |
| Motor | `--gazetteer` / `--no-gazetteer` | activado | Detección de lugar y puerta geográfica. |
| Motor | `--soft-rules` | ninguna | Nombres de reglas cuya violación pasa de dura a blanda (ver sección 4). |
| Motor | `--max-soft` | `1` | Violaciones blandas que una fila `OK` puede llevar (0 a 20). |
| Motor | `--gate-fallback` / `--no-gate-fallback` | desactivado | Si la puerta deja cero candidatos, conserva los ordenados por texto con nivel limitado a `manzana`. |

Códigos de salida: `0` correcto; `1` abortado (`--on-error raise` o falta de memoria); `2` uso, mapeo o configuración; `3` error de E/S (incluye artefactos faltantes); `4` todas las filas terminaron en `ERROR`; `130` interrumpido.

### 2.2 Otros subcomandos

| Subcomando | Opciones | Efecto |
|------------|----------|--------|
| `setup` | `--artifacts-dir`, `--force`, `--device`, `--batch-size` (4096) | Verifica los artefactos, regenera `catastro_emb.pt` a partir de `model.pt` y `catastro_docs.parquet` (escritura atómica) y ejecuta una autoverificación con tres direcciones inventadas. |
| `web` | `--host` (127.0.0.1), `--port` (8000), `--artifacts-dir`, `--device`, `--no-browser` | Sirve la API y la página de carga en la máquina local. |
| `inspect` | `--json` y las opciones de entrada | Lee un conjunto de datos sin normalizarlo; muestra formato, columnas, columna sugerida y vista previa. |
| `formats` | - | Lista los formatos de entrada y salida admitidos. |

## 3. Archivo de configuración TOML

`--config run.toml` acepta las mismas opciones, con guion o guion bajo, planas o agrupadas en tablas. Precedencia: valores por defecto, luego archivo, luego línea de comandos. Las claves desconocidas y el TOML mal formado son errores (código 2). Las rutas relativas dentro del archivo (`input`, `output`, `summary_json`, `artifacts_dir`, `basemaps`) se resuelven respecto al directorio del archivo.

```toml
input = "datos.xlsx"
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

## 4. Parámetros del motor de decisión

Los mismos parámetros se aceptan en la CLI, en el cuerpo de las peticiones de la API (salvo los marcados) y en `normalize_strict`.

| Parámetro | Defecto | Rango | Efecto y compromiso |
|-----------|--------:|-------|---------------------|
| Umbral de confianza | 0.73 (de `tuning.json`) | 0-1 | Puntaje fusionado mínimo. Solo configurable por `tuning.json` o `--threshold` de la CLI (no es parámetro de la API, para que una integración no lo relaje por accidente). Subirlo reduce cobertura y falsos positivos. |
| `min_struct` | 0.6 | 0-1 | Acuerdo estructural mínimo entre entrada y candidato. |
| `plate_tolerance` | 0 | 0-2 | Con 0 se exige placa exacta; una diferencia de 1 a 2 se acepta como blanda con nivel `manzana`. Valores mayores no se admiten. |
| `barrio_buffer` | 1000 m | 0-20000 en la API | Holgura de la puerta por barrio. Los límites de barrio pasan por mitad de calle y el puntaje usa centroides de predio. |
| `zone_buffer` | 300 m | 0-20000 en la API | Holgura de la puerta por comuna o corregimiento. |
| `gate_escalate` | `true` | - | Amplía barrio, comuna y corregimiento en lugar de descartar la fila. |
| `ambiguity_delta` | 0.02 | 0-0.2 | Una fila `OK` con `margen` menor pasa a `SIN_MATCH` (`ambiguo: ...`). |
| `gazetteer` | `true` | - | Desactivarlo elimina la detección de lugar y la puerta geográfica. |
| `soft_rules` (solo CLI, biblioteca o `tuning.json`) | ninguna | `letra_via`, `letra_cruce_cardinal`, `letra_cruce_una_cara`, `letra_cruce`, `bis_via`, `bis_cruce`, `cuadrante_compl` | Reglas cuya violación pasa a blanda. Las reglas de identidad (tipo y número de vía, número y tipo de cruce, placa fuera de la ventana, sufijo numérico, piso estructural) nunca se relajan. |
| `max_soft` | 1 | 0-20 | Número de violaciones blandas tolerables. |
| `gate_fallback` | `false` | - | Si la puerta deja cero candidatos, conserva los ordenados por texto. |

Constantes de código que no son configurables sin editar el fuente: pesos de fusión por defecto (`inference.FUSE_WEIGHTS`, sobrescritos por `tuning.json`), escala geográfica `GEO_TAU_M = 600`, ventana blanda de placa `PLATE_SOFT_WINDOW = 2`, tope del margen `MARGIN_CLIP = 0.3`, ventana de competidores `COMPETITOR_WINDOW = 0.05`, número de candidatos por vista `k = 20`.

La API HTTP no expone `soft_rules`, `max_soft` ni `gate_fallback`: los toma del bloque `chosen.decision` de `tuning.json` si existe (en el `tuning.json` publicado no existe, de modo que rigen los valores por defecto históricos).

## 5. `artifacts/tuning.json`

Lo escribe `scripts/tune_threshold.py` y lo lee `inference.load_tuning`. Si falta, el motor usa los valores por defecto del código.

| Campo | Contenido |
|-------|-----------|
| `method` | Procedimiento: `positives` (documentos catastrales retenidos), `negatives` (cadenas basura sintéticas), `queries_replayed` (39 645), `positives_in_distribution` y `positives_hard` (10 000 cada una), `hard_intensity` (2.0), `negatives_used` (5 000), `junk_rate_assumed` (0.2), semillas (`subsample_seed` 1234, `augmentation_seed` 42), `weight_grid_step` (0.05), `weight_combinations` (1 540), `min_geo_weight_constraint` (0.1), `selection_rule`, `threshold_objective` (F1 de "coincide implica documento correcto"), `geo_tau_m` (600), `evaluation_rows_used` (`none`). |
| `chosen.weights` | Pesos de fusión: `sim` 0.7, `fuzz` 0.1, `struct` 0.1, `geo` 0.1. |
| `chosen.threshold` | Umbral de la variante completa: 0.73. |
| `chosen.thresholds` | Umbral por variante de ablación (sus escalas son distintas): `full (+ coord head)` 0.73, `rule (no model)` 0.79, `neural only` 0.88, `neural + rerank` 0.80. |
| `chosen.variant_weights` | Pesos de cada variante de ablación. |
| `chosen.decision` (opcional) | `soft_rules`, `max_soft`, `gate_fallback`. No existe en el archivo publicado. |
| `diagnostics` | Resultado del ajuste: `top1_at_chosen_weights`, `top1_in_distribution_at_chosen`, `top1_hard_at_chosen`, pesos óptimos sin restricción, `f1_at_chosen_threshold`, precisión, exhaustividad, `junk_rate_sensitivity`, comparación de variantes. |

## 6. `artifacts/reliability.json`

Lo escribe `scripts/fit_reliability.py` y lo lee `reliability.load_reliability`; si falta o está mal formado, `confiabilidad` y `confiabilidad_manzana` quedan `null`. El campo `kind` define el formato (`hybrid` por defecto, `cells`, `logistic`). El archivo publicado es `hybrid` (v4). El archivo `artifacts/reliability_v2.json` es un `logistic` (`reliability-v5-gold`) ajustado solo con etiquetas consistentes con el texto: el cargador actual lo lee sin cambios, pero el servicio no lo usa mientras no se renombre a `reliability.json` (cambia la escala; ver [08](08-metricas-y-confiabilidad.md), sección 4). Además de los campos comunes, registra `label_quality`, `code_fingerprint` y `cv_protocol`.

| Campo (`hybrid`) | Contenido |
|------------------|-----------|
| `kind`, `version`, `fit_date` | `hybrid`, `reliability-v4-20260925`, `2026-09-25T21:31:46`. |
| `m`, `m_grid` | Fuerza del prior de suavizado elegida (40) y rejilla probada (10, 20, 40). |
| `n_train` | Número de filas `OK` con verdad de terreno usadas en el ajuste. |
| `ship_rule_passed` | `true` si la validación cruzada cumplió la regla de publicación (ver [04](04-entrenamiento.md)). |
| `source_rows` | Nombre del archivo de filas de ajuste (datos privados, no incluidos). |
| `clip` | Cotas de la probabilidad de manzana: `[0.02, 0.98]`. |
| `min_step_support` | Mínimo de filas por escalón isotónico: 15. |
| `isotonic.x`, `isotonic.y`, `isotonic.n` | Puntos de la función `P(manzana \| confianza)`: abscisas (`confianza`), ordenadas (probabilidad) y filas de apoyo de cada escalón. |
| `q_global`, `q` | `P(predio \| manzana)` global y por tipo de fila (`exacto`, `relajada`, `compartida`, `parcial`) con `n`, `hits` y `q`. |
| `cv` | Métricas de validación cruzada dejando un conjunto fuera (Brier, log-loss, ECE, AUC) para `hybrid`, `logistic`, `cells`, `raw` e `isotonic`. |
| `notes` | Salvedades de las etiquetas (verdad de terreno por GPS). |

### 6.1 `artifacts/reliability_v2.json` (calibrador alternativo; no se sirve por defecto)

Mismo formato de servicio que un `reliability.json` de tipo `logistic`, ajustado con `scripts/fit_reliability.py --kind logistic --gold-only` sobre filas consistentes con el texto (`label_quality = gold`). Versión `reliability-v5-gold-20261017`, 13 variables (`feature_names`: `confidence`, `struct`, `relaxed_letter`, `relaxed_plate`, `log_n_predios`, `nivel_*`, `gate_applied`, márgenes y `n_competidores`), con `means` y `stds` para estandarizarlas y los coeficientes de predio y de manzana. Solo contiene coeficientes y metadatos de procedencia; no hay filas ni direcciones.

- **No se sirve por defecto**: el cargador lee únicamente `reliability.json`. Para probarlo, copie `reliability_v2.json` con ese nombre en un directorio de artefactos aparte (`--artifacts-dir`).
- **Cambia la escala** de `confiabilidad` (media cercana a 0.94 en lugar de 0.63): la regla `confiabilidad >= 0.58` no se traslada; el corte equivalente elegido en desarrollo fue 0.81.
- **No mejora la manzana** de forma medible.
- Su calidad se midió una vez, en un conjunto pequeño; trate cualquier banda construida sobre él como **no validada** (ver [09](09-advertencias-y-limitaciones.md), advertencia 24).

## 7. Hiperparámetros de entrenamiento

Todos son banderas de `scripts/train.py` (`--lr 1e-3`) o pares `--set clave=valor`. Los valores por defecto reproducen la receta de producción (ver [04-entrenamiento.md](04-entrenamiento.md)).

| Grupo | Claves (valor por defecto) |
|-------|----------------------------|
| Optimización | `lr` 3e-3, `weight_decay` 0.01, `epochs` 8, `batch_size` 2048, `pct_start` 0.15, `div_factor` 25, `final_div_factor` 100, `grad_clip` 1.0, `seed` 42, `max_minutes` 24 (script) |
| Pérdida | `label_smoothing` 0.02, `init_temperature` 0.05, `w_coord` 0.5, `w_comuna` 0.2, `w_barrio` 0.2 |
| Arquitectura | `d_model` 256, `n_layers` 4, `n_heads` 8, `d_out` 256, `dropout` 0.1, `max_len` 40 |
| Vistas y negativos | `p_raw_view` 0.25, `n_hard_neg` 1, `plate_neg_weight` 0.0, `plate_neg_count` 8, `plate_neg_max_delta` 20 |
| Pares reales | `real_frac` 0.0, `real_oversample` 1, `real_val_frac` 0.10, `real_split_seed` 42 |
| Modo | `mode` (`mixed` o `finetune`), `freeze_aux` false, `init_from`, `steps_per_epoch` |
| Datos | `realistic_noise` false, `n_variants` 3, `real_pairs` |

Otras banderas de `train.py`: `--out-dir` (`--artifacts-dir`), `--tag`, `--device`, `--smoke`.

Otras banderas de los scripts de evaluación y ajuste: `tune_threshold.py --artifacts-dir`; `fit_reliability.py --kind {hybrid,cells,logistic} --rows --out --candidate-out --force --artifacts-dir`; `eval_strict.py` (`--tag`, `--datasets`, `--gold-only`, `--final`, `--limit`, `--min-struct`, `--plate-tolerance`, `--ambiguity-delta`, `--barrio-buffer`, `--zone-buffer`, `--gate-escalate`, `--soft-rules`, `--max-soft`, `--gate-fallback`, `--basemaps`, `--device`, `--out-dir`, `--artifacts-dir`); `run_experiment.py` (`--tag`, `--set`, `--mode`, `--smoke`, `--final`, `--eval-set`, `--eval-only`, `--from-dir`, `--overwrite`, `--root`, `--leaderboard`, `--no-hash`).

## 8. Docker y Railway

| Elemento | Valor |
|----------|-------|
| Imagen base | `python:3.12-slim` |
| torch | `torch==2.14.0` instalado en una capa propia desde el índice de ruedas de CPU |
| Dependencias | `requirements.txt` (versiones fijadas; sin torch) |
| Copiado a la imagen | `src/`, `docs/`, `basemaps/`, `deploy/artifacts/` |
| Verificación de construcción | `python -c "import cali_address.service, cali_address.api.main"` y `test -f` sobre los siete archivos de servicio y sobre `basemaps/barrios_veredas.geojson` |
| Servidor | `uvicorn cali_address.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1` |
| Puerto | `EXPOSE 8000` |
| Healthcheck | cada 30 s, tiempo de espera 5 s, periodo de arranque 180 s, 5 reintentos, sobre `/health` |
| Variables fijadas | `NORMALIZER_DEVICE=cpu`, `ARTIFACTS_DIR=/app/artifacts`, `BASEMAPS_DIR=/app/basemaps`, `DOCS_DIR=/app/docs` |
| `railway.toml` | Constructor `DOCKERFILE`; `healthcheckPath=/health`; `healthcheckTimeout=300`; `restartPolicyType=ON_FAILURE`; `restartPolicyMaxRetries=3`; `numReplicas=1` |
| Exclusiones de carga | `.dockerignore` y `.railwayignore` excluyen `context/`, `outputs/`, `artifacts/`, `scripts/`, `tests/`, el cuaderno y los directorios de control de versiones |

Los archivos `.dockerignore` y `.railwayignore` citan directorios privados (`context/`, `outputs/`) que no existen en este repositorio; se conservan por seguridad.

Dimensionamiento: se recomienda al menos 2 GB de RAM por instancia (un límite de 1 GB o menos no alcanza). Mida la memoria y el rendimiento en su entorno; las mediciones no se publican (ver `docs/DEPLOY.md`).

## 9. Archivos de proyecto

| Archivo | Contenido |
|---------|-----------|
| `pyproject.toml` | Metadatos y extras (`api`, `sql`, `excel`, `geo`, `train`, `dev`, `all`); Python 3.11 o superior; script `cali-address`. No declara licencia. |
| `requirements.txt` | Dependencias de servicio con versión fija: `fastapi==0.115.12`, `uvicorn[standard]==0.34.2`, `python-multipart==0.0.26`, `pydantic==2.13.2`, `numpy==2.4.4`, `pandas==2.3.3`, `pyarrow==25.0.1`, `openpyxl==3.1.5`, `rapidfuzz==3.14.5`, `pyshp==3.1.6`, `shapely==2.1.2`, `pyproj==3.7.2`. |
| `uv.lock` | Resolución completa de dependencias. |
| `Dockerfile`, `railway.toml`, `.dockerignore`, `.railwayignore` | Despliegue. |
| `deploy/prepare_artifacts.py` | Copia los siete archivos de servicio de `artifacts/` a `deploy/artifacts/` y falla si falta alguno. |
