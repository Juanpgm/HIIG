# 03 - Configuración

El modelo no usa archivos de configuración ni variables de entorno en tiempo de ejecución. Todos los parámetros se pasan como argumentos de función o de línea de comandos. Este documento enumera cada parámetro, su valor por defecto y su efecto. Para el significado de los puntajes ver [01](01-teoria.md); para interpretar resultados ver [06](06-predicciones.md).

## Contenido

1. [Parámetros de la API](#1-parámetros-de-la-api)
2. [Opciones de `scripts/emparejar.py`](#2-opciones-de-scriptsemparejarpy)
3. [Opciones de `scripts/vincular.py`](#3-opciones-de-scriptsvincularpy)
4. [Códigos de salida](#4-códigos-de-salida)
5. [Variables de entorno](#5-variables-de-entorno)
6. [Constantes fijas en el código](#6-constantes-fijas-en-el-código)
7. [Efecto del umbral y de la tolerancia](#7-efecto-del-umbral-y-de-la-tolerancia)
8. [Parámetros de las herramientas de evaluación](#8-parámetros-de-las-herramientas-de-evaluación)

## 1. Parámetros de la API

| Parámetro | Dónde | Valor por defecto | Valores válidos | Efecto |
|---|---|---|---|---|
| `threshold` | `match`, `match_parsed`, `AddressIndex.find` | `0.90` (`DEFAULT_THRESHOLD`) | Número real en [0, 1]; se rechazan `NaN`, booleanos y no numéricos | Un par es MATCH si `score >= threshold`. Entre 0,86 y 0,9126 las decisiones son idénticas (ver [01](01-teoria.md), sección 8) |
| `plate_tolerance` | `match`, `match_parsed`, `AddressIndex(...)` | `0` (`DEFAULT_PLATE_TOLERANCE`) | Entero mayor o igual a 0 (no booleano) | Permite que placas con la misma letra difieran hasta N unidades numéricas, con penalización 0,04. Con 0, cualquier diferencia de placa es veto |
| `level` | `match`, `match_parsed`, `AddressIndex.find` (solo palabra clave) | `"unit"` (`DEFAULT_LEVEL`) | `"unit"`, `"building"`, `"structure"` | Qué parte del complemento se ignora (ver [01](01-teoria.md), sección 6) |
| `max_block_scan` | `AddressIndex.find` | `None` (sin tope) | `None` o entero mayor o igual a 1 | Máximo de registros puntuados por cubeta. Si una cubeta es mayor, se recorre parcialmente y `LinkResult.truncated = True`; puede faltar un mejor candidato |
| `record_id` | `AddressIndex.add` | - | Cualquier objeto hashable | Identificador del registro; se compara con la igualdad de Python |

Notas:

- En `AddressIndex`, la tolerancia de placa se fija al construir el índice (`AddressIndex(plate_tolerance=N)`) y `find` la reutiliza; `find` no recibe `plate_tolerance`.
- Los ajustes inválidos son errores de programación y lanzan `InvalidThresholdError`, `InvalidToleranceError`, `InvalidLevelError` o `ValueError`. Una dirección inválida nunca lanza: se puntúa como veto (ver [06](06-predicciones.md)).
- Los niveles `building` y `structure` solo están disponibles en la API de Python; los scripts de línea de comandos no los exponen.

## 2. Opciones de `scripts/emparejar.py`

```text
python scripts/emparejar.py <direccion_a> <direccion_b> [--umbral/--threshold N] [--tolerancia-placa/--plate-tolerance N] [--json]
```

| Opción | Alias | Defecto | Efecto |
|---|---|---|---|
| `direccion_a` | - | obligatorio | Primera dirección (canónica) |
| `direccion_b` | - | obligatorio | Segunda dirección (canónica) |
| `--umbral` | `--threshold` | 0.9 | Umbral de coincidencia en [0, 1] |
| `--tolerancia-placa` | `--plate-tolerance` | 0 | Tolerancia entera de placa |
| `--json` | - | desactivado | Imprime un único objeto JSON con las claves `match`, `resultado`, `score`, `threshold`, `plate_tolerance`, `reason`, `a`, `b`, `fields`, `warnings` |

La salida estándar se reconfigura a UTF-8 para evitar errores con tildes en consolas de Windows.

## 3. Opciones de `scripts/vincular.py`

```text
python scripts/vincular.py --base base.csv --id-column ID --address-column DIR [--encoding utf-8] [--umbral N] [--tolerancia-placa N] [--top N] [--max-block-scan N] [--json] "direccion"
```

| Opción | Alias | Defecto | Efecto |
|---|---|---|---|
| `direccion` | - | obligatorio | Dirección a vincular (canónica) |
| `--base` | - | obligatorio | Archivo CSV con la base |
| `--id-column` | - | obligatorio | Nombre de la columna del identificador |
| `--address-column` | - | obligatorio | Nombre de la columna de la dirección |
| `--encoding` | - | `utf-8` | Codificación del CSV; se omite un BOM UTF-8 inicial |
| `--umbral` | `--threshold` | 0.9 | Umbral en [0, 1]; fuera de rango termina con código 2 |
| `--tolerancia-placa` | `--plate-tolerance` | 0 | Tolerancia de placa del índice |
| `--top` | - | 5 | Cantidad máxima de candidatos mostrados; debe ser mayor o igual a 1 |
| `--max-block-scan` | `--max-bloque` | sin tope | Tope por cubeta; debe ser mayor o igual a 1; si se supera, el resultado se marca TRUNCADO |
| `--json` | - | desactivado | Imprime un único objeto JSON |

Comportamiento del CSV: delimitador coma; lectura estricta (comilla sin cerrar, texto tras comilla de cierre, fila con más campos no vacíos que el encabezado, o un encabezado que parece usar otro delimitador terminan con código 2); las filas con id vacío o solo espacios se omiten y se cuentan (`aviso: N filas sin id omitidas` en el error estándar y `ids_vacios` en el JSON); una fila corta sin dirección cuenta como rechazada.

Claves del JSON de `vincular`: `consulta`, `canonica`, `parse_ok`, `motivo`, `umbral`, `tolerancia_placa`, `n_comparados`, `n_candidatos`, `ambiguo`, `casi_ambiguo`, `alternativas`, `truncado`, `mejor`, `candidatos` (cada uno con `id`, `puntaje`, `canonica`, `avisos`), `registros`, `rechazados`, `ids_vacios`.

## 4. Códigos de salida

| Script | Código | Significado |
|---|---|---|
| `emparejar.py` | 0 | MATCH |
| `emparejar.py` | 1 | NO_MATCH (incluye entrada no legible) |
| `emparejar.py` | 2 | Error de uso (argumentos inválidos, umbral o tolerancia inválidos) |
| `vincular.py` | 0 | Existe al menos un candidato sobre el umbral |
| `vincular.py` | 1 | Ningún candidato (incluye consulta no legible) |
| `vincular.py` | 2 | Error de uso (argumentos, archivo ilegible o vacío, codificación desconocida, columna inexistente, CSV mal formado, `--top < 1`, `--max-block-scan < 1`, umbral fuera de [0, 1]) |

Salidas reales verificadas: `--umbral 1.5` en `emparejar.py` imprime `error: umbral fuera de rango [0,1]: 1.5` y termina con 2; `--id-column nope` en `vincular.py` imprime `error: columna 'nope' no encontrada; columnas disponibles: ['id', 'direccion']` y termina con 2.

Atención: en `vincular.py`, el código 1 mezcla "consulta legible sin coincidencia" y "consulta ilegible". Para distinguirlos use `--json` y revise `parse_ok`.

## 5. Variables de entorno

| Variable | Dónde | Efecto |
|---|---|---|
| `CATASTRO_PARQUET` | Solo `tools/eval_catastro.py` (`ENV_CATASTRO`) | Ruta por defecto del parquet del catastro si no se pasa `--catastro` |

No hay otras variables. En tiempo de ejecución del comparador no se lee ninguna.

## 6. Constantes fijas en el código

No son configurables sin editar el código fuente. Cambiarlas altera el comportamiento y puede invalidar las métricas de [08](08-metricas-y-confiabilidad.md).

| Constante | Valor | Archivo | Efecto |
|---|---|---|---|
| `_PENALTIES` | `one_sided` 0,15; `conflict` 0,30; `cross_type_one_sided` 0,03; `cross_type_conflict` 0,30; `plate_tolerance` 0,04; `complement_one_sided` 0,02 | `matcher.py:132` | Pesos del producto de penalizaciones |
| `MAX_PLATE_DIGITS` | 6 | `parser.py:411` | Máximo de dígitos significativos de la placa; más produce `placa_invalida` |
| `_MAX_TAIL_CHARS` | 40 | `parser.py:136` | Longitud máxima de la cola de complemento |
| `_SHORT_VALUE_MAX_CHARS`, `_SHORT_VALUE_MAX_TOKENS` | 3 y 2 | `parser.py:214-215` | Condición para que `SECTOR`/`AGRUPACION` actúen como tipo |
| `VIA_TYPES` | tabla de alias a código | `parser.py:58` | Tipos de vía reconocidos |
| `COMPLEMENT_KINDS` | tabla de alias a código | `parser.py:139` | Tipos de complemento reconocidos |
| `QUADRANTS` y abreviaturas de una letra | `NORTE`, `SUR`, `ESTE`, `OESTE`; `N`, `S`, `E`, `O`, `W` | `parser.py:116-128` | Cuadrantes |
| `BUILDING_IGNORED_KINDS`, `STRUCTURE_UNIT_KINDS`, `STRUCTURE_DISTINGUISHING_KINDS` | conjuntos | `matcher.py:532, 565, 566` | Definición de los niveles |

Dado que no existe una forma de configuración externa, adaptar el modelo a otra ciudad implica editar estas tablas (ver [10](10-estandar-y-generalizacion.md)).

## 7. Efecto del umbral y de la tolerancia

Los resultados de las corridas no se publican. Para medir el efecto del umbral y de la tolerancia de placa sobre la precisión, el recall y el F1 con sus datos, ejecute la sección F de `tools/eval_catastro.py` (por ejemplo `--sample 20000 --seed 42`; positivos = variantes de formato generadas por el propio proyecto, negativos = mutaciones de un solo campo). Es una medición parcialmente circular (ver [08](08-metricas-y-confiabilidad.md)).

Lectura cualitativa: los puntajes posibles forman un conjunto discreto con un hueco entre 0,85 y 0,9126 (ver [01](01-teoria.md)), de modo que mover el umbral dentro del hueco no cambia las decisiones. Bajar el umbral a 0,85 o 0,70 admite pares con elementos faltantes o en conflicto, a costa de precisión. Una tolerancia de placa mayor que 0 hace que placas vecinas se consideren la misma puerta: úsela solo si su fuente de datos tiene errores de digitación de placa conocidos.

## 8. Parámetros de las herramientas de evaluación

Detalle y esquemas de entrada en [04](04-evaluacion-y-calibracion.md).

| Herramienta | Opción | Defecto |
|---|---|---|
| `eval_catastro.py` | `--catastro` (o `CATASTRO_PARQUET`) | obligatorio |
| | `--seed` | 42 |
| | `--sample` | 20 000 |
| | `--full` | desactivado |
| | `--max-pairs` | 200 000 |
| | `--link-sample` | 5 000 |
| | `--address-col` | `direccion` |
| | `--parcel-col` | `numero_predial_nacional` |
| | `--json-out` | ninguno |
| `build_groundtruth.py` | `--catastro` | obligatorio |
| | `--out-dir` | `data/groundtruth` |
| | `--private-dir` | `<out-dir>_private` |
| | `--seed` | 20260926 |
| | `--n-synthetic`, `--per-stratum`, `--n-sanity` | 250, 60, 30 |
| `eval_groundtruth.py` | `--labels` | opcional |
| | `--pairs-dir` | `data/groundtruth` |
| | `--scores-dir` | ninguno (puntúa en vivo) |
| | `--threshold`, `--plate-tolerance`, `--top`, `--json-out` | 0.90, 0, 10, ninguno |
| `eval_realworld.py` | `--train-pairs`, `--dev`, `--catastro` | `--dev` y `--catastro` obligatorios |
| | `--input` | `raw` (`raw` o `normalized`) |
| | `--normalized-column` | `rule_canonical` |
| | `--level` | `building` (`unit` o `building`; no evalúa `structure`) |
| | `--out-dir` | `data/realworld_private` |
| | `--seed`, `--threshold` | 42, 0.90 |
| | `--json-out`, `--final` | ninguno, desactivado |
