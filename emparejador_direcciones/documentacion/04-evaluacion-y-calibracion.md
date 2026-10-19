# 04 - Evaluación y calibración

**No hay entrenamiento.** El modelo no aprende parámetros a partir de datos: no hay conjunto de entrenamiento, función de pérdida, épocas, pesos ni artefacto serializado (ver [01](01-teoria.md)). Lo que el proyecto sí incluye es un **arnés de evaluación** que mide el comportamiento del comparador contra un catastro y contra pares de referencia. Este documento describe ese arnés y cómo repetirlo con el catastro de su ciudad.

La palabra "calibración" aquí significa elegir el umbral y la tolerancia de placa observando métricas; **no** significa calibrar el puntaje como probabilidad (eso no se midió, ver [08](08-metricas-y-confiabilidad.md)).

## Contenido

1. [Herramientas](#1-herramientas)
2. [Protocolo de eval_catastro.py](#2-protocolo-de-eval_catastropy)
3. [Verdad de referencia: build_groundtruth y eval_groundtruth](#3-verdad-de-referencia-build_groundtruth-y-eval_groundtruth)
4. [Evaluación con texto real: eval_realworld.py](#4-evaluación-con-texto-real-eval_realworldpy)
5. [Esquemas de datos requeridos](#5-esquemas-de-datos-requeridos)
6. [Cómo repetirlo en su ciudad](#6-cómo-repetirlo-en-su-ciudad)
7. [Semillas, tiempos y determinismo](#7-semillas-tiempos-y-determinismo)
8. [Privacidad de las salidas](#8-privacidad-de-las-salidas)

## 1. Herramientas

| Herramienta | Qué mide | Requiere |
|---|---|---|
| `tools/eval_catastro.py` | Cobertura de lectura, invariantes, recall en variantes de formato, rechazo de diferencias reales, riesgo de falsos vínculos dentro del catastro, sensibilidad al umbral, rendimiento y vinculación contra el catastro completo | Parquet del catastro, `pyarrow` |
| `tools/build_groundtruth.py` | Genera pares de referencia (positivos sintéticos, candidatos a etiquetar, negativos de control) | Parquet del catastro, `pyarrow` |
| `tools/eval_groundtruth.py` | Puntúa los pares de referencia y, si hay etiquetas humanas, calcula precisión y recall por estrato | Salida del anterior; etiquetas opcionales |
| `tools/eval_realworld.py` | Recall, falsos positivos y recuperación sobre texto real sucio con etiquetas derivadas de GPS | Dos parquets etiquetados y el catastro, `pyarrow` |

Las herramientas importan `pyarrow` solo al ejecutar `main`; sus funciones puras no lo necesitan.

## 2. Protocolo de eval_catastro.py

Cada ejecución toma una muestra determinista (por defecto 20 000 filas, semilla 42) y mide:

| Sección | Qué mide | Cómo se construye |
|---|---|---|
| A | Cobertura de lectura (todas las filas) | Se parsea cada fila; se clasifican las causas de fallo |
| B | Autocoincidencia y simetría | `match(a, a)` debe dar 1,0; `match(a, b) == match(b, a)` sobre mutaciones y pares aleatorios |
| C | Recall sobre variantes de formato | Positivos por construcción: 14 variantes |
| D | Rechazo de diferencias reales | Negativos por construcción: 27 mutaciones de un solo campo |
| E | Riesgo de falsos vínculos dentro del catastro | Pares de la misma cara de manzana con distinta parcela (E1), ambigüedad inherente (E2), distinta dirección con misma placa (E3) |
| F | Precisión, recall y F1 por umbral y tolerancia | Positivos de C contra negativos de D |
| G | Rendimiento | Pares por segundo y tiempo de parseo, un hilo |
| H | Vinculación contra el catastro completo | Índice sobre todas las filas; consulta de la dirección exacta y de una variante de formato; mide si la propia parcela es recuperada |

### Operadores de variantes (positivos, sección C)

`lowercase`, `extra_whitespace`, `accent_marks`, `glued`, `glued_type`, `hash_tight`, `hash_no_dot`, `hash_ordinal`, `plate_zero_pad`, `alias_long`, `alias_dotted`, `complement_alias`, `complement_zero_pad`, `complement_reorder`.

### Operadores de mutación (negativos, sección D)

| Familia | Mutaciones | Resultado esperado |
|---|---|---|
| Número de vía | `via_number_plus1`, `via_number_minus1` | no coincide |
| Tipo de vía | `via_type_change` | no coincide |
| Número de cruce | `cross_number_plus1`, `cross_number_minus1` | no coincide |
| Placa | `plate_plus1`, `plate_minus1`, `plate_plus2`, `plate_minus2` | no coincide |
| Letra de vía y cruce | `via_letter_add`, `via_letter_drop`, `via_letter_change`, `cross_letter_add`, `cross_letter_drop`, `cross_letter_change` | no coincide |
| BIS | `via_bis_add`, `via_bis_remove`, `cross_bis_add`, `cross_bis_remove` | no coincide |
| Cuadrante | `via_quadrant_add`, `via_quadrant_remove`, `cross_quadrant_add`, `cross_quadrant_remove` | no coincide |
| Complemento | `complement_value_change`, `complement_removed_opaque` | no coincide |
| Complemento (diseño) | `complement_added`, `complement_removed` | **coincide** (0,98 por diseño) |

Criterio de aceptación de la corrida (configuración del proyecto, no un resultado): cobertura de lectura mayor o igual a 99,0 %; cero violaciones de autocoincidencia y simetría; recall de variantes mayor o igual a 99,99 %; ninguna diferencia real con puntaje mayor o igual a 0,90; cero falsos vínculos con placa distinta; recuperación de la propia parcela en el 100 % de la muestra consultada.

Advertencia metodológica: las secciones C y D son **circulares**. Los positivos se generan reescribiendo direcciones con las mismas reglas que el parser conoce, y los negativos son mutaciones de un campo que los vetos rechazan por construcción. Sirven como pruebas de regresión e invariantes, no como estimación de precisión y recall reales (ver [08](08-metricas-y-confiabilidad.md)).

### Cómo se ejecuta

```bash
python tools/eval_catastro.py --catastro <ruta>/catastro.parquet --json-out resultado.json
# o con variable de entorno
CATASTRO_PARQUET=<ruta>/catastro.parquet python tools/eval_catastro.py
# muestra distinta, o todas las filas (lento)
python tools/eval_catastro.py --catastro <ruta>/catastro.parquet --sample 50000 --seed 7
python tools/eval_catastro.py --catastro <ruta>/catastro.parquet --full
```

## 3. Verdad de referencia: build_groundtruth y eval_groundtruth

Objetivo: medir con pares cuya verdad **no** proviene del propio comparador. La unidad de análisis primaria es el edificio o predio (la puerta), no la unidad interior.

`build_groundtruth.py` genera:

| Producto | Contenido |
|---|---|
| Positivos sintéticos (250 por defecto) | Una dirección real del catastro y una variante de escritura producida con reglas del constructor. Transformaciones: `type_alias_long`, `type_alias_short`, `complement_alias`, `zero_padded`, `glued_tokens`, `lowercase`, `extra_whitespace`, `hash_to_no`, `hash_absent`, `punctuation`, `unicode_spacing`, `unicode_symbols`, `trailing_whitespace`, `combo`, y las "novedosas" `novel_type_alias`, `novel_complement_alias`, `novel_number_word` (grafías que no estaban en las tablas al diseñar el conjunto) |
| Candidatos a etiquetar (60 por estrato) | Pares difíciles en 6 estratos: `complement_one_sided`, `cross_type_presence`, `plate_letter`, `adjacent_or_swap`, `same_base_diff_complement`, `opaque_or_degenerate`. Sin puntaje, para no anclar a quien etiqueta |
| Negativos de control (30) | Pares aleatorios de comunas distintas (prueba de humo) |
| Hoja de etiquetado | Archivo CSV con las columnas `pair_id`, `addr_a`, `addr_b`, ..., `same_door`, `same_unit`, `evidence_source`, `notes`; filas mezcladas con semilla |
| Manifiesto | Semilla, conteos por estrato y transformación, faltantes de cuota, rangos de puntaje, hash SHA-256 de cada archivo |
| Archivo lateral privado | `pair_id`, `score`, `decision` por par; se guarda en un directorio hermano para no entregarlo a quien etiqueta |

Protocolo de etiquetado (resumen de `../docs/groundtruth.md`): dos columnas independientes con valores `yes`, `no` o `unsure`. `same_door` (principal) indica si ambas direcciones son el mismo edificio o predio (misma vía y misma placa, ignorando unidad). `same_unit` (secundaria) solo se pregunta si `same_door=yes`. Las fuentes de evidencia aceptables dependen del estrato: para letras de placa, placas vecinas y complementos se necesita el visor catastral o evidencia de campo, porque los mapas comerciales interpolan y no distinguen esos casos.

`eval_groundtruth.py` puntúa los pares con el comparador vivo (o con el archivo lateral congelado) y reporta cobertura, métricas de `same_door`, desglose por estrato y por transformación, recall sintético separado en vocabulario conocido y `novel_*`, sección de `same_unit` y principales falsos positivos y negativos. Las razones de lectura honesta: la muestra de candidatos está estratificada y sobremuestreada, por lo que la fila humana da tasas condicionales por estrato y **no** la precisión poblacional.

Mientras la hoja de etiquetas humanas no se diligencie, no existen precisión ni recall con etiquetas humanas (ver [08](08-metricas-y-confiabilidad.md)).

## 4. Evaluación con texto real: eval_realworld.py

Compara texto real escrito por personas contra la dirección del catastro de la parcela etiquetada, contra parcelas vecinas distintas y contra el catastro completo mediante `AddressIndex`.

Características del protocolo:

- Etiquetas derivadas de una consulta GPS de punto en polígono, **ruidosas** (el acierto con frecuencia es la parcela vecina, a unos 6 m).
- Filas `gold`: la vía, el cruce y la placa del texto coinciden con la dirección propia de la parcela etiquetada. Filas `noisy`: solo se muestran distribuciones. El titular se limita a filas `gold`.
- `--input raw` usa el texto sucio (prueba de estrés); `--input normalized` usa una columna ya normalizada (el escenario objetivo del modelo).
- Se reportan siempre los niveles `unit` y `building`; el titular elige uno con `--level`.
- Negativos difíciles: placa más o menos 2, misma placa con otra letra, misma placa con otro cruce; se excluyen los de la misma parcela (primeros 21 caracteres del número predial) y los homónimos.
- Intervalos de Wilson al 95 %.
- Guardia contra el uso indebido del conjunto reservado: cualquier ruta que contenga `frozen` se rechaza salvo que se pase `--final`, con una advertencia de que ese conjunto queda "quemado" tras un uso.
- El lector de parquet admite solo una lista de columnas permitidas y rechaza columnas de predicción, puntaje o coincidencia.

## 5. Esquemas de datos requeridos

Los tipos son los esperados por el código; no se describe aquí cómo obtener datos privados.

### 5.1 Catastro para eval_catastro.py

| Columna | Tipo | Obligatoria | Descripción |
|---|---|---|---|
| `direccion` (configurable con `--address-col`) | texto (UTF-8) | Sí | Dirección del predio, idealmente ya en forma canónica |
| `numero_predial_nacional` (configurable con `--parcel-col`) | texto de dígitos | No | Identificador de parcela; sin esta columna, la sección E trata cada fila como un registro distinto y se imprime una nota |

### 5.2 Catastro para build_groundtruth.py

| Columna | Tipo | Obligatoria | Descripción |
|---|---|---|---|
| `direccion` | texto | Sí | Dirección |
| `numero_predial_nacional` | texto | Sí | Número predial (de 30 posiciones en el estándar vigente) |
| `manzana` | texto | Sí | Código de manzana |
| `comuna` | texto | Sí | Código o nombre de comuna |
| `barrio` | texto | Sí | Código o nombre de barrio |

### 5.3 Etiquetas humanas para eval_groundtruth.py

CSV o JSONL con al menos `pair_id` (texto) y `same_door` (`yes`/`no`/`unsure`, sin importar mayúsculas; en JSONL también `true`/`false`). Opcional `same_unit` (`yes`/`no`/`unsure`/`n/a`). Se admiten BOM, CRLF y columnas adicionales (se ignoran). La columna antigua `label` se rechaza. `pair_id` duplicado es un error; ids desconocidos generan advertencia.

### 5.4 Conjuntos para eval_realworld.py

| Archivo | Columnas usadas | Descripción |
|---|---|---|
| Catastro | `doc_id` (entero), `direccion` (texto), `numero_predial_nacional` (texto; solo se conservan los primeros 21 caracteres y nunca se escribe en salidas) | Registros de referencia |
| `--dev` | `raw_address` o una columna normalizada (`rule_canonical`, `canonical_address`, `their_normalized`), `dataset` (texto), `gt_doc_id` (entero, clave a `doc_id`), `gt_quality` (`gold`, `noisy` o `no_gt`) | Conjunto de desarrollo etiquetado |
| `--train-pairs` | `raw_address`, `dataset`, `gt_doc_id` | Conjunto etiquetado donde toda fila se trata como `gold`; solo con `--input raw` |

## 6. Cómo repetirlo en su ciudad

1. Verifique que sus direcciones ya estén en la forma canónica del modelo. Si siguen el estándar nacional sin `#` ni `-`, adáptelas primero (ver [10](10-estandar-y-generalizacion.md)).
2. Cree un entorno con `pyarrow`: `python -m venv eval-venv` e instale `pip install -r requirements-dev.txt` (`pyarrow==25.0.1`).
3. Ejecute `eval_catastro.py` sobre su catastro (esquema 5.1). Lea primero la sección A: la cobertura de lectura y las "causas de fallo" indican qué escrituras locales faltan en `VIA_TYPES`, `COMPLEMENT_KINDS` y la gramática.
4. Extienda las tablas (ver [10](10-estandar-y-generalizacion.md)) y repita hasta que las secciones A, B, C y D alcancen los criterios de aceptación. Recuerde que C y D son circulares.
5. Con el esquema 5.2, ejecute `build_groundtruth.py` y entregue **solo** la hoja de etiquetado a personas expertas; nunca el archivo de candidatos con estratos ni el archivo lateral de puntajes.
6. Reúna evidencia según el estrato (visor catastral o campo) y evalúe con `eval_groundtruth.py --labels <hoja_diligenciada>`.
7. Si dispone de texto real con etiquetas, use `eval_realworld.py` con `--input normalized`.
8. Decida el umbral y la tolerancia de placa con la tabla de [03](03-configuracion.md), sección 7, **reproducida con sus datos**. No asuma que el hueco entre 0,85 y 0,9126 implique que 0,90 sea óptimo en su territorio: eso depende de los datos, no del hueco.
9. Registre el resultado con intervalos de confianza y las limitaciones, como en [08](08-metricas-y-confiabilidad.md).

Calibración del puntaje como probabilidad: no está implementada. Si se necesita, requiere etiquetas humanas en una muestra representativa y un ajuste posterior (por ejemplo, regresión isotónica o por cubetas de puntaje) que no forma parte de este proyecto.

## 7. Semillas, tiempos y determinismo

| Elemento | Valor | Fuente |
|---|---|---|
| Semilla de `eval_catastro.py` | 42 (`--seed`) | código |
| Semilla de `build_groundtruth.py` | 20260926 (`--seed`) | código y manifiesto |
| Semilla de `eval_realworld.py` | 42 | código |
| Determinismo | Mismo `--seed` produce salidas idénticas (salvo tiempos); `build_groundtruth` produce archivos idénticos byte a byte | `../docs/groundtruth.md` (protocolo) |

El comparador en sí no usa números aleatorios.

## 8. Privacidad de las salidas

- `eval_catastro.py` imprime hasta 5 direcciones de ejemplo por categoría (`MAX_EXAMPLES`) además de agregados. Si el catastro contiene direcciones reales, esos ejemplos son datos reales: no publique el informe sin anonimizarlo.
- `build_groundtruth.py` escribe números prediales y direcciones reales en sus archivos de candidatos y hoja de etiquetado. Trate esos archivos como privados.
- `eval_realworld.py` escribe sus salidas por defecto en `data/realworld_private/` (que debe estar fuera del control de versiones) y limita stdout y `--json-out` a agregados.
