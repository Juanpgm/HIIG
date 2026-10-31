# 14. Reentrenamiento 2026-10: especificación técnica y veredicto de confiabilidad

Fecha del documento: 2026-10-31.

> **Veredicto programático: NO APRUEBA.** Según las reglas de aceptación escritas antes de entrenar, 6 de 12 criterios se aprueban (A1, A3, A5, A7, A8, A11) y 6 no se aprueban (A2, A4, A6, A9, A10, A12). Ningún criterio quedó sin evaluar.
>
> **Promoción a producción: pendiente de aprobación explícita del responsable del proyecto.** Este documento no autoriza la promoción. Los artefactos publicados en este repositorio (`artifacts/model.pt`, `tuning.json`, `reliability.json`) no cambian, y el servicio en producción sigue sirviendo el modelo anterior. La carpeta [`normalizador_produccion/`](../../normalizador_produccion/README.md) contiene la **imagen candidata** (herramienta de despliegue de nivel producción con los pesos del candidato), pendiente de aprobación del titular; no es el servicio en producción.

Este documento describe el reentrenamiento del normalizador con un índice construido solo desde la nomenclatura domiciliaria oficial. Está dirigido a quien decide si el modelo reentrenado es al menos tan confiable como el de producción. Contiene solo agregados (conteos, tasas y valores de métricas). No incluye direcciones, identificadores de predio, filas de datos ni datos de entrenamiento.

> **Código del proyecto de origen, aún no publicado en este espejo.** Los elementos marcados con † en este documento (scripts y módulos como `build_index.py`†, `index_union.py`†, `calibration_report.py`†, `final_evidence.py`†, `retrain_compare.py`†, `jev_check.py`†, `fit_decision.py`†, `index_fallback.py`†, `compare_runs.py`†, `write_retrain_report.py`†, `monitor_signals.py`†, `download_nomenclatura.py`†, `clean_nomenclatura.py`†, `reproject_nomenclatura.py`†, `export_onnx.py`†, y los documentos `docs/onnx.md`† y `docs/retrain_2026-10_*.md`) pertenecen al proyecto de origen y **no están en `normalizador_direcciones/` de este repositorio**; no se pueden ejecutar desde aquí. `decision.py`†, `output_schema.py`†, `onnx_backend.py`† y `tokenizer.py`† solo se publican como copia de servicio en [`normalizador_produccion/src/cali_address/`](../../normalizador_produccion/src/cali_address/), no en `normalizador_direcciones/src/`. Las cifras y los comportamientos descritos se tomaron del proyecto de origen y no se pueden reproducir con solo el contenido de este espejo.

## Contenido

1. [Resumen](#1-resumen)
2. [Alcance, fuentes y decisiones](#2-alcance-fuentes-y-decisiones)
3. [Arquitectura](#3-arquitectura)
4. [Flujo de datos](#4-flujo-de-datos)
5. [Composición del índice](#5-composición-del-índice)
6. [Entrenamiento](#6-entrenamiento)
7. [Capa de decisión](#7-capa-de-decisión)
8. [Validación de salidas](#8-validación-de-salidas)
9. [Backend ONNX y benchmarks](#9-backend-onnx-y-benchmarks)
10. [Veredicto de confiabilidad](#10-veredicto-de-confiabilidad)
11. [Respaldo de catastro (reversible)](#11-respaldo-de-catastro-reversible)
12. [Limitaciones](#12-limitaciones)
13. [Reproducibilidad y guardas](#13-reproducibilidad-y-guardas)
14. [Documentos relacionados](#14-documentos-relacionados)

## 1. Resumen

- **Candidato evaluado:** corrida `v2_final_catfb`, que usa el modelo de la semilla 42 sobre el índice `index_v2g_cert_pcer_catfb_r3` (401 860 documentos), con la calibración base de la ronda 1 (variante `basecal`).
- **Línea base medida:** `baseline_prod_2026_10`, la misma métrica sobre el modelo de producción con el código vigente.
- **Lo que mejora:** más filas respondidas (dev gold 283 frente a 278; prueba congelada con todas las filas 1 090 frente a 1 002), precisión de lote gold en la prueba congelada (0.9938 frente a 0.9615), y calibración fuera de muestra muy por debajo del límite de A6.
- **Lo que empeora o no se demuestra:** precisión de lote en dev gold (0.9881 frente a 0.9960), número de filas con pérdida de aciertos en dev (121 frente a 108), precisión de manzana en la prueba congelada con todas las filas (0.7121 frente a 0.7314), y la precisión exacta por NPN, que cae de forma marcada porque el candidato emite NPN con sufijo de unidad (sección 10.4).
- **Cobertura:** el índice nuevo solo usa fuentes nuevas. La pérdida titular de cobertura de lotes gold frente a producción es de 1.42 puntos porcentuales. La pérdida inherente a la fuente de nomenclatura (lotes que ningún documento de nomenclatura contiene) es de 2.83 puntos; el respaldo de catastro (sección 11) recupera la diferencia entre ambas.
- **Decisiones abiertas:** cuatro decisiones de criterio del responsable del proyecto (sección 10.3). Ninguna se tomó automáticamente.

## 2. Alcance, fuentes y decisiones

Las decisiones que determinan el significado de las cifras son las siguientes. Las fechas corresponden a cada decisión.

- **2026-10-30, fuente principal.** La capa WFS de la infraestructura de datos espaciales de Cali (IDESC) es la fuente primaria: registros certificados (`cert`) y pendientes de certificación (`pcer`). La copia del shapefile solo sirve como referencia de validación.
- **2026-10-30, coordenadas.** Todas las coordenadas del índice se obtienen de reproyectar `ndcoordx` / `ndcoordy` de EPSG:6249 a EPSG:4326 con el archivo `.prj` del shapefile. Las lon/lat del WFS se conservan solo para auditoría. Las compuertas de comparación se ajustaron a nivel de lote para el NPN (concordancia mayor o igual a 98 %) y a mediana menor o igual a 1 m y p99 menor o igual a 2 m para coordenadas.
- **2026-10-31, solo fuentes nuevas.** El índice se construye únicamente con registros de nomenclatura. El catastro es de solo lectura: se usa para auditoría y para la compuerta del oráculo, nunca como fuente de documentos, NPN, `n_predios`, manzana o centroides. Las direcciones catastrales sin registro de nomenclatura se pierden por construcción y se cuantifican en la sección 5.
- **2026-10-31, semántica predial a nivel de lote.** `gt_predial` es catastral y a nivel de lote, mientras que un NPN de nomenclatura puede llevar un sufijo de unidad. Por eso los criterios A2 y A3 comparan `NPN[:21]` (lote), y la precisión exacta por NPN se reporta junto a ella sin ser criterio.
- **2026-10-31, pre-normalización.** La separación de letras (`pre_normalize_separates_letters`) está desactivada por defecto para que la producción y la línea base no cambien. Cada índice nuevo la activa mediante `index_meta.json`.
- **Validación de salidas.** `validate_outputs` (`off`, `report`, `correct`) está desactivada por defecto. Sus efectos se miden por separado (sección 8).

**Enmiendas a los criterios.** Los criterios de aceptación se escribieron el 2026-10-30 a las 22:40, antes de entrenar. Se enmendaron dos veces el 2026-10-31: primero para A2, A3 y A7 (medición a nivel de lote), y después, tras re-puntuar la línea base, para A6, A10 y la regla de selección. Todas las enmiendas se hicieron antes de entrenar o puntuar cualquier candidato. Aun así, la línea base se re-puntuó antes de la segunda ronda de enmiendas, y el lector debe tenerlo en cuenta al juzgar la neutralidad de los criterios.

## 3. Arquitectura

### 3.1 Modelo

- Codificador dual a nivel de carácter, tipo Transformer: 4 capas, `d_model` 256, 8 cabezas, salida de 256 dimensiones, longitud máxima 40 y 3 375 601 parámetros.
- Las cadenas de consulta y los documentos del catastro se proyectan al mismo espacio y se comparan por similitud coseno.
- El entrenamiento usa solo una receta de producción desde cero: `mode=mixed`, `AddressEncoder` nuevo, con semilla. Las cadenas oficiales son consultas de entrenamiento (`q_source=official`) y las sintéticas se generan con operadores de corrupción.

### 3.2 Servicio

El servicio de normalización (`normalize_strict`) combina varias etapas, en este orden:

1. **Análisis de texto** (`parser`): descompone la dirección en vía, cruce, placa, cuadrante, complemento y tokens de unidad.
2. **Recuperación**: búsqueda coseno top-k sobre la matriz de documentos del índice. Las dos vistas de la consulta (texto canónico y pre-normalizado) se buscan en una sola pasada.
3. **Reglas estructurales y guarda geográfica**: el cuadrante se compara por posición; una placa distinta nunca es puerta verificada; un sufijo unilateral se rechaza. El candidato pasa por una compuerta geográfica.
4. **Fidelidad y estado de catálogo**: mide si la dirección devuelta es otra que la recibida (`fidelity_risk`) y asigna `catalog_status` y `estado`.
5. **Niveles de posición**: `predio`, `lote`, `tramo`, `manzana`, `via` o `none`, con radios geométricos derivados del catastro.
6. **Capa de decisión calibrada** (sección 7): calcula la confianza de emisión, las bandas y los motivos.
7. **Validación de salidas** (sección 8), opcional.

La fusión de puntuación del candidato usa pesos 0.55 / 0.15 / 0.15 / 0.15 con umbral 0.83. Producción usa 0.70 / 0.10 / 0.10 / 0.10 con umbral 0.73 (valores de `calib_design/SPEC.md`, sección 1).

### 3.3 Backends de inferencia

| Backend | Variable | Notas |
|---|---|---|
| `torch` | `NORMALIZER_BACKEND=torch` (defecto) | Referencia. Único backend que puede regenerar `catastro_emb.pt`. |
| `onnx` | `NORMALIZER_BACKEND=onnx` | Un único grafo `ids -> emb, xy` (opset 17). Recomendado si se adopta ONNX (sección 9). |
| `onnx-int8` | `NORMALIZER_BACKEND=onnx-int8` | Cuantización dinámica de pesos. Opcional; no invariante al lote. |

## 4. Flujo de datos

Reentrenamiento, en orden:

1. **Reproyección del shapefile** de referencia (EPSG:6249 a EPSG:4326), con compuertas de caja envolvente y de cordura de distancia.
2. **Descarga del WFS** de nomenclatura (paginada y reanudable), con validación contra el shapefile.
3. **Limpieza** (`clean_nomenclatura`†): reglas por conteo, cuarentena con motivo y compuerta de cuarentena menor a 0.5 %.
4. **Construcción del índice** (`build_index`†) por niveles (`cert`, `cert,pcer`), con contabilidad de registros (cada registro cae en exactamente un grupo) y compuerta de techo de lote por oráculo.
5. **Respaldo de catastro** (opcional, reversible; sección 11) en un directorio nuevo.
6. **Entrenamiento** sobre el índice (o reutilización de un modelo entrenado sobre un índice de texto y orden idénticos).
7. **Calibración** de la capa de decisión con pares reales de entrenamiento (`train_pairs`), que nunca se usaron para entrenar el modelo, con validación cruzada por grupos de dirección.
8. **Evaluación** en dev (completo y gold) y, una sola vez con la bandera `--final`, en la prueba congelada.
9. **Comparación** contra la línea base medida y evaluación de criterios de aceptación.
10. **Auditoría JEV** (opcional, solo con filas públicas o sintéticas).
11. **Informe agregado** en Markdown, con veredicto calculado.

## 5. Composición del índice

### 5.1 Fuentes

| Fuente | Filas | Papel |
|---|---:|---|
| WFS de nomenclatura (primaria) | 692 389 | cert 437 945; pcer 253 627; lote 808; prov 9 |
| Shapefile de nomenclatura (certificada) | 437 825 | Solo referencia de validación; EPSG:6249 |
| Catastro de producción | 330 387 | Solo lectura: auditoría y compuerta del oráculo |

Descarga: 139 páginas, 692 389 coincidencias, 692 389 descargadas, estado correcto. Corte de la capa: 2026-09-03Z. El shapefile (2026-10-30) puede ser más reciente que el corte del WFS; la compuerta a nivel de lote absorbe las diferencias de sufijo de unidad.

### 5.2 Limpieza

| Concepto | Valor |
|---|---:|
| Filas de entrada | 692 389 |
| Filas limpias | 691 263 |
| Filas en cuarentena | 1 126 |
| Proporción en cuarentena (límite 0.0050) | 0.0016 (aprueba) |
| Coordenadas con respaldo del WFS (ceros en proyectadas) | 7 541 |
| Filas con NPN inválido o de longitud atípica | 2 + 1 + 1 |
| Filas con estado `status_invalid` | 817 |

Estados excluidos del índice: `lote` (808) y `prov` (9).

Compuertas WFS frente a shapefile (todas aprueban): proporción de diferencia de conteo certificado 0.00027 (menor o igual a 0.005); concordancia de direcciones 0.99998 (mayor o igual a 0.995); concordancia NPN a nivel de lote 0.9896 (mayor o igual a 0.98); mediana de coordenadas 0.721 m (menor o igual a 1.0); p99 de coordenadas 1.376 m (menor o igual a 2.0). La concordancia del NPN completo de 30 dígitos (0.7003) es informativa: la diferencia es el sufijo de unidad (los dos últimos dígitos).

### 5.3 Índices construidos

| Índice | Niveles | Documentos |
|---|---|---:|
| `index_v2a_cert` | cert, catastro | 343 136 |
| `index_v2b_cert_pcer` | cert, pcer, catastro | 384 157 |
| `index_v2c_cert_r2` | cert | 243 468 |
| `index_v2d_cert_pcer_r2` | cert, pcer | 394 919 |
| `index_v2e_cert_pcer_r3` | cert, pcer | 394 919 |
| `index_v2f_cert_r3` | cert | 243 468 |
| `index_v2g_cert_pcer_catfb_r3` (candidato) | cert, pcer, catastro (respaldo) | 401 860 |

Composición del candidato `index_v2g_cert_pcer_catfb_r3`:

| Fuente de documento | Documentos |
|---|---:|
| `nomenclatura_cert` | 243 468 |
| `nomenclatura_pcer` | 151 451 |
| `catastro` (respaldo, sección 11) | 6 941 (1.7 % del índice; cálculo propio) |

Filas de unidades (sidecar, no documentos): 282 831.

Contabilidad de registros del candidato (suma exacta de la entrada limpia, 691 263):

| Grupo | Filas |
|---|---:|
| `base_doc_new` | 365 567 |
| `base_doc_merged` | 31 695 |
| `unit` | 282 831 |
| `fallback_doc` | 1 107 |
| `dropped` | 0 |
| `excluded_tier` | 0 |
| `incomplete_address` | 10 024 |
| `spread_outlier` | 39 |
| **Total** | **691 263** |

Techo del oráculo (lotes gold): la pérdida de cobertura titular del candidato es de 1.42 puntos porcentuales, y la pérdida inherente a la fuente (lote ausente de la propia nomenclatura) es de 2.83 puntos; el respaldo de catastro explica la diferencia. La compuerta de construcción no se incumple (caída máxima permitida: 2.0 puntos).

## 6. Entrenamiento

| Corrida | Semilla | Épocas | Recall@1 validación, oficial | Recall@1 validación, sintético | Segundos |
|---|---:|---|---:|---:|---:|
| `v2e_cert_pcer_s42_r3_cal` (y `v2g_..._basecal`, el mismo modelo) | 42 | 8/8 | 0.9967 | 0.8806 | 2 298.4 |
| `v2e_cert_pcer_s43_cal` | 43 | 8/8 | 0.9964 | 0.8849 | 2 355.3 |

- Receta: batch 2048, lr 0.0030, hasta 8 épocas y 120 minutos máximos. Hardware: GPU NVIDIA GeForce RTX 5070 Ti (capacidad de cómputo 12.0). Versiones de torch y CUDA no registradas en las fuentes (`n/d`).
- **El modelo de la semilla 42 no se reentrenó para el candidato final.** Se entrenó sobre el índice `index_v2d_cert_pcer_r2`, cuyo texto y orden de documentos son los del índice r3 (`index_v2e`), y se reutilizó. El respaldo de catastro (sección 11) tampoco requiere reentrenar.
- La semilla 43 (entrenamiento completo sobre r3) se usa para medir la banda de ruido entre semillas.
- El recall de validación sintético no es comparable entre índices (documentos y mezcla de corrupción distintos). Ninguno de los dos recalls sirve como selector.

**Regla de selección.** Entre las corridas cuya cobertura OK en dev gold es al menos la de la línea base, se elige la de mayor precisión de lote en dev gold; los empates (dentro de una fila gold) se desempatan por la precisión de manzana en todas las filas dev. La prueba congelada no interviene en la selección.

## 7. Capa de decisión

La capa de decisión convierte la confianza de la búsqueda en una decisión de emisión y en bandas de lectura.

- **Confianza**: `confianza` es la puntuación fusionada del mejor candidato que pasa la compuerta geográfica, acotada a [0, 1]. Es la única entrada probabilística de la decisión. `confianza` no es una probabilidad (ver [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md)).
- **Parámetros** (`decision.json`, versión `decision-calib-v1-cand-20261031`): umbral de predio `tau_p` 0.94; umbral de manzana `tau_m` 0.70; piso de confianza 0.70; bandas `alta` mayor o igual a 0.97 y `media` mayor o igual a 0.94; `baja` por debajo de 0.94. Reglas blandas: `letra_via`, `letra_cruce_cardinal`, `bis_via`, `bis_cruce`, `cuadrante_compl`; `gate_fallback` activo.
- **Ajuste**: 5 087 pares de predio (4 882 positivos) y 5 014 pares de manzana (4 954 positivos) de `train_pairs`, con validación cruzada de 5 pliegues agrupados por clave de dirección.
- **Motivos**: cada fila lleva un texto de motivo que explica por qué se emitió o se retuvo el predial. El vocabulario incluye, entre otros, `direccion compartida por`, `confiabilidad del predio baja`, `confianza`, `posicion maxima`, `letra de cruce`, `fuera de` y `base distinta al`.

**Bandas medidas en dev (lote y manzana con intervalo de Wilson al 95 %).**

| Conjunto | Banda | Filas | Precisión de lote | Precisión de manzana |
|---|---|---:|---|---|
| dev gold | alta | 224 | 0.9955 [0.9751 a 0.9992] (223/224) | 0.9911 [0.9680 a 0.9975] |
| dev gold | media | 28 | 0.9286 [0.7735 a 0.9802] (26/28) | 1.0000 [0.8794 a 1.0000] |
| dev gold | baja | 31 | n/d (sin filas con predial evaluado) | 1.0000 [0.8794 a 1.0000] (28/28) |
| dev completo | alta | 823 | 0.6683 [0.6212 a 0.7123] (272/407) | 0.8600 [0.8229 a 0.8903] |
| dev completo | media | 133 | 0.6271 [0.4995 a 0.7392] (37/59) | 0.8136 [0.6962 a 0.8926] |
| dev completo | baja | 275 | n/d (sin filas evaluadas) | 0.9041 [0.8150 a 0.9528] (66/73) |

La banda alta tiene precisión de lote de 0.67 en dev completo y de 0.996 en dev gold. Esa diferencia es la razón por la que la calibración no basta por sí sola (sección 10.5).

## 8. Validación de salidas

`validate_outputs` valida cada fila con pydantic v2 (`NormalizedRecord`) y cuenta violaciones por regla.

- `off` (defecto): la salida es idéntica a la de antes.
- `report`: cuenta las violaciones sin tocar la salida.
- `correct`: aplica solo correcciones conservadoras (nunca más optimistas).

Resultados: la línea base registra 1 violación (`ok_without_support`) en 2 078 filas de dev completo en modo `report`. El candidato registra 0 violaciones en ambos conjuntos de dev. En modo `correct`, las métricas de dev no cambian (A12 falla por herencia de A2, A4 y A10, no por la corrección). La prueba congelada no se vuelve a puntuar con `correct` por diseño.

## 9. Backend ONNX y benchmarks

### 9.1 Motivo y paridad

El backend ONNX reduce la dependencia de torch en la imagen de servicio. El backend por defecto sigue siendo `torch`.

Paridad frente a torch, 2 000 cadenas públicas:

| Modelo | Backend | Coseno mínimo | Top-1 idéntico | Top-1 distinto | Diferencia máxima de coordenada |
|---|---|---:|---:|---:|---:|
| Producción (330 387 docs) | onnx fp32 | 1.000000 | 100 % | 0 | 0.014 m |
| Producción | onnx-int8 | 0.999618 | 100 % | 0 | 138.8 m |
| Candidato (401 860 docs) | onnx fp32 | 0.99999988 | 99.95 % | 1 | 0.007 m |
| Candidato | onnx-int8 | 0.999500 | 100 % | 0 | 231.8 m |

En el candidato, el único desacuerdo fp32 es un empate entre filas duplicadas con embedding idéntico (la misma cadena de dirección), no un error de incrustación. El int8 no es invariante al lote: la cuantización dinámica calcula la escala sobre todo el lote, así que el resultado de una fila depende de las demás.

Evaluación estricta en dev (CPU, índice de producción): ONNX fp32 reproduce a torch en todas las filas (2 078 de dev y 289 de gold). ONNX int8 cambia 1 predial de 2 078 en dev.

### 9.2 Benchmark

Equipo: AMD Ryzen 9 9950X (16 núcleos físicos), Windows 11, Python 3.14. Pasada final en silencio, con 5 repeticiones. Latencia en milisegundos de una dirección (p50 y p95); rendimiento en filas por segundo.

**16 hilos:**

| Artefactos | Backend | Arranque en frío (s) | p50 (ms) | p95 (ms) | 100 filas/s | 2 000 filas/s | 10 000 filas/s | RSS pico (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Producción | torch | 0.90 | 19.9 | 22.2 | 136 | 142 | 137 | 1 568 |
| Producción | onnx | 1.08 | 17.9 | 19.9 | 209 | 216 | 215 | 1 675 |
| Producción | onnx-int8 | 1.07 | 17.4 | 19.4 | 215 | 222 | 221 | 1 779 |
| Candidato | torch | 1.09 | 23.5 | 25.9 | 116 | 120 | 120 | 1 827 |
| Candidato | onnx | 1.28 | 21.5 | 23.8 | 179 | 182 | 184 | 1 962 |
| Candidato | onnx-int8 | 1.28 | 20.9 | 23.3 | 186 | 190 | 190 | 2 077 |

**4 hilos:**

| Artefactos | Backend | Arranque en frío (s) | p50 (ms) | p95 (ms) | 100 filas/s | 2 000 filas/s | 10 000 filas/s | RSS pico (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Producción | torch | 0.88 | 29.3 | 31.9 | 46 | 49 | 49 | 1 542 |
| Producción | onnx | 1.07 | 21.3 | 23.5 | 187 | 195 | 192 | 1 660 |
| Producción | onnx-int8 | 1.05 | 19.5 | 21.7 | 208 | 211 | 209 | 1 758 |
| Candidato | torch | 1.10 | 35.1 | 38.3 | 41 | 40 | 40 | 1 788 |
| Candidato | onnx | 1.29 | 25.8 | 28.7 | 155 | 155 | 155 | 1 926 |
| Candidato | onnx-int8 | 1.29 | 24.9 | 27.3 | 175 | 176 | 176 | 2 040 |

Lectura: ONNX fp32 rinde unas 1.5 veces lo de torch con 16 hilos y entre 3.8 y 4.1 veces con 4 hilos. int8 aporta entre 3 % y 14 % de rendimiento adicional sobre fp32 y cuesta unos 100 MB más de RSS pico. ONNX no reduce el arranque en frío ni la memoria pico frente a torch. El tamaño instalado es 517 MB para torch y 46 MB para onnxruntime; la imagen sin torch se estimó en unos 1.45 GB frente a 2.33 GB; con `docker build` se midieron **744 MB en disco y 187 MB comprimida** (ver [../../normalizador_produccion/README.md](../../normalizador_produccion/README.md)).

**Anomalía de índice fp16 (medido; causa no verificada).** Con `NORMALIZER_ONNX_INDEX=fp16`, la latencia p50 de una dirección sube a 181.7 ms (p95 197.6 ms), frente a 21.5 ms (p95 23.8 ms) con fp32, en el mismo candidato, 16 hilos y ONNX fp32. El pico de RSS baja de 1 962 a 1 743 MB. El documento ONNX atribuye la diferencia a la conversión fp16 a fp32 en cada búsqueda; esa hipótesis no se verificó en la pasada final y no se midió por separado. Conclusión operativa: fp16 solo conviene si la memoria es más importante que la latencia.

### 9.3 Recomendación

- Si se adopta ONNX, usar `onnx` fp32 por defecto.
- `onnx-int8` es opcional y requiere reevaluar sobre dev completo antes de adoptarlo, por la dependencia del lote y el cambio de un predial.
- `fp16` solo si la memoria manda.

Limitaciones de la medición: una sola máquina; consultas sintéticas (cadenas públicas corrompidas), no tráfico real; las diferencias entre candidato y producción mezclan índice y capa de decisión; no se midió RSS de contenedor.

## 10. Veredicto de confiabilidad

### 10.1 Criterios A1 a A12

| ID | Estado | Criterio (resumen) | Evidencia |
|---|---|---|---|
| A1 | **APRUEBA** | Dev gold OK al menos la línea base menos 1 | 283 frente a un piso de 277 |
| A2 | **NO APRUEBA** | Precisión de lote en dev gold al menos la línea base | 0.9881 frente a 0.9960 |
| A3 | **APRUEBA** | Prueba congelada gold OK y precisión de lote dentro del margen de 2 filas | OK 399 frente a piso 398; precisión de lote 0.9938 frente a piso 0.9565 |
| A4 | **NO APRUEBA** | Pérdida de aciertos en dev menor o igual a la banda de ruido | 121 frente a límite 112 |
| A5 | **APRUEBA** | Precisión de manzana en dev al menos la línea base menos 0.005 | 0.8609 frente a piso 0.8580 |
| A6 | **NO APRUEBA** | Calibración refitada: ECE de ambas cabezas hasta 0.01 sobre la línea base, y regla de envío frente al refit | ECE dentro de límites; regla de envío falla (sección 10.2) |
| A7 | **APRUEBA** | Tasa de fallo del parser menor al 1 %, compuertas CRS, construcción y cobertura, contabilidad sin pérdidas | Fallo 0.0016; construcción 1.0044 frente a piso 0.9771; pérdida de cobertura 0.0142 frente a límite 0.0383 (sección 10.2) |
| A8 | **APRUEBA** | Producción sin modificar en cada corrida | Hashes de producción iguales antes y después |
| A9 | **NO APRUEBA** | Pruebas completas en verde y escáner de direcciones reales sin hallazgos | 6 533 pruebas pasan y 1 falla; escáner sale con código 0 |
| A10 | **NO APRUEBA** | A confianza mayor o igual a 0.7: cobertura, precisión de lote y error de manzana frente a la línea base | Cobertura 0.9792 frente a piso 0.9519 (cumple); precisión de lote 0.9881 frente a piso 0.9910 (no cumple); error de manzana 0.0071 frente a límite 0.0086 (cumple) |
| A11 | **APRUEBA** | Cuarentena menor al 0.5 %; scripts de construcción y descarga salen con código 0 | Cuarentena 0.0016 |
| A12 | **NO APRUEBA** | Con `validate_outputs=correct`, A1 a A5 y A10 se mantienen y no hay violaciones | A2, A4 y A10 no se cumplen con corrección; 0 violaciones en dev |

Resultado: 6 aprobados, 6 no aprobados, 0 sin evaluar. Por la regla escrita, el veredicto es **NO APRUEBA**, y un criterio no evaluado habría bloqueado la aprobación.

### 10.2 Lectura de los criterios no aprobados

- **A2 (precisión de lote en dev gold).** El candidato tiene 3 lotes erróneos entre 252 filas gold que emiten predial; la línea base tiene 1 entre 250. Diferencia pareada de -0.0079, intervalo de bootstrap al 95 % de [-0.0202, +0.0001], P(diferencia < 0) = 0.90. El intervalo incluye el cero, así que la diferencia es compatible con ruido de muestreo. Sin embargo, ambas semillas del candidato quedan por debajo de la línea base (0.9880 y 0.9882), una brecha estable de unas dos filas. Una fila gold equivale a 0.346 puntos porcentuales.
- **A4 (pérdida de aciertos en dev).** 121 frente a límite 112. Diferencia pareada de +13 con intervalo de [-5, +31]: compatible con ruido. Las semillas del candidato dan 115 y 126, así que el valor cae dentro del rango entre semillas.
- **A6 (calibración).** El ECE fuera de muestra cumple con holgura (predio 0.0044 frente a límite 0.1075; manzana 0.0022 frente a límite 0.0438). Falla la regla de envío, que exige que el candidato no quede peor que el refit sin restricción en Brier (diferencia máxima de 0.002) y en AUC, en ambas cabezas. El Brier cumple en ambas cabezas. El AUC fuera de muestra queda por debajo del refit en las dos: predio 0.8951 frente a 0.8985, y manzana 0.9146 frente a 0.9176. Esto no es un problema de ruido, sino un criterio de diseño. La causa se infiere de las cifras de la tabla de calibración; la implementación de la regla no se reauditó línea por línea en este documento.
- **A7 (construcción y cobertura).** La pérdida de cobertura cumple (0.0142, frente a 0.0383 de pérdida inherente y límite). La compuerta de construcción muestra una proporción de 1.0044, por encima de 1. Esto es un artefacto de definición, no una mejora real. El numerador cuenta también filas cuyo lote no está en la fuente de nomenclatura (en dev, 287 coincidencias frente a 286 filas con lote en la fuente). Esas coincidencias solo son posibles mediante documentos de respaldo de catastro. Es una inferencia a partir de las cifras, no un hallazgo auditado en el código. El criterio se cumple, pero la métrica no debe leerse como proporción estricta.
- **A9 (pruebas).** Falla una prueba (`clasificador_edificaciones_cierre`) que pertenece a otro proyecto de la misma carpeta de trabajo. Las pruebas del normalizador pasan. El escáner de direcciones reales sale con código 0.
- **A10 (precisión de lote a confianza 0.7).** 0.9881 frente a piso 0.9910. Es la misma magnitud que A2 con el umbral de confianza, con la misma compatibilidad con ruido.
- **A12 (con corrección).** La corrección no cambia ninguna métrica de dev ni elimina fallos; los fallos se heredan de A2, A4 y A10. La parte congelada no se evalúa con corrección por diseño.

### 10.3 Decisiones abiertas

Estas decisiones corresponden al responsable del proyecto. El informe solo expone los hechos.

1. **A2 con tolerancia cero.** ¿Se mantiene la tolerancia cero (no aprueba) o se acepta una tolerancia de una o dos filas gold, coherente con la banda de ruido?
2. **A6, regla de envío frente al refit.** ¿Se sigue exigiendo esa regla o basta con el límite de ECE?
3. **A9, falla de un proyecto ajeno.** ¿Se cuenta el criterio como aprobado con esa excepción documentada, o se corrige primero el proyecto ajeno?
4. **Pérdida de precisión de manzana en la prueba congelada con todas las filas (segunda mirada).** La precisión de manzana baja de 0.7314 a 0.7121 (diferencia -0.0192; intervalo de [-0.0366, -0.0027]). Es una caída que se distingue del ruido de muestreo, mientras que en dev A5 se cumple y la cobertura de filas OK sube. A5 se evalúa solo en dev y no cubre esas filas. ¿Es aceptable esa pérdida a cambio de cobertura, o debe fijarse un criterio para las filas congeladas ruidosas?

Hay además una decisión de implementación: entre las variantes de calibración `basecal` y `cal`, se eligió `basecal`. Empatan en precisión de lote dentro de una fila gold; `cal` obtiene más filas con predial y menos pérdidas en dev, pero queda por debajo del piso de A5 (precisión de manzana 0.8485). Por eso se eligió `basecal`.

### 10.4 Resultados frente a la línea base

Las tasas son fracciones. La línea base es `baseline_prod_2026_10`, medida con el código vigente.

| Métrica | Línea base | Candidato | Diferencia |
|---|---:|---:|---:|
| dev gold OK (de 289) | 278 | 283 | +5 |
| dev gold precisión de lote | 0.9960 | 0.9881 | -0.0079 |
| dev gold precisión de manzana | 0.9964 | 0.9929 | -0.0035 |
| dev gold precisión predial exacta (NPN completo) | 0.9880 | 0.1865 | -0.8015 |
| dev precisión de manzana (todas las filas) | 0.8630 | 0.8609 | -0.0021 |
| dev precisión de lote (todas las filas) | 0.6566 | 0.6631 | +0.0065 |
| dev `lost_correct` (todas) | 108 | 121 | +13 |
| prueba congelada, gold OK (de 417) | 400 | 399 | -1 |
| prueba congelada, gold precisión de lote | 0.9615 | 0.9938 | +0.0323 |
| prueba congelada, gold precisión de manzana | 0.9949 | 0.9847 | -0.0101 |
| prueba congelada, todas OK (de 2 000) | 1 002 | 1 090 | +88 |
| prueba congelada, todas precisión de lote | 0.4912 | 0.5044 | +0.0132 |
| prueba congelada, todas precisión de manzana | 0.7314 | 0.7121 | -0.0192 |
| prueba congelada, todas `lost_correct` | 223 | 221 | -2 |

Notas de lectura:

- **Precisión exacta por NPN.** La caída de la precisión exacta (de 0.9880 a 0.1865 en dev gold, y de 0.9231 a 0.3096 en la prueba congelada gold) no mide un error de lote. Según la fuente, el candidato emite NPN de 30 dígitos con sufijo de unidad, mientras `gt_predial` es a nivel de lote. Un consumidor que compare NPN completos verá una coincidencia mucho menor. Esa lectura es la de la fuente y no se separó por experimento.
- **Banda de ruido.** Dev gold tiene 289 filas. Dos semillas del mismo procedimiento dan 281 y 280 filas OK en dev gold, con diferencia de 1. La diferencia de 5 filas OK frente a la línea base es significativa en el bootstrap (intervalo [1, 10]).
- **Segunda mirada.** La prueba congelada no es ciega: la decisión de añadir el respaldo de catastro se tomó después de ver el resultado congelado de la primera ronda (brecha de cobertura). No se ajustó ningún parámetro con la prueba congelada, y se puntuó una sola vez, pero esa puntuación no es una prueba independiente.

### 10.5 Calibración

**Qué logra.**

- Sobre pares reales de entrenamiento nunca usados para entrenar el modelo, con validación cruzada de 5 pliegues, el ECE de 10 bins es de 0.0044 para la cabeza de predio (n = 5 087) y de 0.0022 para la de manzana (n = 5 014). El refit de la línea base da 0.0975 y 0.0338.
- Brier de predio 0.0271 (refit 0.0270) y de manzana 0.0040 (refit 0.0039). AUC de predio 0.8951 (refit 0.8985) y de manzana 0.9146 (refit 0.9176). La calibración es buena; la discriminación es ligeramente inferior al refit.
- En dev gold, las filas con confianza mayor o igual a 0.70 cubren el 97.9 % de las filas, frente al 96.2 % de la línea base.

**Qué no logra.**

- **El ECE sobre pares de oro no describe el riesgo en filas ruidosas.** Los pares de `train_pairs` son oro por construcción: el texto coincide con la dirección del predio de referencia. En dev completo, con etiquetas GPS ruidosas, la banda alta tiene precisión de lote de 0.67 (sección 7). Esta lectura viene de `calib_design/SPEC.md`: sobre filas ruidosas, el modelo ajustado con oro sobreestima la probabilidad (predice 0.91 frente a 0.62 observado en lote), y no hay forma de aprender ese sesgo desde `train_pairs`.
- **Comparación con producción.** El documento de diseño (`calib_design/SPEC.md`, sección 1) reporta que el calibrador de producción, sobre `train_pairs`, predice 0.61 para filas de lote correctas en el intervalo de confianza 0.95 a 0.97, con un **ECE de 0.34 en ese intervalo**. Es una brecha de calibración por intervalo, medida con otro procedimiento. No es un ECE agrupado de 10 bins como el de la cifra del candidato (0.0044), así que **las dos cifras no son directamente comparables**. Para comparar con la misma métrica, la referencia correcta es el refit de la línea base (0.0975 y 0.0338), que sí cumple la definición de A6.
- **Confiabilidad de producción en general.** `calib_design/SPEC.md` (sección 1) indica que el `reliability.json` de producción estima el acuerdo con una parcela derivada de GPS, no que el código entregado sea la dirección escrita, y que su AUC sobre dev completo es de unos 0.57.

### 10.6 Auditoría JEV: qué puede y qué no puede validar

JEV es un juez basado en un modelo de lenguaje que compara una cadena de dirección con su normalización. No es verdad de referencia ni valida que el lote sea el correcto.

**Qué puede validar.** Equivalencia textual entre la consulta y la dirección normalizada, sobre filas públicas de nomenclatura y consultas sintéticas. Las direcciones ciudadanas (dev, prueba congelada, pares de entrenamiento) nunca se envían.

**Resultados.**

| Auditoría | Filas | Etiquetadas | Concordancia | IC 95 % (Wilson) | Kappa | AUC |
|---|---:|---:|---:|---|---:|---:|
| Nomenclatura (público), 300 filas | 300 | 296 | 1.0000 | 0.9872 a 1.0000 | indefinido | indefinido |
| Sintética, 300 filas (modelo no identificado en el JSON de auditoría) | 300 | 184 | 1.0000 | 0.9795 a 1.0000 | indefinido | indefinido |
| Sintética, candidato con respaldo de catastro | 300 | 192 | 0.9948 | 0.9711 a 0.9991 | 0.0000 | 0.7833 |
| Catastro (público), candidato con respaldo | 300 | 123 | 0.9837 | 0.9426 a 0.9955 | 0.0000 | 0.6302 |
| Catastro (público), línea base | 300 | 121 | 0.9752 | 0.9296 a 0.9915 | 0.0000 | 0.5918 |

**Qué no puede validar.**

- **No distingue un buen juez de uno permisivo.** Casi no devuelve veredictos decisivos "no": en la auditoría de nomenclatura todos los veredictos etiquetados son "sí" (tp 296, fp 0, fn 0, tn 0). Kappa y AUC son indefinidos o carecen de sentido con tan pocos negativos.
- **Muchos veredictos indeterminados.** En la auditoría sintética, 91 de 300 filas son indeterminadas y 25 se omiten.
- **No valida la corrección del lote.** Una concordancia alta significa que la cadena normalizada es textualmente equivalente a la consulta, no que el predio sea el correcto.
- **Cada discrepancia requiere revisión humana.** No hay revisión humana registrada en las cifras de este documento.

### 10.7 Qué se puede afirmar y qué no

**Sí se puede afirmar.**

- Con las reglas de aceptación escritas antes de entrenar, el candidato **no aprueba** (6 de 12 criterios).
- El candidato responde más filas con estado OK en dev gold y en la prueba congelada con todas las filas, y tiene mejor precisión de lote en la prueba congelada (gold y todas las filas). En gold congelado responde una fila menos (399 frente a 400).
- La calibración fuera de muestra del candidato tiene un ECE muy bajo frente al refit de la línea base.
- Las brechas de A2 y A10 (precisión de lote) son compatibles con ruido de muestreo, pero no con la dispersión entre semillas: ambas semillas quedan por debajo de la línea base. La brecha de A4 (pérdida de aciertos) es compatible con ruido y con la dispersión entre semillas.

**No se puede afirmar.**

- Que el candidato sea "aprobado" o "al menos tan confiable como producción". El veredicto escrito dice lo contrario.
- Que el candidato sea mejor que producción en precisión de lote en dev gold. Empeora en 0.0079 (no significativo, pero en la dirección contraria).
- Que la calibración sea fiable en filas ruidosas o en el uso real. Las cifras de calibración son de oro.
- Que el índice o el modelo funcionen fuera de Cali o con fuentes distintas de la nomenclatura del WFS.

## 11. Respaldo de catastro (reversible)

El índice base solo tiene nomenclatura. Una parte pequeña de los lotes reales existe en el catastro pero en ningún registro de nomenclatura. Para recuperarlos se añade, al final del índice base, un conjunto de documentos del catastro cuyo lote no está en el índice.

- **Regla determinista.** Se añade un documento de catastro si y solo si su lote no está en el índice base (unión de los lotes de documentos, unidades y todos los registros de nomenclatura conservados) y su clave de documento no colisiona con la de un documento existente. Se añadieron 6 941 documentos de 330 387 candidatos; 2 245 colisiones se omitieron.
- **El modelo no se reentrena.** Los embeddings base se copian y se verifican bit a bit; los nuevos se calculan con el mismo modelo en CPU. Puede haber diferencias en los últimos dígitos de fp16 frente a una recomputación en GPU.
- **Efecto medido en dev.** Cobertura de filas gold en el índice: de 285 a 288 de 289. Lotes ausentes del índice en dev completo: de 76 a 16. Precisión de manzana en dev completo con la calibración del candidato: 0.8609 (frente a 0.8614 sin respaldo).
- **Calidad del documento de catastro.** Sus direcciones no están normalizadas como las de nomenclatura. En dev completo, solo 2 de 5 filas con lote de referencia servidas desde catastro acertaron el lote. La muestra es pequeña.
- **Reversibilidad, tres mecanismos independientes:**
  1. Servir el índice base, que no se modifica.
  2. `index_fallback.py remove`†, que corta las filas agregadas y verifica byte a byte contra el manifiesto del índice base.
  3. Interruptor en tiempo de servicio, sin reconstruir: `"exclude_fuentes": ["catastro"]` en `decision.json` o `NORMALIZER_EXCLUDE_FUENTES=catastro`. Con él, el servicio atiende exactamente el índice sin esos documentos.

Ver [../README.md](../README.md) y la documentación de construcción del índice en [04-entrenamiento.md](04-entrenamiento.md) para los comandos.

## 12. Limitaciones

- **Una sola ciudad.** Todo se midió con datos de Cali. No hay medición en zona rural ni en otras ciudades.
- **Verdad de referencia ruidosa y sesgada hacia el catastro.** `gt_predial` es catastral y a nivel de lote. Un NPN certificado legítimamente distinto cuenta como error, así que la precisión predial es una cota inferior de la calidad real.
- **Muestra gold pequeña.** Dev gold tiene 289 filas; una fila equivale a 0.346 puntos porcentuales. Las diferencias de una o dos filas están dentro de la banda de ruido.
- **Prueba congelada no independiente.** Se usó una sola vez, con la corrida seleccionada, pero después de haber visto la brecha de cobertura de la primera ronda.
- **Cobertura.** El índice pierde lotes que solo existen en el catastro (pérdida inherente de 2.83 puntos; pérdida titular de 1.42 puntos con el respaldo). El respaldo recupera la diferencia con documentos de calidad inferior.
- **Pendientes de certificación.** Los registros `pcer` están marcados como `nomenclatura_pcer`. Pueden retirarse del índice sin afectar a los demás si su aporte resulta negativo.
- **Snapshot de nomenclatura.** La capa WFS tiene corte 2026-09-03Z; el shapefile es de 2026-10-30. Las diferencias se absorben a nivel de lote, no a nivel de sufijo.
- **JEV.** Juez basado en modelo de lenguaje, sin veredictos negativos suficientes para medir su calidad (sección 10.6).
- **Promoción.** Requiere la aprobación explícita del responsable del proyecto. Este informe no promueve nada.

## 13. Reproducibilidad y guardas

- **Criterios fijados antes de medir.** Los umbrales se escribieron antes de entrenar. Las enmiendas se hicieron antes de entrenar o puntuar candidatos (sección 2).
- **Hashes.** Cada archivo del índice lleva sha256 y tamaño en `index_meta.json`, escrito al final como marcador de completitud. Un directorio sin ese archivo no se trata como índice.
- **Producción intacta.** Los hashes de los archivos de servicio de producción son iguales antes y después de cada corrida (A8).
- **Guardas de entrenamiento.** Una corrida que se detiene antes de sus épocas configuradas (`stopped_early`) no es candidata. El presupuesto de minutos produce error (código de salida 4) en lugar de un resultado silencioso.
- **Prueba congelada.** Se puntúa solo una vez, con `--final`, sobre la corrida seleccionada.
- **Bootstrap.** 10 000 remuestreos, semilla 20261031, intervalo percentil de 95 %.
- **Guardas de publicación.** `scripts/check_publishable.py` (lista de bloqueo y patrones de secreto) y `scripts/check_no_real_addresses.py` (escaneo local con datos privados, fuera de CI).
- **Pruebas.** Suite de pytest: 6 533 pruebas pasan y 1 falla (ajena al proyecto, criterio A9). Según la documentación de aceptación del proyecto de origen (`docs/retrain_2026-10_acceptance.md`†, no publicada en este espejo), las fronteras de los criterios, los valores NaN y los campos faltantes tienen pruebas dedicadas (`tests/test_retrain_compare.py`†, tampoco publicada); esta sección no re-ejecutó esas pruebas.
- **Auditoría de criterios.** `scripts/compare_runs.py`† evalúa los criterios; `scripts/write_retrain_report.py`† produce el informe agregado con el veredicto calculado.

Comandos de reproducción principales (desde el directorio del modelo, con el entorno de CPU; el entrenamiento usa el entorno de CUDA):

```bash
# Descarga, limpieza e índice
python scripts/reproject_nomenclatura.py --shp <shapefile certificado>  # †
python scripts/download_nomenclatura.py --out artifacts/nomenclatura_wfs.parquet  # †
python scripts/clean_nomenclatura.py --in artifacts/nomenclatura_wfs.parquet --out artifacts/nomenclatura_clean.parquet  # †
python scripts/build_index.py --tiers cert,pcer --out-dir artifacts/index_v2e_cert_pcer_r3  # †

# Respaldo de catastro (opcional, reversible)
python scripts/index_fallback.py add --base-index artifacts/index_v2e_cert_pcer_r3 --splits-dir artifacts/splits --out-dir artifacts/index_v2g_cert_pcer_catfb_r3  # †

# Calibración y evaluación final (una sola vez)
python scripts/run_experiment.py --tag v2_final_catfb --eval-only --from-dir <directorio del candidato> --eval-set both --final
python scripts/compare_runs.py --old artifacts/experiments/baseline_prod_2026_10 --new artifacts/experiments/v2_final_catfb  # †
```

Los comandos completos, con el orden de todos los experimentos, están en el informe agregado del proyecto de origen.

## 14. Documentos relacionados

- [11-entrenar-para-cualquier-ciudad.md](11-entrenar-para-cualquier-ciudad.md): cómo entrenar para otra ciudad.
- [12-guia-metricas-y-validacion.md](12-guia-metricas-y-validacion.md): métricas y validación.
- [13-mecanismos-de-validacion.md](13-mecanismos-de-validacion.md): mecanismos de validación.
- [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md): definición de métricas y escala operativa.
- [09-advertencias-y-limitaciones.md](09-advertencias-y-limitaciones.md): advertencias de uso.
- [../../normalizador_produccion/README.md](../../normalizador_produccion/README.md): servicio en producción.
- [../../CITATION.cff](../../CITATION.cff) y [../../COMO_CITAR_Y_ATRIBUIR.md](../../COMO_CITAR_Y_ATRIBUIR.md): cómo citar y atribuir.
