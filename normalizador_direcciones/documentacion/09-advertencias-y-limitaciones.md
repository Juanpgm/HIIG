# 09. Advertencias y limitaciones

> Nota: Los nombres de las fuentes de evaluación se omiten (fuente A, B, …) por confidencialidad; los resultados de las corridas no se publican.

Lista numerada de lo que el usuario debe saber antes de confiar en el sistema, seguida del inventario de elementos específicos de Cali (con archivo y línea, verificados contra el código) y de las inconsistencias detectadas en la documentación original. Las cifras se explican en [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md); la forma de adaptar el sistema, en [10-estandar-y-generalizacion.md](10-estandar-y-generalizacion.md).

## Contenido

1. [Advertencias numeradas](#1-advertencias-numeradas)
2. [Elementos específicos de Cali](#2-elementos-específicos-de-cali)
3. [Inconsistencias en la documentación original](#3-inconsistencias-en-la-documentación-original)

## 1. Advertencias numeradas

**1. El sistema es solo para Santiago de Cali y no hay una puerta que lo impida.** El índice, los pesos, el generador de ruido, el gazetteer y los umbrales son de Cali. Una dirección de otra ciudad se compara con los predios de Cali y puede devolver una coincidencia aparentemente válida. La única protección es la opción `--municipality-col` de la CLI (`cli.py:134-135`, `io/pipeline.py:259`), que solo funciona si el usuario aporta una columna de municipio fiable. La API HTTP no tiene esa protección.

**2. `FUERA_DE_AREA` es una verificación débil.** Solo se emite cuando existe una columna de municipio, y solo reconoce Cali por nombre o por código DIVIPOLA `76001`, `76001000` o código postal `7600xx` (`io/pipeline.py:118-159`). Un valor vacío, un marcador o solo "Valle del Cauca" no se considera fuera de área. El estado no existe en la API HTTP.

**3. El índice es solo urbano.** Procede de la capa catastral urbana. Las direcciones rurales (veredas y corregimientos) casi no reciben coincidencia: un documento de auditoría no publicado cita 55 predios rurales entre 338 311 y una cobertura casi nula en corregimientos. Los 18 registros del índice con código de zona `00` (frente a `01` urbano) son la excepción.

**4. Vocabulario ASCII y truncamiento a 40 caracteres.** El codificador solo distingue 46 símbolos (`A-Z`, `0-9`, espacio y `# - . , / ' ( ) |`); lo demás es `UNK`. Las entradas se pasan a ASCII en mayúsculas y se truncan a 40 caracteres (`model.py:29-34`, `max_len` 40 del checkpoint). En el índice, 167 direcciones superan 40 caracteres y 172 contienen `Ñ` (sin convertir a `N` en el texto catastral que se codifica). El efecto no se midió.

**5. Rechazo por sufijo unilateral.** `KR 31 F` frente a `KR 31 F 3` se rechaza como calle distinta. En el conjunto de referencia del registro de cambios, 24 de las 79 filas rechazadas por esa causa eran el lote correcto. Es una decisión deliberada de precisión sobre cobertura; las filas se marcan con `regla: sufijo unilateral` para poder revisarlas.

**6. Límites del analizador gramatical.**
   - Una torre escrita **antes** de la dirección (`Torre B, Calle 9 # 41-26`) no se analiza.
   - Una segunda dirección escrita tras una unidad, sin coma ni punto y coma, queda dentro del complemento.
   - El analizador es permisivo: cadenas como `Calle 5 con 10` se aceptan y se convierten en una forma canónica incompleta (`CL 5 CON # 10 -`), y decide después la recuperación y la capa de reglas. Se observó con direcciones inventadas.
   - Un número tras `BIS` (`Cra 100 BIS 2 # 15-30`) no se interpreta como parte de la vía: se desplaza al final como complemento (observado con una dirección inventada: `KR 100 BIS # 15 - 30 2`).
   - No se reconocen `Boulevard` (`BL`) ni `Conjunto` (`CO`) como tipo de vía; `Autopista` sin número tampoco.

**7. Brecha entre sintético y real.** El modelo se entrena con ruido generado por reglas. La validación sintética no anticipa el desempeño en texto real, donde solo una parte de las filas llega a `OK` y la precisión de predio sobre las `OK` es bastante menor que la sintética ([08](08-metricas-y-confiabilidad.md)).

**8. `confianza` no es una probabilidad y la `confiabilidad` servida (v4) casi no ordena.** Una banda alta de `confianza` no garantiza el predio exacto. La `confiabilidad` v4 del predio se ajustó con etiquetas GPS ruidosas y subestima la tasa real en filas cuyo texto coincide con el catastro. El calibrador alternativo v5 (`reliability_v2.json`) ordena mejor en ese tipo de filas, pero no se sirve por defecto, no se midió en filas con texto inconsistente y no mejora la manzana. No utilice ninguna de estas puntuaciones como criterio único de aceptación, y no mezcle cortes entre versiones ([08](08-metricas-y-confiabilidad.md), secciones 4 y 5).

**9. Sin uso sin supervisión humana en decisiones jurídicas o catastrales.** El sistema normaliza y cruza direcciones. No sustituye la identificación registral o catastral, ni los trámites de formalización, avalúo, impuesto, notificación o sanción. Cualquier decisión con efectos sobre personas requiere verificación humana del predio.

**10. Evaluación contra geocodificadores de terceros.** El módulo de evaluación contra geocodificadores de terceros no se incluye en esta versión pública.

**11. Deserialización insegura.** `model.pt` se carga con `torch.load(..., weights_only=False)` (`inference.py:310`, `train.py:170`); `gazetteer.pkl`, con `pickle.load` (`gazetteer.py:949`); `train_data.npz`, con `np.load(..., allow_pickle=True)` (`scripts/train.py:125`, `scripts/tune_threshold.py:261`, `scripts/finish_training_artifacts.py:34`). Cargar un archivo manipulado puede ejecutar código arbitrario. Verificado: `model.pt` no se puede cargar con `weights_only=True` porque contiene arreglos de NumPy. Cargue solo artefactos de origen confiable y compare el SHA-256 ([05](05-uso-e-implementacion.md), sección 9).

**12. La API no tiene autenticación, limitación de tasa ni controles de acceso.** Publicarla en una URL abierta la expone a cualquiera. Ver [05](05-uso-e-implementacion.md), sección 9.

**13. Direcciones compartidas.** Cuando varios predios comparten una dirección (algunos documentos agrupan muchos predios), se devuelve el primero por `OBJECTID` y el nivel baja a `manzana` o `direccion`. No identifica la unidad (apartamento, local).

**14. Coordenadas.** `lat` y `lon` son el centroide del polígono del predio (o la mediana de la dirección si falta el archivo de centroides propios), no el acceso. La proyección métrica está fija en EPSG:3116 (`geo.py:17`).

**15. El texto catastral se devuelve tal como está.** En filas `OK`, `direccion_normalizada` reproduce la dirección catastral, incluidos los tipos de vía de una letra de uso antiguo (`K`, `C`, `A`, `D`, `T`; unas 23 700 de 330 387 direcciones, el 7 %), 23 direcciones con minúsculas y 172 con `Ñ`. Solo las filas por reglas (`fuente_normalizacion = reglas`) usan siempre los códigos canónicos (`KR`, `CL`, ...). Consecuencia: la salida no siempre cumple el estándar nacional de abreviaturas ([10](10-estandar-y-generalizacion.md)).

**16. Las métricas son anteriores a la auditoría de octubre de 2026.** Las métricas de prueba congelada y desarrollo de corridas anteriores se midieron con una versión del código distinta de la actual. La auditoría cambió reglas de decisión; vuelva a medir con el código vigente.

**17. Un único origen de texto en la prueba congelada y etiquetas de GPS ruidosas.** Ver [08](08-metricas-y-confiabilidad.md), sección 7.

**18. El entrenamiento no es reproducible bit a bit** (cuDNN con benchmark, TF32, bf16). Los reentrenamientos con la misma semilla difieren en milésimas de recall y en 1 a 3 filas de desarrollo.

**19. El gazetteer puede alterar el texto.** Elimina menciones de barrio, vereda y corregimiento antes del análisis y las convierte en restricción geográfica. Un nombre de lugar mal detectado puede excluir candidatos correctos (`candidatos fuera de ...`). Sin los dos GeoJSON de límites, el gazetteer se desactiva y se pierde la puerta geográfica.

**20. El índice es una instantánea.** El catastro se descargó en una fecha concreta (`artifacts/catastro_meta.json`); no se actualiza solo. Una dirección nueva o modificada no se encontrará hasta reconstruir el índice y los embeddings (el modelo no se reentrena por ello).

**21. Los endpoints con forma Esri no son un Locator de ArcGIS.** Reproducen solo la forma de la respuesta ([05](05-uso-e-implementacion.md), sección 6).

**22. Rendimiento de referencia limitado.** Solo se ha medido un trabajador en CPU. No hay mediciones de concurrencia ni de latencia por petición.

**23. Datos personales.** Las direcciones de entrada pueden ser datos personales de terceros. El sistema no las envía a servicios externos al normalizar, pero la gestión de los archivos de entrada y salida es responsabilidad del usuario.

**24. Las bandas de confiabilidad no están validadas.** La `confiabilidad` servida (v4, `reliability.json`) casi no distingue filas correctas de incorrectas y subestima; el calibrador alternativo (v5, `reliability_v2.json`) se ajustó solo con etiquetas consistentes con el texto, usa otra escala (corte equivalente 0.81 en lugar de 0.58) y se midió una sola vez en un conjunto pequeño. Ninguna banda (en particular la escala operativa de [08](08-metricas-y-confiabilidad.md), sección 5) se ha validado fuera de Cali ni con otras fuentes. No promueva `reliability_v2.json` a `reliability.json` sin repetir esa medición con sus propios datos.

**25. La guarda estructural es estricta a propósito.** El cuadrante se compara por posición (vía, cruce, final): un cuadrante movido de posición se rechaza; `N` equivale a `NORTE` solo dentro del mismo segmento. Una placa distinta **nunca** es una puerta verificada (`address_verified` es `False`; como máximo `position_tier = tramo`). Un sufijo numérico de vía en un solo lado (`KR 26 H` frente a `KR 26 H 3`) se rechaza con `regla: sufijo unilateral`, y esa regla tiene un costo medido: 24 de 79 filas `gold` rechazadas así eran el predio correcto. Detalle en [06](06-predicciones.md), sección 10.5.

**26. Una dirección en blanco no es un error.** `direccion_normalizada` queda `null` (y `fuente_normalizacion` también) cuando el texto no es confiable: filas no analizables, filas `below_cutoff` o `rejected_mismatch` con un análisis inseguro, y **toda** fila con `fidelity_risk = high`, también las `OK`. La canalización de archivos recurre entonces al texto original (`fuente_direccion = origen`); no trate esos casos como fallas del programa ([06](06-predicciones.md), sección 10.2).

**27. Los radios de posición son geométricos, no una medida de acierto.** `position_radius_m` se derivó del propio catastro (distancias entre el centroide del predio, del lote o de la manzana y sus límites, o el error de extrapolar entre predios vecinos), sin GPS ni verdad de campo. La coordenada de las filas `via` se retiene por defecto (`EXTRAPOLATED_COORDS_ENABLED = False`) porque no existe ningún predio en ese punto.

**28. La medida de fidelidad es independiente del analizador, no infalible.** `fidelity_risk` usa un tokenizador propio para poder detectar errores del analizador de producción, pero puede equivocarse ante entradas muy sucias; `medium` y `high` limitan el nivel de posición y pueden degradar una fila que era correcta. La política se apaga con `service.FIDELITY_CAPS_ENABLED = False` (las columnas se conservan).

**29. Cachés perezosas con pickle.** `interp_index_v1.pkl` y `catalog_keys_v1.pkl` se leen con `pickle`, igual que `gazetteer.pkl`: el hash del archivo `.json` acompañante se verifica antes de leer, pero esa verificación no protege contra quien pueda escribir en el directorio de artefactos. No se versionan; se regeneran solas ([06](06-predicciones.md), sección 10.6).

## 2. Elementos específicos de Cali

Inventario verificado contra el código fuente de origen (ruta relativa a la raíz del repositorio).

| Elemento | Archivo y línea | Qué fija |
|----------|-----------------|----------|
| Cortes del NPN para comuna y barrio | `src/cali_address/dataset.py:44-46` | `comuna = NPN[9:11]`, `barrio = NPN[11:13]`, `barrio_code = NPN[9:13]`. En Cali el "sector" del estándar (posiciones 8-9) es `00` en todo el índice. |
| Cortes equivalentes en evaluación y particiones | `src/cali_address/evaluate.py:246`, `scripts/build_splits.py:109` | `[9:11]` para la comuna de la referencia. |
| Clases de las cabezas auxiliares | `artifacts/model.pt` (`n_comuna` 24, `n_barrio` 342) | Códigos de comuna (`''`, `01` a `22`, `56`) y 342 códigos de barrio de Cali. |
| Media y desviación de coordenadas | `artifacts/model.pt` (`xy_mean`, `xy_std`) | Estandarización de `coord_head` con la media y desviación de los centroides de Cali (EPSG:3116). |
| Comunas válidas 1 a 22 | `src/cali_address/gazetteer.py:129` (uso en `:498`) | Una comuna fuera de ese rango no se acepta. |
| Esquema y nombres de los límites | `src/cali_address/gazetteer.py:22-27`, `:99-100`, lectura en `:868-915` | Campos `COMUNA`, `BARRIO`, `ID_BARRIO`, `VEREDA`, `ID_VEREDA`, `CORREGIMIE`, `ID_CORREG`; archivos `barrios_veredas.geojson` y `comunas_corregimientos.geojson`. |
| Ruido de ciudad en el gazetteer | `src/cali_address/gazetteer.py:139-142` | `CALI`, `SANTIAGO DE CALI`, `VALLE DEL CAUCA`, etc. |
| Ruido de ciudad en el parser | `src/cali_address/parser.py:151-154` | Cola administrativa descartada. |
| Convención de cuadrantes catastral | `src/cali_address/parser.py:88-98` | `NORTE`, `SUR`, `ESTE`, `OESTE`; las letras sueltas no son cuadrantes; `ESTE` no aparece en el catastro de Cali. También `service.py:309` (`_QUADRANT_LETTER_EQUIVALENTS`). |
| Etiquetas de comuna `Comuna N` | `src/cali_address/service.py:552` y `gazetteer.py:906` | Formato del nombre de la comuna. |
| Puerta municipal | `src/cali_address/cli.py:134-135`, `src/cali_address/io/pipeline.py:118-159`, `:259` | Solo Cali y el código `76001`. |
| CRS métrico fijo | `src/cali_address/geo.py:17` | `EPSG:3116`. |
| Caja de coordenadas de Cali | `src/cali_address/eval_data.py:193-198` | Latitud 3.2 a 3.65, longitud -76.75 a -76.4 (saneamiento de evaluación). |
| Grados a metros | `src/cali_address/gazetteer.py:116` | 111 000 m por grado (válido a la latitud de Cali, error de 0.3 %). |
| Sufijos de ciudad y barrios en el generador de ruido | `src/cali_address/augment.py:32-39` | Listas de ciudad y de diez barrios de Cali. |
| Fuente de descarga del catastro | `scripts/download_catastro.py` (`SOURCE_ENV`, `get_service_url`) | Sin fuente por defecto: la URL de la capa catastral municipal se entrega mediante la variable de entorno `CALI_CATASTRO_URL`. |
| Escala geográfica, pesos, umbral | `src/cali_address/inference.py:48`, `:53`, `:59` | `FUSE_WEIGHTS`, `GEO_TAU_M = 600`, `CONFIDENCE_THRESHOLD = 0.73` (sobrescritos por `tuning.json` los dos de fusión y umbral). |
| Holgura de barrio y de zona | `src/cali_address/service.py:146`, `src/cali_address/gazetteer.py:108`, `:111` | 1000 m (servicio), 200 m (módulo, no usado por defecto), 300 m. |
| Mínimo estructural y ambigüedad | `src/cali_address/service.py:154`, `:648` | 0.6 y 0.02. |
| Longitud máxima y vocabulario | `src/cali_address/model.py:29-34`; `TrainConfig.max_len` en `train.py` | 48 símbolos y 40 caracteres. |
| Nombre y título del servicio | `src/cali_address/api/esri.py:48`, `src/cali_address/api/main.py:446` | `CaliNormalizador`, "Normalizador de direcciones - Cali". |
| Archivos de evaluación privados | `src/cali_address/eval_data.py` (`DATASETS`) | Nombres genéricos (`source_a.xlsx`, …) que se sustituyen con las variables de entorno `CALI_EVAL_SOURCE_<A..E>_FILE`, `CALI_EVAL_SOURCE_<A..E>_SHEET` y `CALI_EVAL_CONTEXT_DIR`; los archivos no se distribuyen. |

La línea de `service.py:546` que cita el informe previo para la etiqueta `Comuna N` es en realidad `service.py:552`, y la de `gazetteer.py:113` para los grados a metros es `gazetteer.py:116`.

## 3. Inconsistencias en la documentación original

Detectadas al verificar cada afirmación contra el código y los archivos.

| # | Documento y afirmación | Hallazgo | Estado |
|---|------------------------|----------|--------|
| 1 | `docs/API.md`: la confiabilidad es una "regresión logística pequeña" | `reliability.json` publicado es de tipo `hybrid` (isotónica por tipo de fila) | Desactualizado |
| 3 | README original: "everything except `catastro_emb.pt` is tracked in `artifacts/`" | En este repositorio publicado no están el índice, los centroides propios ni el gazetteer | No aplica a la versión publicada |
| 4 | Mensaje de `cali-address setup`: "These files are tracked in git: run git pull" | Mismo motivo que el punto 3 | No aplica a la versión publicada |
| 5 | `docs/model-artifacts.md`: `catastro_docs.parquet` "Public cadastre ... no personal data" | Contiene direcciones y NPN reales; la licencia de redistribución no está resuelta | No se distribuye |
| 6 | `docs/DEPLOY.md`, `Dockerfile`: "artifacts de 199 MB" y "415 MB" | 207 MiB (217,026,179 bytes) con el archivo de centroides propios; `artifacts/` ocupa 2.1 GB con los experimentos | Cifras antiguas |
| 7 | `docs/DEPLOY.md` frente a `docs/model-artifacts.md`: 161 MB frente a 169 MB para `catastro_emb.pt` | 161.3 MiB = 169.2 MB; es una diferencia de unidades | Coherente (unidades) |
| 8 | `docs/model-artifacts.md`: `model.pt` de "14 MB" | 13 527 959 bytes = 13.5 MB = 12.9 MiB | Redondeo |
| 9 | `model.py`: `n_layers=5`, `MAX_LEN=48`, `n_barrio=400` por defecto | El checkpoint tiene 4 capas, longitud 40 y 342 clases de barrio | Valores por defecto distintos del modelo publicado |
| 10 | `Dockerfile`: exige `gazetteer.pkl` y los límites con `test -f` | El gazetteer es opcional en ejecución (se desactiva con una advertencia) | La construcción es más estricta que el tiempo de ejecución |
| 11 | `scripts/build_notebook.py` (no incluido en esta versión pública) (§7.3.1): el código de barrio de IDESC no coincide con `predial[9:13]` | El notebook retracta la equivalencia de barrio; solo la comuna `predial[9:11]` coincide con IDESC en casi todos los casos | Relevante para quien use el barrio del NPN como etiqueta de barrio oficial |
| 12 | README original: licencia "Not yet specified" | `pyproject.toml` también omite la licencia | Pendiente |
| 13 | `train_report.json` registrado por `finish_training_artifacts.py` | El reporte publicado se reconstruyó tras un fallo al embeber (el reporte conserva la historia por época) y su `config` no incluye todos los hiperparámetros de la receta | Aclaración |
