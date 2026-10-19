# 09 - Advertencias y limitaciones

> Nota: Los nombres de las fuentes de evaluación se omiten por confidencialidad; los resultados de las corridas no se publican.

Este documento enumera lo que puede salir mal, los elementos específicos de Cali y de Colombia que contiene el código (con archivo y línea, verificados), los modos de fallo conocidos y las carencias del proyecto. Las cifras citadas se justifican en [08](08-metricas-y-confiabilidad.md).

**Estado:** versión preliminar probada con datos de Cali.

## Contenido

1. [Advertencias numeradas](#1-advertencias-numeradas)
2. [Elementos específicos de Cali y de Colombia](#2-elementos-específicos-de-cali-y-de-colombia)
3. [Modos de fallo](#3-modos-de-fallo)
4. [Carencias conocidas del proyecto](#4-carencias-conocidas-del-proyecto)

## 1. Advertencias numeradas

1. **Precondición: normalizador previo.** El modelo espera direcciones ya normalizadas. Sobre texto crudo, la lectura y el recall son muy inferiores a los de texto normalizado. Ver [06](06-predicciones.md).
2. **El puntaje no es una probabilidad.** Son constantes de diseño sin calibración medida. Úselo como categoría ordinal ([01](01-teoria.md), sección 9).
3. **Cadenas canónicas ambiguas en la base.** En el catastro evaluado, 268 cadenas canónicas están compartidas por varias parcelas (538 filas); para esas, un puntaje 1,0 vincula con parcelas distintas. Revise `ambiguous` y `near_ambiguous`.
4. **Un 1,0 puede tener alternativas.** Un edificio con unidades que tienen complemento conocido deja alternativas a 0,98 detrás del mejor candidato. Eso no es un error: es la regla de diseño del complemento en un solo lado.
5. **Incertidumbre del veto por letra de placa.** Una letra tras la placa (`20A` frente a `20`) es veto duro. No se confirmó con el propietario de los datos si esa letra es un código de unidad o un sufijo real de la placa. En el catastro evaluado hay letras de placa en cientos de registros (la `S` en 584 filas, la `T` en 425, la `B` en 322, entre otras). En texto real, "misma placa y otra letra" tuvo 1 falso vínculo de 18 (5,56 %, intervalo 0,29 a 27,29 %).
6. **Falsos vínculos entre variantes de tipo de cruce.** Pares con la misma vía y placa pero distinto cruce puntúan 0,97 aunque sean parcelas distintas. El catastro suele omitir el tipo del cruce, por eso se tolera.
7. **Direcciones sin leer.** Una fracción de las filas de un catastro no se lee (códigos rurales deformados y similares). Una consulta ilegible no es "sin candidato"; trátela aparte (`parse_ok`).
8. **Sin validación fuera de Cali.** Otras ciudades colombianas comparten la gramática pero no fueron medidas. Fuera de Colombia el modelo no es utilizable sin rehacer tablas y gramática ([10](10-estandar-y-generalizacion.md)).
9. **Sin etiquetas humanas y con métricas circulares.** Una precisión y un recall de 1,0 en las pruebas sobre el catastro provienen de pruebas circulares; no existen métricas con etiquetas humanas ni de calibración ([08](08-metricas-y-confiabilidad.md)).
10. **No es para decisiones legales ni catastrales sin revisión humana.** Un vínculo por dirección no prueba identidad jurídica del predio ni sustituye al número predial, el folio de matrícula ni la verificación en campo.
11. **Los separadores difieren del estándar nacional.** La gramática usa `#` entre la vía y el cruce y `-` antes de la placa, mientras que el estándar los excluye. Una dirección en formato estándar (`CL 11 SUR 23 A BIS 50`) no se lee. Ver [10](10-estandar-y-generalizacion.md).
12. **Heurística de letra o cuadrante.** Una letra aislada tras un número de sufijo se lee como cuadrante si cierra el segmento (`K 3 A 3 N`). La abreviatura `S` junto al número de vía se interpreta como letra de calle y no como `SUR` (`CL 11 S # 23 - 50` queda con la letra `S`). Verificado ejecutando el analizador. Este tratamiento difiere del normalizador: el normalizador nunca interpreta una letra aislada como cuadrante (`parser.py:94-98` de ese proyecto), mientras que el emparejador a veces sí lo hace.
13. **Datos de apoyo del nivel `building` específicos de Cali.** Los tipos de complemento que se ignoran en el nivel `building` se eligieron midiendo contra los números prediales del catastro de Cali. En otro catastro esa decisión debe volver a medirse.
14. **El nivel `structure` no se evaluó con datos reales** y tiene divergencias conocidas con el índice maestro de referencia: `CS` se trata como opaco y no como `CASA`; `ED`/`CONJ` en un solo lado cuestan 0,98; una letra de torre dentro del valor de la unidad es invisible.
15. **`max_block_scan` rompe la exhaustividad.** Si se usa, un candidato mejor puede quedar fuera (`truncated = True`).
16. **Los scripts de línea de comandos no exponen el nivel** ni consultas por lotes; `vincular.py` reconstruye el índice en cada ejecución y usa el código de salida 1 tanto para "sin coincidencia" como para "consulta ilegible".
17. **Las salidas de las herramientas de evaluación pueden contener direcciones reales** (hasta 5 ejemplos por categoría) y números prediales. Anonimice antes de publicar ([04](04-evaluacion-y-calibracion.md)).
18. **Concurrencia no probada.** La lectura del código indica que `find` no modifica el índice, pero no hay una prueba de concurrencia.
19. **Versión preliminar.** El proyecto no tiene licencia, integración continua ni empaquetado; la suite de pruebas original no se incluye en esta publicación.

## 2. Elementos específicos de Cali y de Colombia

Todas las referencias son del código fuente de este proyecto (verificadas).

| Elemento | Ubicación | Qué es específico |
|---|---|---|
| Gramática "para direcciones catastrales canónicas de Cali" | `src/emparejador/parser.py:1` y `6-7` | Nomenclatura urbana colombiana; `#` para la vía de cruce y `-` para la placa |
| Tabla de tipos de vía | `parser.py:58-100` | `CL`, `KR`, `AV`, `AC`, `AK`, `DG`, `TV`, `PJ`, `PS`, `CV`, `AU`, `CT` y sus alias en español |
| Alias de dos palabras | `parser.py:109-112` | `AVENIDA CALLE` a `AC`, `AVENIDA CARRERA` a `AK` |
| Tipos de vía de una sola letra heredados | `parser.py:63` (`C`), `71` (`K`), `77` (`A`), `84` (`D`), `89` (`T`) | Costumbre del catastro de Cali |
| `CARR` mapeado a `KR` | `parser.py:68` | Puede significar "carretera"; no aparece en el catastro evaluado |
| Cuadrantes | `parser.py:116` (`NORTE`, `SUR`, `ESTE`, `OESTE`) y `122-128` (abreviaturas `N`, `S`, `E`, `O`, `W`) | Convención colombiana de cuadrantes en nomenclatura |
| Tipos de complemento en español | `parser.py:139-206` | `APARTAMENTO`, `TORRE`, `LOCAL`, `BLOQUE`, `MANZANA`, `LOTE`, `EDIFICIO`, `CONJUNTO`, `URBANIZACION`, `INTERIOR`, `PARQUEADERO`, `GARAJE`, `BODEGA`; incluye `BARRIO` (`184`), `CORREGIMIENTO` (`186`) y `VEREDA` (`188`), divisiones político-administrativas colombianas |
| Registro `AGR`, `SEC` y alias | `parser.py:189-206` y `209-247` | Códigos que emite el normalizador hermano; `SECTOR` y `AGRUPACION` son palabras de cabeza únicamente |
| Topes de la cola y la placa | `parser.py:136` (40 caracteres) y `411` (6 dígitos) | Valores elegidos para el catastro de Cali |
| Letra de vía, número sufijo, letra de sufijo | `parser.py:483-548` | Estructura tipo `KR 31 E 3 # 84 - 15`; el estándar nacional no menciona número sufijo |
| Niveles de edificio derivados del número predial | `matcher.py:510-531` (comentario) y `532`, `565-566` | Se midió contra los primeros 21 caracteres del número predial nacional y el régimen colombiano de propiedad horizontal |
| Mensajes en español | `matcher.py:247, 250, 252, 265, 267, 443-446, 681-714`; `index.py:131, 252, 273`; `scripts/*.py` | Motivos y avisos en español (`veto: ...`, `sin_placa`) |
| Sufijo de consulta `", Cali, Valle del Cauca, Colombia"` | `tools/build_groundtruth.py:80` | Solo en la hoja de etiquetado |
| Columnas `comuna`, `barrio`, `manzana`, `numero_predial_nacional` | `tools/build_groundtruth.py:999-1000` | Estructura del catastro colombiano |
| Clave de parcela de 21 caracteres | `tools/eval_realworld.py:100` | Primeros 21 caracteres del número predial nacional (departamento a terreno) |
| Nombre de variable `CATASTRO_PARQUET` | `tools/eval_catastro.py:79` | Término colombiano; la variable es genérica |

No hay lista de barrios, comunas ni nombres de ciudades en `src/` ni en `scripts/`; las únicas apariciones de "Cali" en esas carpetas son la descripción del módulo (`parser.py:1`). No hay `TODO` ni `FIXME` en el código.

## 3. Modos de fallo

| Modo | Síntoma | Causa |
|---|---|---|
| Texto libre | `falta_separador_hash`, `token_inesperado:*` | El modelo no normaliza; solo acepta la forma canónica |
| `#` ausente | `falta_separador_hash` | Solo se recupera la forma `TIPO N N - N` ([01](01-teoria.md), sección 3) |
| Formato estándar sin `#` ni `-` | `falta_separador_hash` | Ver [10](10-estandar-y-generalizacion.md) |
| Ordinales y comas | `CL 5ta # ...` produce `token_inesperado:TA`; `CL 5, # ...` produce `numero_via_faltante` | Verificado ejecutando el analizador |
| Tipos de vía del estándar no registrados | `BL`, `CO` y `BULEVAR` producen `numero_via_faltante` | `BL` es alias de bloque en complementos; `CO` y `BULEVAR` no existen como tipos de vía |
| Circulares, kilómetros, carreteras rurales | Fallo de lectura | `CIR`/`CIRC` deliberadamente no se aliasan; `KM` no existe |
| Complemento muy largo | `complemento_demasiado_largo` | Tope de 40 caracteres |
| Segmentos de vía deformados | Fallo de lectura | Causa más frecuente de fallo de lectura en el catastro de Cali (sección A de `tools/eval_catastro.py`) |
| `CS` frente a `CASA` | NO_MATCH (0,0) | `CS` es opaco; solo `CA`/`CASA` están registrados |
| Cola opaca en un solo lado | 0,85, no vincula | Diseño: un complemento no entendido no se vincula con la dirección sin cola |
| Torre dentro del valor de la unidad | Invisible en `structure` | Se ignoran los valores de unidad |
| Placa vecina | NO_MATCH con tolerancia 0 | El veto de placa es exacto por defecto |
| Cambio de significado por mutación | Puntaje bajo | Una letra aislada puede cambiar entre sufijo y cuadrante al quitar un token vecino |

## 4. Carencias conocidas del proyecto

| Carencia | Detalle |
|---|---|
| Sin licencia | No hay archivo `LICENSE`. Licencia: por definir (ver LICENSE en la raíz cuando exista) |
| Sin integración continua | No hay flujo de CI |
| Sin `pyproject.toml` | Los scripts modifican `sys.path`; no hay paquete instalable ni versión de Python declarada de forma que la haga cumplir |
| Pruebas no incluidas | La suite original (717 pruebas aprobadas según el informe; archivos `tests/`) se excluyó de esta publicación |
| Sin etiquetas humanas | La hoja de etiquetado de pares de referencia quedó en blanco |
| Documentos originales en inglés | `../docs/*.md` están en inglés y describen datos privados por agregados; el README original mencionaba un índice maestro de un cuaderno privado que no se incluye |
| Sin registro de actividad ni métricas de operación | El proyecto no emite registros |
| Concurrencia sin pruebas | Ver advertencia 18 |
| Sin validación fuera de Cali | Ver advertencia 8 |
