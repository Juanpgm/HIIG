# 06. Predicciones: entradas, salidas e interpretación

Este documento describe qué acepta el normalizador, qué columnas devuelve y cómo leerlas. Todas las direcciones, números y coordenadas de los ejemplos son **inventados**. Para el razonamiento detrás de cada decisión véase [01-teoria.md](01-teoria.md) y [02-arquitectura.md](02-arquitectura.md); para cuánto confiar en cada columna, [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md).

## Contenido

1. [Formatos de entrada](#1-formatos-de-entrada)
2. [Esquema de salida](#2-esquema-de-salida)
3. [Valores de `estado`](#3-valores-de-estado)
4. [Valores de `motivo`](#4-valores-de-motivo)
5. [`nivel_precision`](#5-nivel_precision)
6. [`confianza`, `confiabilidad` y `margen`](#6-confianza-confiabilidad-y-margen)
7. [Flujo de decisión resumido](#7-flujo-de-decisión-resumido)
8. [Guía de interpretación con ejemplos](#8-guía-de-interpretación-con-ejemplos)
9. [Lectura recomendada para quien consume los resultados](#9-lectura-recomendada-para-quien-consume-los-resultados)
10. [Posición, fidelidad y estado de catálogo](#10-posición-fidelidad-y-estado-de-catálogo)

## 1. Formatos de entrada

La unidad de entrada es una **cadena de texto con una dirección**. Puede llegar de tres maneras:

| Origen | Cómo se indica la dirección |
|--------|-----------------------------|
| Tabla (CSV, Excel, Parquet, JSON, SQL, etc.) | Una columna (`--address-col`, `address_column`) o varias partes que se unen en orden (`--address-parts`). Si no se indica, la columna se detecta por sinónimos en español e inglés; si dos columnas son igual de plausibles, el proceso se detiene y lista las candidatas. |
| API JSON | `addresses` (lista de cadenas) o `records` más `address_field`. |
| Forma Esri | `SingleLine`, `Address` o `text`. |

Qué entiende el analizador (formas de escritura toleradas, con ejemplos inventados):

| Variante de escritura | Resultado canónico |
|-----------------------|--------------------|
| `Cl. 5 # 34-12`, `Calle 5 No 34 12`, `Cll 5 34 12` | `CL 5 # 34 - 12` |
| `Carrera 7ma # 45-10 apto 301` | `KR 7 # 45 - 10 AP 301` |
| `Avenida 6 Norte # 28N-12` | `AV 6 NORTE # 28 N - 12` |
| `CARRERA 31 F 2 # 84 - 15` | `KR 31 F 2 # 84 - 15` |
| `Calle 9 # 41-26, Santiago de Cali, Valle del Cauca` | `CL 9 # 41 - 26` (se descarta la cola de ciudad y departamento) |

El texto se convierte a mayúsculas y a ASCII (se eliminan acentos), se unifican guiones y separadores, y se interpretan `No`, `Nro`, `N°`, `Num` y `#` como separador de cruce. Los ordinales (`7ma`, `1ra`) se descartan.

Entradas que producen un rechazo sin intentar coincidencia:

- Vacías o marcadores sin información: `N/A`, `NO REPORTA`, `SIN DIRECCION`, `-`, `0`, `NULL`, etc. Cualquier cadena sin letras y con menos de 3 dígitos. Resultado: `SIN_MATCH` con `motivo = dato vacio o marcador sin informacion`.
- Texto que la gramática no reconoce (por ejemplo, solo el nombre de un barrio, o texto libre): `NO_PARSEABLE`.

Límites de la entrada:

- Una dirección con un nombre de barrio, vereda, comuna o corregimiento escrito se limpia primero (si el gazetteer está activo) y esa mención se usa como restricción geográfica.
- Una torre escrita **antes** de la dirección (`Torre B, Calle 9 # 41-26`) no se analiza. La torre después de la dirección sí (`Calle 9 # 41-26, Torre B`).
- Una segunda dirección escrita a continuación de una unidad, sin coma ni punto y coma, queda dentro del complemento.
- La parte útil para la búsqueda neuronal se trunca a 40 caracteres.
- El municipio no se deduce del texto. Solo se verifica si se pasa `--municipality-col`.

## 2. Esquema de salida

Las columnas originales se conservan primero (salvo `--keep-columns`) y se añaden las siguientes, siempre en este orden. Un nombre de columna original igual a uno de salida (o un par `lat`/`lon` en la entrada) se conserva como `<nombre>_original`.

| Columna | Tipo | Descripción | Es `null` cuando |
|---------|------|-------------|------------------|
| `direccion_entrada` | texto | Texto recibido (en la CLI se omite si la dirección es una sola columna conservada). | nunca |
| `estado` | texto | `OK`, `SIN_MATCH`, `NO_PARSEABLE`, y solo en la CLI y la canalización de archivos: `FUERA_DE_AREA`, `ERROR`. | nunca |
| `motivo` | texto | Por qué la fila no es `OK`; en filas `OK`, vacío o notas separadas por `; `. | vacío en una fila `OK` simple |
| `direccion_normalizada` | texto | Dirección base del catastro (filas `OK`) o forma canónica por reglas (filas no `OK` pero analizables), seguida del complemento de la entrada. | entrada vacía, no analizable, `ERROR`, `FUERA_DE_AREA` |
| `fuente_normalizacion` | texto | `catastro` (coincidencia real), `reglas` (solo gramática) o vacío. | si `direccion_normalizada` es nula |
| `numero_predial_nacional` | texto | NPN de 30 dígitos del predio. | filas no `OK`; catastro sin NPN |
| `manzana` | texto | Código de manzana catastral. | filas no `OK` |
| `lat`, `lon` | decimal | Centroide del predio cuyo NPN se reporta, en EPSG:4326. Si no existe el archivo de centroides propios, el centroide de la dirección (mediana de los predios que la comparten). | filas no `OK` |
| `confianza` | decimal 0-1 | Puntaje fusionado, sin calibrar. | filas no `OK` |
| `confiabilidad` | decimal 0-1 | Probabilidad estimada de que el NPN sea correcto. Informativa; no altera `estado`. | filas no `OK`; falta `reliability.json` |
| `confiabilidad_manzana` | decimal 0-1 | Ídem para la manzana; siempre mayor o igual que `confiabilidad`. | ídem |
| `margen` | decimal | Diferencia de puntaje con el mejor candidato alternativo que cumple las reglas y está en **otra manzana**. | filas no `OK`; no hay competidor |
| `nivel_precision` | texto | `predio`, `via`, `esquina`, `manzana` o `direccion` (sección 5). | filas no `OK` |
| `barrio_vereda` | texto | Barrio o vereda oficial. En filas `OK`, el polígono donde cae el punto reportado; en las demás, el lugar detectado en el texto. | nada conocido |
| `comuna_corregimiento` | texto | `Comuna N` o nombre del corregimiento (misma procedencia). Sin gazetteer, se deriva de los dígitos de comuna del NPN. | nada conocido |
| `complemento_normalizado` | texto | Parte de unidad de `direccion_normalizada` (por ejemplo `TO F AP 101`), tomada de la **entrada** y escrita de forma canónica. | la entrada no trae unidades |
| `position_tier` | texto | Qué tan bien ubicada quedó la fila: `predio`, `lote`, `tramo`, `manzana`, `via` o `none` (sección 10.1). | nunca (`none` si no hay posición) |
| `position_radius_m` | decimal | Radio de error geométrico (p90, derivado del catastro, en metros) del nivel de posición (sección 10.1). | `position_tier` es `none` |
| `position_source` | texto | Origen de la coordenada: `parcel` (centroide de un predio), `interp` (interpolada o extrapolada) o `none`. | nunca (`none` si no hay coordenada) |
| `selected_doc_id` | entero | Índice del documento catastral elegido (auditoría). | filas no `OK` |
| `override_rejected` | booleano | `True` si el candidato recuperado fue rechazado por las reglas estructurales porque su base difiere de la dirección de entrada. | nunca (`False` por defecto) |
| `direccion_catastral` | texto | Dirección del predio coincidente tal como la escribe el catastro (solo filas `OK`; para auditoría). | filas no `OK` |
| `fidelity_risk` | texto | Riesgo de que `direccion_normalizada` sea **otra** dirección que la de entrada: `none`, `low`, `medium` o `high` (sección 10.3). | nunca (`none` por defecto) |
| `fidelity_flags` | texto | Banderas de `fidelity_risk` separadas por `;` (por ejemplo `plate_changed;letters_dropped`); `no_address` si no hay dirección de salida. | vacío si no hay hallazgos |
| `catalog_status` | texto | Estado frente al catastro: `verified`, `ambiguous`, `door_not_catalogued`, `rejected_mismatch`, `unparsed` o `below_cutoff` (sección 10.4). | nunca |
| `address_verified` | booleano | `True` exactamente cuando `catalog_status` es `verified`. | nunca |

Notas:

- Las filas que no son `OK` **no reciben ningún campo catastral**. Es deliberado: una conjetura de baja calidad no debe confundirse con una coincidencia.
- `ERROR` y `FUERA_DE_AREA` solo traen `direccion_entrada` y `motivo`.
- La API HTTP devuelve las mismas columnas (sin `ERROR` ni `FUERA_DE_AREA`). El endpoint con forma Esri solo expone una selección (ver [05](05-uso-e-implementacion.md), sección 6).
- Las diez columnas finales (`position_tier` ... `address_verified`) se describen en la sección 10; el endpoint con forma Esri no las expone.
- Las coordenadas son una ayuda, nunca un criterio de corrección: son el centroide de un polígono, no la puerta de acceso.
- `direccion_normalizada` en filas `OK` reproduce la dirección **tal como está escrita en el catastro**, incluidos usos antiguos como tipos de vía de una letra (`K`, `C`, `A`, `D`, `T`), que en el índice de Cali encabezan aproximadamente el 7 % de las direcciones. No se reformatea al estándar nacional (ver [10](10-estandar-y-generalizacion.md)).

## 3. Valores de `estado`

| `estado` | Significado | Qué hacer |
|----------|-------------|-----------|
| `OK` | Se analizó, coincidió con un predio sobre el umbral, el predio cae dentro del lugar detectado (si hubo), cumple las reglas estructurales y no hay un competidor demasiado cercano en otra manzana. | Usar según `nivel_precision`, `confiabilidad` y `margen`. Ver la guía de la sección 9. |
| `SIN_MATCH` | Se analizó, pero no se puede asignar un predio. `motivo` explica por qué. | Revisión manual o completar la dirección. No hay candidato. |
| `NO_PARSEABLE` | La gramática no reconoció el texto como dirección. Las columnas de zona se llenan si se detectó un lugar. | Corregir la fuente o descartar. |
| `FUERA_DE_AREA` | Solo con `--municipality-col`: el municipio indicado no es Cali. No se intentó coincidencia. | Enrutar a otro catastro. |
| `ERROR` | Solo CLI y archivos: la fila hizo fallar al normalizador. La corrida continúa. | Reportar el caso; `motivo` trae el tipo de excepción. |

## 4. Valores de `motivo`

### 4.1 En filas `SIN_MATCH`

| Texto de `motivo` | Causa |
|-------------------|-------|
| `dato vacio o marcador sin informacion` | Entrada vacía o marcador. |
| `confianza X < umbral Y` | El mejor candidato no alcanzó el umbral (0.73 por defecto). |
| `candidatos fuera de <barrio, comuna o corregimiento> <nombre>` | Ningún candidato cae dentro del lugar detectado en el texto. |
| `tipo de via A != B`, `numero de via A != B`, `numero de cruce A != B`, `tipo de cruce A != B` | Regla de identidad violada. |
| `placa A != B` | La placa difiere (en más de la tolerancia y la ventana blanda, o con otras violaciones blandas excediendo `max_soft`). |
| `numero de sufijo de via - != 3` (y análogos para el cruce) | Sufijo numérico distinto o presente en un solo lado. Si es la única causa, `motivo` termina con `regla: sufijo unilateral`. |
| `letra de via`, `letra de cruce`, `letra de sufijo de ...` seguidos de `A != B` | Letras de vía o cruce distintas. |
| `bis de via si != no`, `bis de cruce ...` | BIS presente en un solo lado. |
| `cuadrante de via A != B`, `cuadrante A != B` | Cuadrantes en desacuerdo. |
| `cuadrante en complemento X no esta en catastro` | Un cuadrante escrito en el complemento no aparece en el candidato. |
| `coincidencia estructural X < Y` | `struct` por debajo de `min_struct` (0.6). |
| `candidato catastral no parseable` | La dirección catastral candidata no se pudo analizar. |
| `ambiguo: N candidatos cercanos en otras manzanas` | `margen` menor que `ambiguity_delta` (0.02). |

Varias razones se concatenan con `; `.

### 4.2 En filas `NO_PARSEABLE`

Las notas del analizador, por ejemplo `explicit_hash; via_unparsed`, `implicit_hash; empty_via_segment`, `empty_input`, `dropped_city_noise; ...`. No son frases para el usuario final; sirven para diagnosticar.

### 4.3 En filas `OK`

Vacío (fila simple) o una o varias notas:

| Nota | Significado |
|------|-------------|
| `aproximado: <regla>` | Una violación blanda aceptada (letra de vía en un solo lado, placa a 1 o 2 unidades, cuadrante en un solo lado). |
| `nivel maximo <nivel>: <razon>` | Techo de `nivel_precision` impuesto por la aproximación. |
| `fuera de <lugar>` | Solo con `gate_fallback`: el lugar detectado excluía a todos los candidatos. |
| `direccion compartida por N predios` | Varias parcelas comparten la dirección; se devuelve la primera por `OBJECTID`. |
| `coordenada: centroide de la direccion compartida por N predios` | Faltan los centroides propios; el punto es la mediana de la dirección. |
| `NPN nulo en catastro` | El catastro no trae NPN para la dirección (nivel máximo `direccion`). |
| `NPN anomalo en catastro (<motivo>)` | El NPN no tiene 30 dígitos o tiene caracteres no numéricos; se conserva. |
| `complemento distinto al catastro: <tipos>` | La entrada da para un tipo de unidad un valor que el catastro no tiene; gana la entrada. |

## 5. `nivel_precision`

Indica cuánto de la terna vía, cruce y placa fue confirmado. El nivel de una fila es el **menos preciso** de los que le aplican. De menos a más preciso:

| Nivel | Cuándo | Qué respalda |
|-------|--------|--------------|
| `direccion` | Dirección compartida entre varias manzanas, o NPN nulo. | Solo la dirección; el predio y a veces la manzana son ambiguos. |
| `via` | La entrada no trae número de cruce. | La vía; no un predio concreto. |
| `esquina` | Sin placa. | La intersección de las vías. |
| `manzana` | Placa a 1 o 2 unidades, letra de vía o cuadrante en un solo lado, dirección compartida dentro de una manzana, o `gate_fallback`. | La manzana (aproximadamente). |
| `predio` | Vía, cruce y placa confirmados sin aproximación. | El predio, sujeto a la fiabilidad real medida (ver [08](08-metricas-y-confiabilidad.md)). |

Una fila con cualquier aproximación nunca sale como `predio`.

## 6. `confianza`, `confiabilidad` y `margen`

| Campo | Qué es | Qué no es |
|-------|--------|-----------|
| `confianza` | `0.7 sim + 0.1 fuzz + 0.1 struct + 0.1 geo`. Similitud del candidato con la entrada. | **No es una probabilidad.** Una banda alta de `confianza` no garantiza que el predio sea exacto. |
| `confiabilidad` | Probabilidad estimada de que el NPN sea correcto, a partir de `confianza` y del tipo de fila. | Con el calibrador servido (v4), discrimina poco entre filas `OK` (ajustado con etiquetas GPS ruidosas). Tiende a subestimar la precisión real de las filas cuyo texto coincide con el catastro. Ver [08](08-metricas-y-confiabilidad.md), sección 4. |
| `confiabilidad_manzana` | Ídem para la manzana. | Discrimina algo mejor que la del predio, pero débil; verifíquela con su propia muestra. |
| `margen` | Cuán cerca está el mejor competidor en otra manzana. Menor es más ambiguo. | No es un error estimado. |

Las filas `OK` con `margen` pequeño (por ejemplo `< 0.05`) tienden a acertar la manzana con mucha menos frecuencia que el resto; mida el efecto con su propia muestra. Con el valor por defecto de `ambiguity_delta`, las filas con `margen < 0.02` ya pasan a `SIN_MATCH`.

## 7. Flujo de decisión resumido

El diagrama completo está en [02-arquitectura.md](02-arquitectura.md), sección 4. En orden:

1. Entrada vacía o marcador: `SIN_MATCH`.
2. Detección de lugar y limpieza del texto.
3. Análisis gramatical; si falla: `NO_PARSEABLE`.
4. Recuperación de hasta 40 candidatos y puntaje fusionado.
5. Puerta geográfica por el lugar detectado; si no quedan candidatos: `SIN_MATCH`.
6. Mejor candidato; si `confianza < 0.73`: `SIN_MATCH`.
7. Reglas estrictas; si hay violación dura o más de una blanda: `SIN_MATCH`.
8. Ambigüedad; si `margen < 0.02`: `SIN_MATCH`.
9. Si todo pasa: `OK`, con nivel de precisión, notas y confiabilidad.

## 8. Guía de interpretación con ejemplos

Todos los ejemplos son inventados. Solo se muestran las columnas relevantes.

### 8.1 `OK` con nivel `predio`

| Campo | Valor |
|-------|-------|
| Entrada | `Carrera 12 # 34-56` |
| `estado` | `OK` |
| `motivo` | (vacío) |
| `direccion_normalizada` | `KR 12 # 34 - 56` |
| `nivel_precision` | `predio` |
| `confianza` | 0.94 |
| `confiabilidad` | 0.64 |
| `margen` | 0.12 |

Lectura: vía, cruce y placa coinciden y no hay competidor cercano. Es la mejor categoría, pero no es certeza: incluso entre filas con referencia coherente con el texto, el acierto del predio es mayor que sobre todas las filas `OK` con referencia, pero sigue sin ser total (ver [08](08-metricas-y-confiabilidad.md)). Verificar en decisiones de consecuencia.

### 8.2 `OK` con aproximación

| Campo | Valor |
|-------|-------|
| Entrada | `Carrera 12 # 34-58` (la placa catastral es 56) |
| `estado` | `OK` |
| `motivo` | `aproximado: placa 58 != 56` |
| `nivel_precision` | `manzana` |

Lectura: la manzana es probable; el predio concreto no está respaldado.

### 8.3 `OK` por dirección compartida

| Campo | Valor |
|-------|-------|
| Entrada | `Calle 7 # 20-10` |
| `estado` | `OK` |
| `motivo` | `direccion compartida por 12 predios` |
| `nivel_precision` | `manzana` o `direccion` |

Lectura: varios predios (por ejemplo, de una propiedad horizontal) comparten la dirección; se devuelve el primero por `OBJECTID`. No identifica la unidad.

### 8.4 `SIN_MATCH` por rechazo de regla

| Campo | Valor |
|-------|-------|
| Entrada | `Carrera 31 F # 84-15` |
| `estado` | `SIN_MATCH` |
| `motivo` | `numero de sufijo de via - != 3; regla: sufijo unilateral` |
| `direccion_normalizada` | `KR 31 F # 84 - 15` (forma por reglas) |
| `fuente_normalizacion` | `reglas` |

Lectura: el mejor candidato tiene sufijo numérico de vía y la entrada no. Es una calle distinta en el papel, pero una parte apreciable de las filas rechazadas por esa sola causa son el lote correcto. Es un buen conjunto para revisión manual.

### 8.5 `SIN_MATCH` por umbral

| Campo | Valor |
|-------|-------|
| Entrada | `Calle 99 # 1-1` (calle inexistente en el índice) |
| `estado` | `SIN_MATCH` |
| `motivo` | `confianza 0.58 < umbral 0.73` |

Lectura: nada se parece lo suficiente. La dirección puede no existir, no estar en el índice (por ejemplo, zona rural) o estar muy deformada.

### 8.6 `SIN_MATCH` por ambigüedad

| Campo | Valor |
|-------|-------|
| Entrada | `Calle 9 # 41-26` (se repite en distintos sectores) |
| `estado` | `SIN_MATCH` |
| `motivo` | `ambiguo: 2 candidatos cercanos en otras manzanas` |

Lectura: dos predios en manzanas distintas puntúan casi igual. Ayuda añadir el barrio o la comuna al texto, para que la puerta geográfica desempate.

### 8.7 `SIN_MATCH` por puerta geográfica

| Campo | Valor |
|-------|-------|
| Entrada | `Calle 9 # 41-26 barrio Ejemplo` |
| `motivo` | `candidatos fuera de barrio Ejemplo` |

Lectura: el texto menciona un barrio pero los candidatos están fuera de él y de su holgura. Puede ser un error del barrio escrito.

### 8.8 `NO_PARSEABLE`

| Campo | Valor |
|-------|-------|
| Entrada | `esto no es una direccion` |
| `estado` | `NO_PARSEABLE` |
| `motivo` | `implicit_hash; via_unparsed` |
| `barrio_vereda`, `comuna_corregimiento` | llenos si el texto menciona un lugar |

Lectura: no hay estructura de dirección. No se intentó coincidencia.

### 8.9 `FUERA_DE_AREA`

| Campo | Valor |
|-------|-------|
| Entrada | `Calle 9 # 41-26` con la columna de municipio `Municipio de ejemplo` |
| `estado` | `FUERA_DE_AREA` |
| `motivo` | `municipio distinto de Cali: 'Municipio de ejemplo'` |

### 8.10 `ERROR`

| Campo | Valor |
|-------|-------|
| `estado` | `ERROR` |
| `motivo` | `<TipoDeExcepcion>: <mensaje corto>` |

## 9. Lectura recomendada para quien consume los resultados

1. Mire primero `estado` y `nivel_precision`. Descarte lo que no sea `OK`.
2. Para uso con el predio, exija `nivel_precision == predio`.
3. Use `confiabilidad` como señal secundaria, no como probabilidad exacta: su capacidad de ordenar filas es muy limitada. La regla operativa y sus límites están en [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md), sección 5.
4. Lea `motivo` en filas `OK`: las notas `aproximado`, `compartida` y `NPN` cambian el sentido del resultado.
5. Revise manualmente las filas con `margen` bajo y las `SIN_MATCH` con `regla: sufijo unilateral`.
6. Con `position_tier` y `catalog_status` sepa **qué tan bien ubicada** quedó la fila y si la dirección existe tal cual en el catastro (`address_verified`); `fidelity_risk` alto indica que la dirección devuelta puede ser otra que la recibida (sección 10).
7. No use las coordenadas para validar un cruce: un punto cercano no confirma el predio, y un punto lejano puede ser un centroide legítimo de una dirección compartida.

## 10. Posición, fidelidad y estado de catálogo

Estas diez columnas (`position_tier`, `position_radius_m`, `position_source`, `selected_doc_id`, `override_rejected`, `direccion_catastral`, `fidelity_risk`, `fidelity_flags`, `catalog_status`, `address_verified`) se añaden **al final** de la salida, de modo que ninguna columna anterior cambia de lugar. No alteran `estado`. Todos los ejemplos son inventados.

### 10.1 Niveles de posición (`position_tier`) y radios

El nivel dice qué tan bien se pudo ubicar la fila, y `position_radius_m` el error geométrico esperado. Los radios son **geométricos**: se derivaron del propio catastro (p90), sin usar GPS, y por eso no son una medida de acierto frente a la realidad de campo. `nivel_precision` conserva su vocabulario y se deriva del nivel de posición.

| `position_tier` | Cuándo | `position_radius_m` | Coordenada | NPN | `nivel_precision` |
|-----------------|--------|--------------------:|------------|-----|-------------------|
| `predio` | La base de la entrada es idéntica a la del catastro. | 15 | Centroide del predio (`position_source = parcel`). | Se conserva. | `predio` |
| `lote` | Una unidad de sitio (torre, bloque, casa, lote...) que **ambos** lados declaran difiere: el lote es correcto, la unidad no. | 44 | Centroide del lote. | Retenido (`null`). | `manzana` |
| `tramo` | Placa a 1 o 2 unidades, o letras o `BIS` de la placa distintos: el tramo de calle es correcto, la puerta no. | 15 a 23 según la brecha entre los predios vecinos (piso 15 m, tope 23 m) | Interpolada entre los dos predios más cercanos del mismo costado del par de vías (`position_source = interp`). | Retenido (`null`). | `manzana` |
| `manzana` | Letra de vía o cuadrante en un solo lado, dirección compartida o hallazgos suaves. | 56 | Punto del mejor candidato (aproximado). | Retenido si hay letra de vía o cuadrante en un solo lado; en los demás casos se conserva. | `manzana` |
| `via` | Placa más allá del último predio conocido de su costado, parse truncado o riesgo de fidelidad alto. | 21 (10 a 56 si hay extrapolación, según la distancia al último predio) | **Retenida** (`null`) salvo que el servicio active `EXTRAPOLATED_COORDS_ENABLED` (por defecto `False`). | `null`; también la manzana. | `via` |
| `none` | No hubo posición (fila no `OK`). | `null` | `null` | `null` | `null` |

Detalles verificables en el código (`service.POSITION_RADIUS_M`, `interp.TRAMO_RADIUS_STEPS`, `interp.EXTRAP_RADIUS_STEPS`):

- La interpolación no se usa si las placas distan más de 60 unidades o los predios vecinos más de 50 m, ni se extrapola más de 25 m desde el último predio.
- El radio de una fila `tramo` es una función escalonada de la brecha real entre los predios que la flanquean; el de una fila `via` extrapolada, de su distancia al último predio.
- Cuando una guarda o un riesgo de fidelidad limitan el nivel, el nivel nunca mejora, solo empeora (`predio` > `lote` > `tramo` > `manzana` > `via` > `none`).
- Una letra de vía presente en un solo lado queda limitada a `manzana` (antes `esquina`) y `motivo` lo explica.

### 10.2 Dirección en blanco (reglas de emisión)

`direccion_normalizada` es el texto que el sistema afirma que corresponde a la entrada. Cuando ese texto no es confiable se **deja en blanco** (`null`, igual que `fuente_normalizacion`) en lugar de emitir una dirección a medias; la canalización de archivos trata una dirección normalizada vacía como "no normalizada" y recurre al texto original. Se deja en blanco en estos casos:

1. `catalog_status = unparsed` (sin dirección, gramática no reconocida o parse truncado): nunca se emite.
2. `catalog_status = below_cutoff` o `rejected_mismatch` con un parse inseguro: solo se emite si el riesgo de fidelidad es `none` o `low` **y** el parse está completo (tipo y número de vía, número de cruce, placa, sin truncar).
3. **Toda** fila con `fidelity_risk = high`, también las `OK`: el texto de reglas perdió o inventó dígitos de puerta, letras o cuadrante, es decir, es texto corrupto. En una fila `OK` que se deja en blanco, `motivo` añade `fidelidad alta: direccion no emitida`.

`estado`, `motivo`, `catalog_status`, `position_tier` y `direccion_catastral` no cambian al dejar en blanco. Las filas `OK` con riesgo de fidelidad distinto de `high` conservan su texto.

### 10.3 Medida de fidelidad (`fidelity_risk`, `fidelity_flags`)

`fidelity_risk` responde una sola pregunta: **¿el sistema devolvió una dirección distinta de la que recibió?** Compara la entrada original con `direccion_normalizada` con un tokenizador propio, independiente del analizador de producción (para poder detectar errores del propio analizador), del catastro y de cualquier coordenada. Son inofensivos y no cuentan: abreviaturas, espacios, mayúsculas, tildes, insertar `#` o `-`, `No.`/`N°` a `#`, `N` a `NORTE` dentro del mismo segmento, quitar el sufijo de ciudad y los ceros a la izquierda. Sí cuentan los datos que identifican la puerta: tipo de vía, números de vía, cruce y placa, las letras pegadas a ellos y el cuadrante (**su posición es significativa**).

| `fidelity_risk` | Qué indica | Efecto en la fila `OK` |
|-----------------|------------|------------------------|
| `high` | Cambio, pérdida, reordenamiento o invención de números de puerta, letras, cuadrante o tipo de vía; placa de otro predio; salida sin entrada que la respalde. | `position_tier` pasa como máximo a `via` (NPN, manzana y coordenadas retenidos); la dirección se deja en blanco (sección 10.2). |
| `medium` | Dígitos de unidad perdidos (`APTO 301`), segunda dirección truncada, 30 % o más de los tokens significativos perdidos, `#` movido. | `position_tier` como máximo `manzana` (sin NPN). |
| `low` | Unidad renombrada, números de anotación (comuna 3, etapa 2) o pocos tokens perdidos, ceros de relleno. | Sin cambio de nivel. |
| `none` | Nada cambió más allá de la normalización inofensiva, o no hay salida (`no_address`). | Sin cambio. |

Las banderas (`fidelity_flags`) nombran el hallazgo, por ejemplo `plate_changed`, `door_digits_dropped`, `letters_dropped`, `quadrant_moved`, `trailing_number_dropped`, `multi_address_truncated`. La política de límites se puede apagar con `service.FIDELITY_CAPS_ENABLED = False` (las columnas se conservan y los niveles no se tocan). `confiabilidad` conserva su valor calibrado; **no** incorpora el riesgo de fidelidad.

### 10.4 Estado de catálogo (`catalog_status`, `address_verified`)

`catalog_status` dice qué puede afirmar con honestidad el sistema sobre la fila. Se decide por precedencia (gana la primera regla que se cumple):

1. `unparsed`: sin dirección, gramática no reconocida o parse **truncado** (la base analizada no es la dirección).
2. `verified` / `ambiguous`: fila `OK` cuya base es igual a la del catastro (guarda exacta, sin límites ni hallazgos suaves) con riesgo de fidelidad `none` o `low`; `ambiguous` si varios predios comparten la dirección.
3. `door_not_catalogued`: la base se analiza completa pero su clave exacta (vía, cruce, placa, cuadrantes por posición, letras, `BIS`) no existe en el catastro.
4. `below_cutoff`: fila `SIN_MATCH` bajo el umbral de confianza (o sin candidatos).
5. `ambiguous`: fila `SIN_MATCH` que se abstuvo porque compiten predios de manzanas distintas.
6. `rejected_mismatch`: cualquier otro resultado no verificado (rechazo por guarda o por regla, límites de nivel, conflictos de unidad, riesgo de fidelidad medio o alto).

`address_verified` es `True` solo para `verified`. **Una placa distinta nunca es una puerta verificada**: una diferencia de placa (aunque sea de una unidad) produce como máximo `tramo`, nunca `verified`. La probabilidad calibrada de `confiabilidad` no se expone aquí a propósito, porque solo es válida bajo la mezcla sintética con la que se ajustó.

### 10.5 Reglas de la guarda estructural

La guarda compara el análisis de la entrada con el del candidato catastral elegido antes de aceptarlo (`inference.base_mismatches(..., mode="position")`). Reglas que conviene conocer:

- **El cuadrante se compara por posición** (vía, cruce, final tras la placa): un cuadrante en la vía y otro en el cruce son lugares distintos, y un cuadrante que se movió de posición se rechaza (`base distinta al catastro: quadrant_moved`).
- Una letra cardinal equivale al cuadrante deletreado de la **misma** posición (`N` = `NORTE`), nunca al de otra.
- Una placa ausente en un lado es una diferencia; las unidades de edificio (`AP`) no se comparan, pero sí las que identifican el sitio (torre, bloque, casa, lote).
- Un sufijo numérico de vía o cruce que difiere o aparece en un solo lado (`KR 26 H` frente a `KR 26 H 3`) es un rechazo duro, con un costo que conviene medir (ver [08](08-metricas-y-confiabilidad.md)): `motivo` termina con `regla: sufijo unilateral` para poder auditar o recuperar esos casos.
- Un parse truncado nunca se verifica: la guarda devuelve `truncated_parse` para todo candidato.

### 10.6 Cachés construidas de forma perezosa

Dos índices se construyen la primera vez que se usan y se guardan junto a los artefactos; **no se versionan** y se regeneran solos si faltan:

| Archivo | Para qué | Tamaño aprox. | Costo |
|---------|----------|---------------|-------|
| `artifacts/interp_index_v1.pkl` + `interp_index_v1.json` | Índice de interpolación de posiciones (niveles `tramo` y `via`). | 8 MB | lenta en frío, rápida en caliente |
| `artifacts/catalog_keys_v1.pkl` + `catalog_keys_v1.json` | Conjunto de claves de base exactas del catastro (`door_not_catalogued`). | 6 MB | se construye una vez (no se midió el tiempo) |

El archivo `.json` (hash de la fuente y SHA-256 del `.pkl`) se verifica **antes** de leer el `.pkl`; si falta, está corrupto o no coincide, la caché se ignora y se reconstruye. Si el directorio de artefactos no es escribible solo se emite una advertencia y el índice queda en memoria. Para construirlas de antemano (por ejemplo, antes de crear la imagen), ejecute `python deploy/prepare_artifacts.py` (`--no-warm` las omite). Como son pickles, valen las mismas advertencias de seguridad de la sección 9 de [05](05-uso-e-implementacion.md): no cargue cachés de origen no confiable.
