# API del normalizador de direcciones de Cali

Servicio HTTP que normaliza direcciones de Santiago de Cali contra el catastro
(catastro urbano indexado), devuelve el numero predial nacional, la manzana,
la coordenada del centroide del predio y el barrio/vereda y comuna/corregimiento
oficiales del IDESC.

El servicio expone tambien `/docs` (Swagger UI generado por FastAPI, con todos
los esquemas de peticion y respuesta) y `/docs-integracion`, que es esta misma
pagina renderizada en HTML.

---

## Contrato de estado

Cada fila de salida trae un `estado` con exactamente uno de tres valores:

| `estado` | Significado |
| --- | --- |
| `OK` | La direccion se parseo, coincidio con un predio por encima del umbral de confianza calibrado, el predio cae dentro del lugar detectado en el texto y la direccion catastral concuerda estructuralmente con la entrada. |
| `SIN_MATCH` | No se puede asignar ningun predio. El campo `motivo` dice por que: dato vacio o marcador sin informacion, confianza por debajo del umbral, una regla estructural violada por el mejor candidato, o todos los candidatos fuera del barrio/comuna/corregimiento detectado. |
| `NO_PARSEABLE` | La gramatica de direcciones no reconocio el texto. Las columnas de zona se llenan igual si se detecto un lugar. |

**Solo las filas `OK` traen `numero_predial_nacional`, `manzana`, `lat`, `lon`,
`confianza`, `confiabilidad`, `confiabilidad_manzana` y `margen`.** Una fila que no pasa el umbral, la reja geografica o las reglas
estructurales no recibe candidato alguno. Esto es deliberado: evita que una
conjetura de baja calidad se confunda con una coincidencia real. El motivo exacto
del rechazo siempre queda en `motivo`.

**Confiabilidad.** `confiabilidad` y `confiabilidad_manzana` son probabilidades
calibradas (regresion logistica pequena, `artifacts/reliability.json`, ajustada con
`scripts/fit_reliability.py`) de que el predio, respectivamente la manzana, entregados
sean los correctos. Son **solo informativas**: el `estado` no depende de ellas. Se
estimaron sobre un conjunto de validacion pequeno (~530 filas con verdad terreno) y las
etiquetas de predio son ruidosas (verdad terreno por GPS; predios vecinos a ~6 m), por lo
que `confiabilidad` tiende a **subestimar** la probabilidad real y discrimina poco entre
filas `OK`. Si `reliability.json` no esta disponible, ambas salen `null` (nunca se
sustituyen por `confianza`). `reliability.json` admite dos formatos (campo `kind`): `logistic` (el actual; sin `kind` se trata como `logistic`) y `cells` (tabla de tasas por tipo de fila y banda de `margen`, suavizadas hacia la tasa global; ver `scripts/fit_reliability.py --kind`).

`fuente_normalizacion` indica de donde sale `direccion_normalizada`:

- `catastro` — el registro catastral que coincidio (solo en filas `OK`);
- `reglas` — la forma canonica producida por la gramatica, en filas parseables
  pero sin coincidencia;
- `null` — no hubo nada que normalizar.

### Columnas de salida

| Columna | Tipo | Descripcion |
| --- | --- | --- |
| `direccion_entrada` | texto | El texto original recibido. |
| `estado` | texto | `OK`, `SIN_MATCH` o `NO_PARSEABLE`. |
| `motivo` | texto | Razon del rechazo. En filas `OK` es vacio o una o varias notas separadas por `; `: `aproximado: <regla>` (letra de via presente en un solo lado, placa a 1-2 unidades o cuadrante presente en un solo lado; ninguna es `predio`); `nivel maximo <nivel>: <razon>` (el techo de `nivel_precision` que impone esa aproximacion); `direccion compartida por N predios` (se reporta el primer predio por OBJECTID); `coordenada: centroide de la direccion compartida por N predios` (solo si los artefactos no traen el centroide del predio reportado, ver `lat` / `lon`); `NPN nulo en catastro`; `NPN anomalo en catastro (<por que>)` (no tiene 30 digitos o trae caracteres no numericos; no se descarta). Un sufijo numerico de via o cruce distinto o presente en un solo lado (`KR 31 F` frente a `KR 31 F 3`) es otra calle: rechaza la fila con `numero de sufijo de via - != 3`; cuando el unico motivo es un sufijo presente en un solo lado, `motivo` termina con la marca estable `regla: sufijo unilateral` (filtrable: ver el costo medido mas abajo). Un cuadrante distinto en campos distintos (`CL 9 NORTE # 41-26` frente a `CL 9 # 41 OESTE - 26`) tambien rechaza: `cuadrante NORTE != OESTE`. |
| `nivel_precision` | texto o null | `predio`, `via`, `esquina` (sin placa), `manzana` (placa a 1-2 unidades, letra de via o cuadrante presente en un solo lado, o direccion compartida) o `direccion` (compartida entre manzanas, o NPN nulo); null si no es `OK`. Se deriva del nivel de posicion (`position_tier`). Una fila con cualquiera de esas aproximaciones nunca sale como `predio`. |
| `direccion_normalizada` | texto o null | Base de la direccion (`<via> # <cruce> - <placa>`) catastral o por reglas, seguida de `complemento_normalizado` si existe. El complemento escrito en el catastro no se copia: solo se emite el de la entrada. |
| `fuente_normalizacion` | texto o null | `catastro`, `reglas` o null. |
| `complemento_normalizado` | texto o null | Parte de unidad de `direccion_normalizada` (por ejemplo `TO F AP 101`), tomada de la **entrada** y escrita en forma canonica (`TORRE`/`T-` pasan a `TO`, orden fijo de tipos). Solo se emiten unidades reconocidas (torre, bloque, apto, piso, casa...) con un codigo corto o un nombre, tambien cuando vienen tras una coma o un punto y coma (`CL 9 # 41-26, Torre B` -> `TO B`; un barrio, la ciudad o texto libre tras la coma se ignoran); `TORRE O` y `T B` (una `T` suelta seguida de una letra o un numero corto) se reconocen como torre; el texto libre (`FRENTE A LA TIENDA`, `CASA ESQUINERA`, `TORRE BLANCA`, `EDIFICIO DE TRES PISOS`) se descarta. Barrio, vereda y corregimiento (`BR`, `VDA`, `CGTO`) se reconocen pero **no** se emiten: no identifican un edificio. Un valor con varias partes se conserva entero y con una sola grafia: las partes de distinta clase se pegan (`101 A`, `101-A` -> `101A`; `F-2` -> `F2`) y las de la misma clase se unen con guion (`3 02`, `3 - 02` -> `3-02`, distinto de `302`); el primer numero pierde los ceros a la izquierda (`0101` -> `101`). Las palabras marcador antes del valor se omiten (`TORRE N 3` -> `TO 3`, `TORRE LETRA A` -> `TO A`) y `TORRE UNO A` es `TO 1A`. Es exactamente la misma cadena en filas `OK` y no `OK` para la misma entrada (aunque el catastro escriba `TO 02` o `AP 0101`); `null` si la entrada no trae unidades. En filas `OK`, si la entrada da para un tipo un valor que el catastro no tiene, gana la entrada y `motivo` incluye `complemento distinto al catastro: <tipos>`; repetir el valor del catastro con texto libre alrededor no es un conflicto. En el endpoint ESRI **no** se expone: ahi solo viaja `direccion_normalizada`, que ya la incluye. |
| `position_tier` | texto | Que tan bien ubicada quedo la fila: `predio` (centroide del predio, radio 15 m), `lote` (centroide del lote; unidad distinta al catastro, sin NPN, 44 m), `tramo` (coordenada interpolada entre predios vecinos del mismo costado, 15 a 23 m segun la brecha), `manzana` (punto del mejor candidato, aproximado, 56 m), `via` (extrapolacion o parse truncado: coordenadas, NPN y manzana retenidos salvo que el servicio active `EXTRAPOLATED_COORDS_ENABLED`; 21 m) o `none`. |
| `position_radius_m` | numero o null | Radio de error geometrico (p90, derivado del catastro, sin GPS; metros) del nivel de posicion; null si `none`. |
| `position_source` | texto | Origen de la coordenada: `parcel` (centroide de un predio), `interp` (interpolada o extrapolada) o `none`. |
| `selected_doc_id` | entero o null | Indice del documento catastral elegido; null si la fila no es `OK`. |
| `override_rejected` | booleano | `true` si el candidato recuperado fue rechazado por las reglas estructurales (la base catastral difiere de la direccion de entrada). |
| `direccion_catastral` | texto o null | Direccion del predio coincidente tal como la escribe el catastro (solo filas `OK`). `direccion_normalizada` es la base catastral solo en los niveles `predio` y `lote`; en `tramo`, `via` y filas aproximadas es el texto normalizado de la consulta. |
| `fidelity_risk` | texto | Riesgo de que `direccion_normalizada` sea OTRA direccion que la de entrada (deriva semantica, no de caracteres): `high`, `medium`, `low` o `none`. `high` limita `position_tier` a `via` y `medium` a `manzana` (sin NPN); `high` ademas deja `direccion_normalizada` en blanco. |
| `fidelity_flags` | texto | Banderas de `fidelity_risk` separadas por `;` (por ejemplo `plate_changed;letters_dropped`); `no_address` si no hay direccion de salida; vacio si no hay hallazgos. |
| `catalog_status` | texto | Estado frente al catastro, por precedencia: `unparsed` (sin direccion, gramatica no reconocida o parse truncado), `verified` (la base coincide exactamente con el catastro y el riesgo de fidelidad es none/low), `ambiguous` (varios predios comparten la direccion, o la fila se abstuvo por competencia entre manzanas), `door_not_catalogued` (la base es valida pero no existe en el catastro), `below_cutoff` (bajo el umbral de confianza) o `rejected_mismatch` (cualquier otro rechazo). |
| `address_verified` | booleano | `true` exactamente cuando `catalog_status` es `verified`. |
| `numero_predial_nacional` | texto o null | 30 digitos del predio; `null` si el catastro no trae NPN para la direccion (nunca el texto `None`). |
| `manzana` | texto o null | Codigo de manzana catastral; `null` si el catastro no la trae. |
| `lat` | numero o null | Latitud (EPSG:4326). Es el centroide **del predio cuyo NPN se reporta** (el primer predio por OBJECTID de la direccion) cuando los artefactos traen `catastro_own_centroids.parquet` (va en la imagen Docker) o `catastro.parquet`; en una direccion de un solo predio coincide con el centroide de la direccion. Si no hay ninguno de los dos archivos, o ese predio no tiene centroide, es el centroide de la **direccion** (mediana de todos los predios que la comparten, a veces a mas de 50 m del reportado) y `motivo` lo dice con `coordenada: centroide de la direccion compartida por N predios`. |
| `lon` | numero o null | Longitud (EPSG:4326), con la misma regla que `lat`. |
| `confianza` | numero o null | Puntaje fusionado 0-1, **sin calibrar** (no es una probabilidad: una `confianza` alta no implica que el predio sea exacto). |
| `confiabilidad` | numero o null | Probabilidad **calibrada** (0-1) de que `numero_predial_nacional` sea el predio correcto. |
| `confiabilidad_manzana` | numero o null | Probabilidad calibrada (0-1) de que `manzana` sea la correcta; siempre `>= confiabilidad`. |
| `margen` | numero o null | Diferencia de puntaje fusionado entre el predio entregado y el mejor candidato alternativo que tambien pasa las reglas y esta en **otra manzana**. Cuanto menor, mas ambigua la manzana; `null` si no hay competidor o la fila no es `OK`. Con el `ambiguity_delta` por defecto (0.02) las filas con `margen < 0.02` pasan a `SIN_MATCH`; con `ambiguity_delta=0` es solo informativo. Las filas `OK` con `margen` pequeno acierta la manzana con menos frecuencia que el resto; mida el efecto con su propia muestra. |
| `barrio_vereda` | texto o null | Nombre oficial IDESC del barrio o de la vereda. |
| `comuna_corregimiento` | texto o null | `Comuna 19` o el nombre del corregimiento. |

### Regla recomendada para un NPN confiable

Regla de uso sugerida (los resultados de las corridas no se publican; valide el corte con su propia muestra,
ver `documentacion/08-metricas-y-confiabilidad.md`):

```
estado == "OK"  y  nivel_precision == "predio"  y  confiabilidad >= 0.58
```

El corte `0.58` corresponde a la escala del `reliability.json` servido (v4). Si se sirve el v5
(`artifacts/reliability_v2.json`), la escala cambia y el equivalente elegido en desarrollo es `>= 0.81`
(ver `documentacion/08-metricas-y-confiabilidad.md`, secciones 4 y 5).

No suba `confiabilidad` por encima de ~0.62 sin validarlo: la relacion con el acierto no es monotona. El GPS de campo cae en la fachada y no es el lote exacto, por lo que un acierto medido contra GPS es un piso, no la precision real.

### Costos de las reglas duras

- **Sufijo numerico presente en un solo lado.** Se mantiene como rechazo duro (otra calle en el papel). Costo: parte de las
  filas que eran `OK` antes de la regla y ahora se rechazan por un sufijo en un solo lado eran el lote correcto
  (mida la fraccion con su propia muestra). Las filas rechazadas
  por esa sola razon llevan `regla: sufijo unilateral` en `motivo` para auditarlas. Un candidato que solo falla esa regla
  **si cuenta** como competidor de ambiguedad (`margen`, `ambiguo: ...`); uno con sufijos distintos en ambos lados no.
- **`nivel_precision == esquina`** significa "sin placa". Una letra de via presente en un solo lado queda en `manzana` y `motivo` lo explica.

### Reproducibilidad

Las columnas de `normalize_strict`, la CLI y la API no dependen del tamano del lote: el codificador de consultas y la cabeza de
coordenadas corren en fp32 y la similitud de cada candidato se recalcula en fp32 (misma entrada dos veces: sin diferencias). `normalize_batch` (usado por `scripts/ablation.py` y `smoke_eval.py`) conserva el camino bf16 de
`encode_queries` / `search`; la independencia del tamano del lote vale para `score_components` / `normalize_strict`, no para ese camino.

---

## Parametros del motor

Todos los endpoints de normalizacion aceptan los mismos parametros. Los valores
por omision son los calibrados sobre la particion de validacion catastral
retenida; cambielos solo con una razon medida. La CLI, la API, el pipeline de
archivos y la funcion `normalize_strict` de la libreria comparten los mismos
valores por omision de la reja (`CLI_BARRIO_BUFFER_M` y `DEFAULT_GATE_ESCALATE`
en `cali_address.service`): antes la libreria usaba 200 m y sin escalamiento.

| Parametro | Omision | Que hace |
| --- | --- | --- |
| `min_struct` | `0.6` | Coincidencia estructural minima (0-1) exigida al candidato. Por debajo, la fila pasa a `SIN_MATCH` con el motivo `coincidencia estructural X < Y`. |
| `plate_tolerance` | `0` | Diferencia maxima aceptada entre la placa de entrada y la del predio, entre `0` y `2`. `0` exige placa exacta; valores mas altos no se aceptan porque amplian falsos OK. Dentro de la tolerancia no hay violacion (nivel `predio`); una diferencia de 1-2 unidades por encima de ella se acepta como `aproximado` (nivel `manzana`). |
| `barrio_buffer` | `1000` | Holgura en metros alrededor del barrio detectado en el texto. |
| `zone_buffer` | `300` | Holgura en metros alrededor de la comuna o corregimiento detectado. |
| `gate_escalate` | `true` | Si ningun candidato queda dentro del barrio, ampliar a la comuna o corregimiento en vez de descartar la fila. |
| `ambiguity_delta` | `0.02` | Abstencion por ambiguedad, entre `0` y `0.2`. `0` la desactiva (los estados no cambian). Si es mayor que 0, una fila `OK` cuyo `margen` sea **menor** que este valor pasa a `SIN_MATCH` con el motivo `ambiguo: N candidatos cercanos en otras manzanas`. Reduce las `OK` a cambio de quitar filas con mayor probabilidad de error; una fila sin competidor (`margen` `null`) nunca se abstiene. En la CLI es `--ambiguity-delta` (por defecto 0.02). |
| `gazetteer` | `true` | Usar el gazetteer IDESC para limpiar el texto (`Barrio Siloé calle 1 # 2-7` no parsea; `calle 1 # 2-7` si) y aplicar la reja geografica. |

El umbral de confianza es **`0.73`**, calibrado y leido de `artifacts/tuning.json`;
no es un parametro de la peticion, por lo que una integracion no puede relajarlo
por accidente. Se puede consultar en `/health`.

Por que `barrio_buffer` vale 1000 y no 200: los nombres de barrio de las fuentes con GPS de campo vienen de un
geocodificador inverso cuyas extensiones coloquiales no coinciden con los poligonos oficiales del IDESC, por lo que el
barrio del texto concuerda con el poligono del predio mucho menos que la comuna. Una reja de 200 m descartaba demasiadas
filas validas; con 1000 m mas escalamiento se conservan mas filas `OK` sin aumentar la proporcion de errores grandes.
Los datos de esa medicion no se publican; repita la comparacion con su propia fuente antes de cambiar el valor.

---

## `GET /health`

Estado del servicio. Devuelve **503** mientras el modelo se esta cargando o si la
carga fallo, y **200** cuando el servicio puede atender. Es el endpoint que debe
apuntar el health check de la plataforma.

```bash
curl -s https://<app>.up.railway.app/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "catastro_size": 330387,
  "gazetteer_loaded": true,
  "threshold": 0.73,
  "detail": null
}
```

Mientras carga (503):

```json
{
  "status": "loading",
  "model_loaded": false,
  "device": "cpu",
  "catastro_size": 0,
  "gazetteer_loaded": false,
  "threshold": null,
  "detail": "cargando modelo e indice catastral"
}
```

---

## `POST /api/v1/inspect`

Lee un archivo **sin normalizarlo** y responde con lo que hace falta saber para
la siguiente llamada: formato detectado, columnas, columna de direccion sugerida,
todas las columnas puntuadas y una muestra de hasta 5 filas.

Este endpoint nunca falla por ambiguedad de columnas. Es exactamente el paso con
el que un cliente (la interfaz web o un sistema externo) descubre que enviar
despues.

**Cuerpo:** `multipart/form-data`

| Campo | Requerido | Descripcion |
| --- | --- | --- |
| `file` | si | El archivo. |
| `sheet` | no | Hoja a leer, cuando es un libro de Excel. |
| `header` | no | Fila de encabezado 0-based. Si se omite, se infiere. |

### Formatos aceptados

- `.xlsx`, `.xls` — se lee con `pandas.read_excel`. Si no se envia `header`, se
  detecta la fila de encabezado: se toma la fila con mas celdas no vacias y se
  desempata a favor de la que parezca una fila de rotulos cortos en vez de datos.
  Esto resuelve los exportes de ArcGIS de este proyecto, que traen 2 a 5 filas de
  metadatos antes del encabezado real.
- `.csv` — se intenta UTF-8 y luego latin-1; el delimitador (`,` o `;`) se
  detecta contando ocurrencias en la primera linea no vacia.
- `.geojson`, `.json` — debe ser un `FeatureCollection`. Las `properties` de cada
  feature se vuelven columnas y el centroide de la geometria se agrega como
  `_lon` y `_lat`.
- `.zip` con un shapefile — debe contener `.shp`, `.shx` y `.dbf`; el `.prj` es
  opcional y, si esta, se usa para reproyectar los centroides a EPSG:4326. Sin
  `.prj` se asume EPSG:4326 y se avisa en `warnings`. Un `.shp` suelto se rechaza
  con un mensaje que pide comprimir los tres (o cuatro) archivos juntos.

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/inspect \
  -F "file=@archivo.xlsx"
```

```json
{
  "format": "xlsx",
  "row_count": 2218,
  "columns": ["id_edan", "ObjectID", "direccion", "direccion_norm", "barrio"],
  "suggested_column": "direccion",
  "candidates": [
    {"column": "direccion", "score": 100.0},
    {"column": "direccion_norm", "score": 95.0},
    {"column": "barrio", "score": 40.0},
    {"column": "ObjectID", "score": 33.3},
    {"column": "id_edan", "score": 30.0}
  ],
  "preview": [{"id_edan": "XXXXX", "direccion": "KR 99 BIS # 9B2 99", "barrio": "Barrio de ejemplo"}],
  "warnings": ["fila de encabezado detectada automaticamente: 2"],
  "header_row": 2,
  "sheets": ["Hoja1"]
}
```

`candidates` esta ordenado de mayor a menor puntaje e incluye **todas** las
columnas, de modo que sirve directamente para poblar un desplegable.
`suggested_column` es `null` cuando ninguna columna alcanza un puntaje plausible:
en ese caso la interfaz debe pedirle al usuario que elija.

---

## `POST /api/v1/normalize`

Normaliza un archivo completo.

**Cuerpo:** `multipart/form-data`

| Campo | Omision | Descripcion |
| --- | --- | --- |
| `file` | — | El archivo. |
| `sheet` | primera hoja | Hoja de Excel. |
| `header` | inferido | Fila de encabezado 0-based. |
| `address_column` | detectada | Columna de direccion. |
| `output_format` | `json` | `json`, `xlsx`, `csv` o `geojson`. |
| `min_struct`, `plate_tolerance`, `barrio_buffer`, `zone_buffer`, `gate_escalate`, `ambiguity_delta`, `gazetteer` | ver arriba | Parametros del motor. |

### Salida `json`

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/normalize \
  -F "file=@archivo.xlsx" \
  -F "address_column=direccion" \
  -F "output_format=json"
```

```json
{
  "summary": {"OK": 1487, "SIN_MATCH": 612, "NO_PARSEABLE": 119, "total": 2218},
  "address_column": "direccion",
  "warnings": ["fila de encabezado detectada automaticamente: 2"],
  "results": [
    {
      "direccion_entrada": "Carrera7#14-58",
      "estado": "OK",
      "motivo": "",
      "direccion_normalizada": "KR 7 # 14 - 58",
      "fuente_normalizacion": "catastro",
      "numero_predial_nacional": "<NPN_30_DIGITOS>",
      "manzana": "<MANZANA_17_DIGITOS>",
      "lat": 3.4500,
      "lon": -76.5300,
      "confianza": 0.969,
      "confiabilidad": 0.675,
      "confiabilidad_manzana": 0.906,
      "barrio_vereda": "San Pedro",
      "comuna_corregimiento": "Comuna 3"
    },
    {
      "direccion_entrada": "Calle 9 # 41-63, San Fernando",
      "estado": "SIN_MATCH",
      "motivo": "placa 63 != 69",
      "direccion_normalizada": "CL 9 # 41 - 63",
      "fuente_normalizacion": "reglas",
      "numero_predial_nacional": null,
      "manzana": null,
      "lat": null,
      "lon": null,
      "confianza": null,
      "confiabilidad": null,
      "confiabilidad_manzana": null,
      "barrio_vereda": "San Fernando Nuevo",
      "comuna_corregimiento": "Comuna 19"
    }
  ]
}
```

Los valores de `numero_predial_nacional`, `manzana`, `lat` y `lon` de este ejemplo son marcadores ficticios; no corresponden a ningún predio real.

### Salidas de archivo

Con `output_format=xlsx`, `csv` o `geojson` la respuesta es el archivo, con
`Content-Disposition: attachment`.

- **xlsx** — tres hojas: `normalizado` (todas las filas), `revisar` (solo las que
  no quedaron en `OK`) y `resumen` (conteo por estado). Es el mismo libro que
  produce `scripts/normalizar.py --output salida.xlsx`.
- **csv** — la tabla plana, codificada en UTF-8 con BOM para que Excel la abra
  bien en Windows.
- **geojson** — un `FeatureCollection` con geometria `Point` donde hubo
  coordenada y `"geometry": null` donde no. Las filas nunca se descartan: el
  numero de features siempre coincide con el de filas, de modo que se puede ver
  que no se pudo ubicar. Las demas columnas van en `properties`.

```bash
curl -X POST https://<app>.up.railway.app/api/v1/normalize \
  -F "file=@archivo.xlsx" -F "address_column=direccion" \
  -F "output_format=xlsx" -o archivo_normalizado.xlsx
```

---

## `POST /api/v1/normalize-json`

Punto de integracion para otras fuentes que **ya tienen los datos en memoria** y
no quieren subir un archivo. Se envia uno de dos cuerpos, nunca los dos:

```json
{"records": [{"id": 1, "dir": "Calle 9 # 41-63"}], "address_field": "dir"}
```

```json
{"addresses": ["Calle 9 # 41-63", "Carrera 7 # 14-58"]}
```

Los parametros del motor van en el mismo objeto JSON (`"min_struct": 0.7`, etc.).
La respuesta tiene la misma forma que `/api/v1/normalize` con
`output_format=json`, y `results` conserva el orden de entrada.

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/normalize-json \
  -H "Content-Type: application/json" \
  -d '{"records": [{"id": 1, "dir": "Carrera7#14-58"}], "address_field": "dir"}'
```

### Ejemplo en Python con `requests`

```python
import requests

BASE = "https://<app>.up.railway.app"

# Filas que ya vienen de una base de datos, un CSV leido localmente, etc.
filas = [
    {"id": 1, "direccion": "Carrera7#14-58"},
    {"id": 2, "direccion": "Calle 9 # 41-63, San Fernando"},
    {"id": 3, "direccion": "Corregimiento los andes, vereda la reforma, casa 204"},
]

respuesta = requests.post(
    f"{BASE}/api/v1/normalize-json",
    json={
        "records": filas,
        "address_field": "direccion",
        "barrio_buffer": 1000,
        "zone_buffer": 300,
        "gate_escalate": True,
    },
    timeout=600,
)
respuesta.raise_for_status()
datos = respuesta.json()

print(datos["summary"])
# Los resultados vienen en el mismo orden que las filas enviadas.
for fila, salida in zip(filas, datos["results"]):
    if salida["estado"] == "OK":
        print(fila["id"], salida["numero_predial_nacional"],
              salida["lat"], salida["lon"], salida["barrio_vereda"])
    else:
        print(fila["id"], salida["estado"], salida["motivo"])
```

Para volumenes grandes, envie lotes de unos pocos miles de filas: el limite por
peticion es `MAX_ROWS` (20 000 por omision) y el servicio corre con un solo
worker.

---

## `POST /api/v1/normalize-address`

Alias minimo del anterior: solo acepta `addresses`. Existe para pruebas de humo e
integraciones triviales, y responde con la misma forma.

```bash
curl -s -X POST https://<app>.up.railway.app/api/v1/normalize-address \
  -H "Content-Type: application/json" \
  -d '{"addresses": ["Carrera7#14-58", "no es una direccion"]}'
```

```json
{
  "summary": {"OK": 1, "SIN_MATCH": 0, "NO_PARSEABLE": 1, "total": 2},
  "address_column": null,
  "warnings": [],
  "results": [
    {"direccion_entrada": "Carrera7#14-58", "estado": "OK", "confianza": 0.969, "...": "..."},
    {"direccion_entrada": "no es una direccion", "estado": "NO_PARSEABLE", "...": "..."}
  ]
}
```

---

## Endpoints con forma Esri

`GET|POST /arcgis/rest/services/CaliNormalizador/GeocodeServer/findAddressCandidates`
y
`POST /arcgis/rest/services/CaliNormalizador/GeocodeServer/geocodeAddresses`
responden con la forma JSON de las operaciones homonimas de ArcGIS REST.

**Reproducen unicamente la forma de la respuesta. No constituyen un Locator de
ArcGIS registrado ni certificado.** Los detalles y las limitaciones estan en
[ESRI_INTEGRATION.md](ESRI_INTEGRATION.md).

---

## Errores

Todos los errores traen un cuerpo JSON. Los tres que una integracion debe manejar:

### 422 — hay que definir la columna de direccion

Se devuelve cuando no se envio `address_column` (o `address_field`) y la
deteccion automatica no es concluyente: ninguna columna es plausible, o hay
varias igualmente plausibles.

```json
{
  "error": "address_column_required",
  "message": "varias columnas podrian ser la direccion; indique cual usar",
  "requested_column": null,
  "available_columns": ["id", "direccion_bien", "domicilio", "barrio"],
  "candidates": [
    {"column": "direccion_bien", "score": 95.0},
    {"column": "domicilio", "score": 95.0}
  ]
}
```

La reaccion correcta es mostrar `candidates` al usuario y repetir la peticion con
`address_column`. Si se envio un nombre que no existe, `requested_column` lo
repite y `available_columns` lista lo que si hay.

El mismo codigo 422 se usa cuando el archivo excede `MAX_ROWS`, con un cuerpo
`{"detail": "el archivo trae N filas y el limite por peticion es M..."}`.

### 413 — archivo demasiado grande

```json
{
  "error": "payload_too_large",
  "message": "el cuerpo de la peticion pesa 48.2 MB y el limite es 25 MB"
}
```

El limite se configura con `MAX_UPLOAD_MB` (25 por omision).

### 400 — archivo invalido

```json
{
  "error": "bad_file",
  "message": "el .zip no contiene un shapefile completo; falta .dbf. Comprima juntos .shp, .shx y .dbf (y .prj si lo tiene)."
}
```

Tambien cubre un archivo de 0 bytes, un JSON malformado, un GeoJSON que no es
`FeatureCollection` y una extension no soportada.

### 503 — el servicio aun no esta listo

El modelo se carga en un hilo aparte para que el puerto quede disponible de
inmediato. Mientras eso ocurre, `/health` responde 503 con
`"status": "loading"` y los endpoints de normalizacion esperan hasta
`WARMUP_TIMEOUT_S` segundos antes de responder 503. Un cliente debe reintentar.

---

## Variables de entorno

| Variable | Omision | Para que sirve |
| --- | --- | --- |
| `NORMALIZER_DEVICE` | `cpu` | `cpu` o `cuda`. |
| `ARTIFACTS_DIR` | `<raiz>/artifacts` | Directorio con `model.pt`, `catastro_emb.pt`, `catastro_docs.parquet`, `tuning.json`, `gazetteer.pkl`, `reliability.json`. |
| `BASEMAPS_DIR` | `<raiz>/basemaps` | Directorio con los dos GeoJSON del IDESC. |
| `DOCS_DIR` | `<raiz>/docs` | Directorio desde donde se sirve `/docs-integracion`. |
| `MAX_UPLOAD_MB` | `25` | Tamano maximo del cuerpo de la peticion. |
| `MAX_ROWS` | `20000` | Filas maximas por peticion de normalizacion. |
| `WARMUP_TIMEOUT_S` | `300` | Segundos que una peticion espera a que el modelo termine de cargar. |

---

## Nota de privacidad

Ningun valor por omision de este servicio apunta a `context/` ni a `outputs/`.
Esos directorios contienen datos reales de ciudadanos (registros de inmuebles,
registros de personas afectadas con nombres y numeros de documento,
reportes de campo) y estan excluidos en `.gitignore`,
`.dockerignore` y `.railwayignore`: nunca deben subirse a un repositorio, ni
incluirse en una imagen de contenedor, ni desplegarse. Los bytes que llegan por
`multipart/form-data` se procesan en memoria y se descartan al terminar la
peticion; el servicio no escribe archivos subidos en disco.
