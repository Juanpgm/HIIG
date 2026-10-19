# 05. Uso e implementación

Guía para instalar, ejecutar y desplegar el normalizador. Los ejemplos de direcciones, números y respuestas son **inventados** y sirven solo para ilustrar la forma de los datos. Los parámetros se describen en [03-configuracion.md](03-configuracion.md); el significado de cada columna de salida, en [06-predicciones.md](06-predicciones.md).

## Contenido

1. [Antes de empezar: artefactos necesarios](#1-antes-de-empezar-artefactos-necesarios)
2. [Instalación](#2-instalación)
3. [Uso por línea de comandos](#3-uso-por-línea-de-comandos)
4. [Formatos de entrada y salida (Excel, CSV, SQL y otros)](#4-formatos-de-entrada-y-salida)
5. [Uso de la API HTTP](#5-uso-de-la-api-http)
6. [Endpoints con forma Esri](#6-endpoints-con-forma-esri)
7. [Despliegue con Docker y Railway](#7-despliegue-con-docker-y-railway)
8. [Rendimiento y memoria](#8-rendimiento-y-memoria)
9. [Seguridad](#9-seguridad)

## 1. Antes de empezar: artefactos necesarios

Este repositorio publica el código, los pesos del codificador (`artifacts/model.pt`) y los archivos `tuning.json` y `reliability.json`. **No publica** el índice catastral, los embeddings, los límites geográficos ni los datos de entrenamiento o evaluación (política completa en [07-requisitos.md](07-requisitos.md), sección 5). Por lo tanto, **la predicción exige que el usuario construya el índice con su propio catastro de Cali**.

Ruta mínima para predecir con el modelo publicado:

| Paso | Acción | Detalle |
|------|--------|---------|
| 1 | Obtener su tabla catastral de Cali | Esquema exacto en [04-entrenamiento.md](04-entrenamiento.md), sección 2.1 (`OBJECTID`, `numero_predial_nacional`, `numero_predial_manzana`, `direccion`, `centroid_lon`, `centroid_lat`). |
| 2 | Construir `catastro_docs.parquet` y `catastro_own_centroids.parquet` | Fragmento de Python de la sección "Escenario de solo predicción" de [04](04-entrenamiento.md), o `scripts/prepare_training.py` (genera además tensores de entrenamiento innecesarios para predecir). |
| 3 | Calcular `catastro_emb.pt` | `cali-address setup` (rápido en GPU, más lento en CPU; vuelve a generar `catastro_emb.pt`, que no se versiona porque pesa 161 MB). |
| 4 | (Opcional) Aportar los límites geográficos | Dos GeoJSON en `basemaps/` con el esquema descrito en la sección 3.3 de [07](07-requisitos.md). Sin ellos, el gazetteer se desactiva: no hay detección de barrio ni de comuna en el texto, ni puerta geográfica, y las columnas de zona se llenan solo con la comuna derivada del NPN. |

Cachés que se construyen solas (no se versionan): la primera vez que se necesitan se crean `artifacts/interp_index_v1.pkl` + `.json` (índice de interpolación de posiciones, unos 8 MB, lenta de construir en frío) y `artifacts/catalog_keys_v1.pkl` + `.json` (claves exactas del catastro para `catalog_status`, unos 6 MB). Para construirlas de antemano, por ejemplo antes de crear la imagen Docker, ejecute `python deploy/prepare_artifacts.py` (`--no-warm` las omite); el directorio de artefactos debe ser escribible o traerlas ya construidas, de lo contrario el servicio solo emite una advertencia y las mantiene en memoria. Detalle en [06](06-predicciones.md), sección 10.6.

Si no se completa el paso 2, `cali-address normalize` termina con código 3 y lista los archivos faltantes. El mensaje de `setup` sugiere ejecutar `git pull` porque en el repositorio de origen esos archivos estaban versionados; en este repositorio publicado **no lo están**.

## 2. Instalación

Requiere Python 3.11 o superior. Instale primero torch en su variante de CPU para evitar que `pip` descargue la versión CUDA (unos 3 GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[api,excel]"
```

Extras disponibles (de `pyproject.toml`):

| Extra | Añade |
|-------|-------|
| `excel` | Salida `.xlsx` y lectura `.xls` (`openpyxl`, `xlrd`) |
| `sql` | Orígenes SQL (`SQLAlchemy`; además, el controlador de su base de datos) |
| `geo` | Lectura de GeoPackage (`geopandas`, `pyogrio`) |
| `api` | Servicio FastAPI (`fastapi`, `uvicorn`, `python-multipart`, `pydantic`) |
| `train` | Scripts de entrenamiento y evaluación |
| `dev` | `pytest`, `httpx`, `scikit-learn`, `ruff` |
| `all` | Todo excepto `dev` |

Si el comando `cali-address` no está en su `PATH` (frecuente en Windows), use `python -m cali_address ...`.

Para entrenamiento, instale la variante de torch con CUDA que corresponda a su GPU.

## 3. Uso por línea de comandos

```bash
cali-address setup                            # primera vez: genera catastro_emb.pt y se autoverifica
cali-address formats                          # formatos admitidos
cali-address inspect datos.csv                # formato, encabezado, columnas, columna sugerida, vista previa
cali-address normalize datos.csv              # escribe datos_normalizado.csv junto a la entrada
cali-address normalize datos.csv -o salida.csv --dry-run 20      # vista previa de 20 filas, no escribe
```

Casos frecuentes:

```bash
# Excel con encabezado en la fila 3 (base 0 = 2) y una hoja concreta
cali-address normalize reporte.xlsx -o salida.xlsx --sheet Hoja1 --header-row 2

# Dirección repartida en varias columnas
cali-address normalize datos.csv -o salida.csv --address-parts via,numero,complemento

# Descartar filas de otros municipios (requiere una columna con el municipio)
cali-address normalize datos.csv -o salida.csv --municipality-col municipio

# Configuración en un archivo TOML (las banderas de la línea de comandos tienen prioridad)
cali-address normalize --config corrida.toml

# Servir la página de carga en la máquina local
cali-address web --port 8001 --no-browser
```

Comportamiento a tener en cuenta:

- Un archivo de salida existente se **sobrescribe** de forma atómica (se escribe a un temporal y se mueve al final); la entrada nunca se sobrescribe.
- Una fila que falla no aborta la corrida (`estado = ERROR`), salvo con `--on-error raise`.
- El orden y la cantidad de filas se conservan para cualquier `--chunk-size`, y el resultado no depende de él.
- `--municipality-col` solo reconoce Cali (nombres como `Cali` o `Santiago de Cali`, o el código `76001`, `76001000`). Cualquier otro municipio no vacío produce `FUERA_DE_AREA`. Sin esa opción **no hay ningún control**: una dirección de otra ciudad se comparará con los predios de Cali.
- Resumen de la corrida por la salida de error y, con `--summary-json`, en un archivo.

## 4. Formatos de entrada y salida

| Tipo | Formatos | Notas |
|------|----------|-------|
| Entrada tabular | CSV, TSV, TXT, Excel (`.xlsx`, `.xls`), Parquet, JSON, JSONL | El delimitador y la codificación del CSV se detectan (BOM, UTF-8, cp1252, UTF-16). Un `.txt` sin delimitador se trata como una dirección por línea sin encabezado. |
| Entrada geoespacial | GeoJSON (`FeatureCollection`), shapefile (`.shp` con `.shx` y `.dbf`, o `.zip`), GeoPackage (necesita `geo`) | El centroide de cada geometría se añade como `_lon` y `_lat` informativos. |
| Entrada remota | URL http(s), URL de SQLAlchemy | Una URL se descarga primero a un temporal. |
| Salida | CSV (UTF-8 con BOM), TSV, XLSX, Parquet, JSON, JSONL, GeoJSON | El XLSX tiene tres hojas: `normalizado`, `revisar` (filas que no quedaron `OK`) y `resumen`. El GeoJSON tiene un `Point` por fila ubicada y geometría `null` en las demás. |

Excel: se detecta la fila de encabezado cuando se omite `--header-row`; solo se lee la primera hoja salvo que se pase `--sheet`. El `.xlsx` de salida se arma en memoria y Excel limita una hoja a 1 048 575 filas; para resultados enormes use CSV o Parquet.

CSV y TSV, XLSX (modo de solo lectura), Parquet, JSONL, `.shp` y SQL se leen por bloques, de modo que archivos mayores que la memoria funcionan. `.xls`, `.json`, GeoJSON, `.zip` y GeoPackage se leen completos y luego se parten.

SQL: use `--table [esquema.]nombre` con una tabla o una **vista**; no existe una opción de consulta libre porque un filtro de texto no es una frontera de seguridad para SQL. Para uniones o filtros cree una vista de solo lectura en la base de datos. La sesión se abre en modo de solo lectura donde el controlador lo permite (SQLite y PostgreSQL con `psycopg2`, `psycopg` o `pg8000`) y termina siempre con una reversión; conéctese de todos modos con una cuenta de solo lectura. Las contraseñas y los parámetros secretos de la URL se enmascaran en los mensajes (con una excepción para contraseñas de 1 o 2 caracteres fuera de una URL; use contraseñas de al menos 3).

```bash
cali-address normalize sqlite:///local.db -o salida.parquet --table direcciones
cali-address normalize "postgresql://USUARIO:CLAVE@HOST/BASE" -o salida.csv \
    --table public.predios --address-parts via,numero
```

Los nombres `USUARIO`, `CLAVE`, `HOST` y `BASE` son marcadores de posición.

## 5. Uso de la API HTTP

Arranque local (con los artefactos de la sección 1):

```bash
export ARTIFACTS_DIR=./artifacts BASEMAPS_DIR=./basemaps
uvicorn cali_address.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Los endpoints se verificaron en `src/cali_address/api/main.py`. La documentación interactiva está en `/docs` y la guía de integración, en `/docs-integracion`.

| Método y ruta | Cuerpo | Respuesta |
|---------------|--------|-----------|
| `GET /health` | - | 200 con `status: ok` o 503 con `status: loading` o `error`. |
| `POST /api/v1/inspect` | `multipart/form-data`: `file`, `sheet`, `header` | Formato, columnas, columna sugerida, candidatas con puntaje, vista previa, advertencias. |
| `POST /api/v1/normalize` | `multipart/form-data`: `file`, `sheet`, `header`, `address_column`, `output_format` (`json`, `xlsx`, `csv`, `geojson`) y los parámetros del motor | JSON con `summary`, `address_column`, `warnings`, `results`, o el archivo. |
| `POST /api/v1/normalize-json` | JSON: `records` más `address_field`, o `addresses`; más parámetros del motor | Misma forma que `normalize` en JSON; el orden se conserva. |
| `POST /api/v1/normalize-address` | JSON: `addresses` | Igual, para pruebas rápidas. |

Parámetros del motor aceptados: `min_struct`, `plate_tolerance`, `barrio_buffer`, `zone_buffer`, `gate_escalate`, `ambiguity_delta`, `gazetteer`. El umbral de confianza no es un parámetro de la petición.

### 5.1 Ejemplo: lista de direcciones

Petición (direcciones inventadas):

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/normalize-json \
  -H "Content-Type: application/json" \
  -d '{"addresses": ["Carrera 12 # 34-56", "esto no es una direccion"]}'
```

Respuesta ilustrativa (los valores son inventados; la forma es la real):

```json
{
  "summary": {"OK": 1, "SIN_MATCH": 0, "NO_PARSEABLE": 1, "total": 2},
  "address_column": null,
  "warnings": [],
  "results": [
    {
      "direccion_entrada": "Carrera 12 # 34-56",
      "estado": "OK",
      "motivo": "",
      "direccion_normalizada": "KR 12 # 34 - 56",
      "fuente_normalizacion": "catastro",
      "numero_predial_nacional": "<NPN_30_DIGITOS>",
      "manzana": "<MANZANA_17_DIGITOS>",
      "lat": 3.4500,
      "lon": -76.5300,
      "confianza": 0.9412,
      "confiabilidad": 0.64,
      "confiabilidad_manzana": 0.86,
      "margen": 0.12,
      "nivel_precision": "predio",
      "barrio_vereda": "Barrio de ejemplo",
      "comuna_corregimiento": "Comuna 3",
      "complemento_normalizado": null
    },
    {
      "direccion_entrada": "esto no es una direccion",
      "estado": "NO_PARSEABLE",
      "motivo": "implicit_hash; via_unparsed",
      "direccion_normalizada": null,
      "fuente_normalizacion": null,
      "numero_predial_nacional": null,
      "manzana": null,
      "lat": null,
      "lon": null,
      "confianza": null,
      "confiabilidad": null,
      "confiabilidad_manzana": null,
      "margen": null,
      "nivel_precision": null,
      "barrio_vereda": null,
      "comuna_corregimiento": null,
      "complemento_normalizado": null
    }
  ]
}
```

Los valores de `numero_predial_nacional`, `manzana`, `lat` y `lon` de este ejemplo son marcadores ficticios; no corresponden a ningún predio real.

En la respuesta real, los conteos de `summary` suman `total`. Los valores de `motivo` en filas `NO_PARSEABLE` son las notas del analizador gramatical.

### 5.2 Ejemplo: registros con una columna de dirección

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/normalize-json \
  -H "Content-Type: application/json" \
  -d '{"records": [{"id": 1, "dir": "Carrera 12 # 34-56"}], "address_field": "dir", "barrio_buffer": 1000}'
```

### 5.3 Ejemplo: archivo

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/normalize \
  -F "file=@datos.xlsx" -F "address_column=direccion" -F "output_format=xlsx" \
  -o datos_normalizado.xlsx
```

### 5.4 Errores

| Código | Cuerpo | Causa |
|-------:|--------|-------|
| 400 | `{"error": "bad_file", "message": ...}` | Archivo vacío, formato no admitido, JSON mal formado, zip sin shapefile completo. |
| 413 | `{"error": "payload_too_large", ...}` | Cuerpo mayor que `MAX_UPLOAD_MB` (25 por defecto). |
| 422 | `{"error": "address_column_required", "candidates": [...], ...}` | Columna de dirección ambigua o inexistente. También: más filas que `MAX_ROWS` (20 000). |
| 503 | `{"status": "loading"}` o detalle de falla | El modelo aún no cargó; el cliente debe reintentar. |

La API no emite los estados `ERROR` ni `FUERA_DE_AREA`.

### 5.5 Cliente en Python

```python
import requests

BASE = "http://127.0.0.1:8000"
filas = [{"id": 1, "direccion": "Carrera 12 # 34-56"},
         {"id": 2, "direccion": "Calle 9 No 8-07"}]
r = requests.post(f"{BASE}/api/v1/normalize-json",
                  json={"records": filas, "address_field": "direccion"}, timeout=600)
r.raise_for_status()
for fila, salida in zip(filas, r.json()["results"]):
    if salida["estado"] == "OK":
        print(fila["id"], salida["numero_predial_nacional"], salida["nivel_precision"])
    else:
        print(fila["id"], salida["estado"], salida["motivo"])
```

Para volúmenes grandes envíe lotes de unos pocos miles de filas: el límite por petición es `MAX_ROWS` y el servicio corre con un solo trabajador.

## 6. Endpoints con forma Esri

Rutas bajo `/arcgis/rest/services/CaliNormalizador/GeocodeServer`:

| Operación | Método | Entrada | Salida |
|-----------|--------|---------|--------|
| `findAddressCandidates` | `GET` o `POST` (formulario) | `SingleLine`, `Address` o `text` | `spatialReference` y exactamente un candidato (o ninguno si la entrada está vacía) |
| `geocodeAddresses` | `POST` JSON | `{"records": [{"attributes": {"OBJECTID": 1, "SingleLine": "..."}}]}` | `locations` con `attributes.ResultID` igual a `OBJECTID` |

Ejemplo ilustrativo:

```bash
curl -s "http://127.0.0.1:8000/arcgis/rest/services/CaliNormalizador/GeocodeServer/findAddressCandidates?SingleLine=Carrera%2012%20%23%2034-56&f=json"
```

```json
{
  "spatialReference": {"wkid": 4326, "latestWkid": 4326},
  "candidates": [
    {
      "address": "KR 12 # 34 - 56",
      "location": {"x": -76.5300, "y": 3.4500},
      "score": 94.12,
      "attributes": {
        "Status": "OK", "Numero_Predial": "<NPN_30_DIGITOS>",
        "Manzana": "<MANZANA_17_DIGITOS>", "Barrio_Vereda": "Barrio de ejemplo",
        "Comuna_Corregimiento": "Comuna 3", "Motivo": "",
        "Confiabilidad": 0.64, "Margen": 0.12
      },
      "extent": {"xmin": -76.5305, "ymin": 3.4495, "xmax": -76.5295, "ymax": 3.4505}
    }
  ]
}
```

Los valores de `numero_predial_nacional`, `manzana`, `lat` y `lon` de este ejemplo son marcadores ficticios; no corresponden a ningún predio real.

Advertencias de uso:

- **Reproducen solo la forma JSON.** No son un Locator de ArcGIS registrado ni certificado. No implementan metadatos del servicio, `reverseGeocode`, `suggest`, `outFields`, `magicKey` ni autenticación por token.
- Una fila `SIN_MATCH` o `NO_PARSEABLE` también devuelve un candidato, con `score` 0 y sin coordenadas. Filtre por `attributes.Status` o por `score` antes de usar la ubicación.
- `score` es `confianza` por 100; hereda la salvedad de que no es una probabilidad.
- La extensión del candidato es una caja sintética de unos 55 m de semilado; no representa el predio.
- Detalle y uso desde ArcGIS Pro: `docs/ESRI_INTEGRATION.md`.

## 7. Despliegue con Docker y Railway

El `Dockerfile` construye una imagen de CPU. Requisitos previos en el contexto de construcción:

1. `deploy/artifacts/` con los **siete** archivos de servicio. `python deploy/prepare_artifacts.py` los copia desde `artifacts/` y falla si falta alguno. En este repositorio, `catastro_docs.parquet`, `catastro_own_centroids.parquet`, `catastro_emb.pt` y `gazetteer.pkl` **no se distribuyen**: debe generarlos el usuario (sección 1).
2. `basemaps/barrios_veredas.geojson` (la construcción de la imagen verifica su existencia con `test -f`). Los límites geográficos tampoco se distribuyen. Si no los tiene, construya la imagen sin mapas base haciendo **tres ediciones** (más la del paso 1): (a) en el `Dockerfile`, elimine la línea `COPY basemaps/ ./basemaps/` (línea 41); (b) en el mismo `Dockerfile`, elimine la verificación `test -f /app/artifacts/gazetteer.pkl` (línea 52, que exige el caché del gazetteer) y la de `test -f /app/basemaps/barrios_veredas.geojson` (línea 53); (c) en `deploy/prepare_artifacts.py`, quite `"gazetteer.pkl"` de `SERVING_ARTIFACTS` (línea 30), de modo que el script no falle por ese archivo y copie seis archivos en lugar de siete. Asuma las consecuencias descritas en la sección 1 (sin detección de barrio ni comuna en el texto y sin puerta geográfica).

```bash
python deploy/prepare_artifacts.py
docker build -t normalizador-direcciones .
docker run --rm -p 8000:8000 normalizador-direcciones
curl -s http://127.0.0.1:8000/health
```

Railway:

```bash
npm i -g @railway/cli && railway login
railway init
railway variables --set NORMALIZER_DEVICE=cpu --set ARTIFACTS_DIR=/app/artifacts \
                  --set BASEMAPS_DIR=/app/basemaps --set DOCS_DIR=/app/docs \
                  --set MAX_UPLOAD_MB=25 --set MAX_ROWS=20000 --set WARMUP_TIMEOUT_S=300
railway up --no-gitignore
```

`--no-gitignore` es necesario porque `deploy/artifacts/` suele estar excluido del control de versiones; entonces `.railwayignore` pasa a ser la lista de exclusiones vigente. Compruebe en el registro de la subida que el tamaño es el esperado (unos 205 MB con los datos de Cali) y detenga el despliegue si es notablemente mayor. La guía completa de despliegue (en español) es `docs/DEPLOY.md`.

La plataforma lanza `/health` hasta 300 s; mientras el modelo carga, responde 503. La imagen usa un solo trabajador y `numReplicas = 1`.

## 8. Rendimiento y memoria

Las mediciones de memoria, rendimiento por lote, arranque en frío y tamaño de imagen no se publican; mídalas en su entorno (ver `docs/DEPLOY.md`). Recomendación de dimensionamiento: al menos 2 GB de RAM por instancia (un límite de 1 GB o menos provoca falta de memoria al cargar el modelo y procesar el primer lote grande).

Con GPU (`NORMALIZER_DEVICE=cuda` o `--device cuda`) el procesamiento por lotes es más rápido, pero la imagen publicada es de CPU.

## 9. Seguridad

1. **Deserialización insegura de artefactos.** `model.pt` se carga con `torch.load(..., weights_only=False)` (`inference.py:310`, `train.py:170`) porque contiene arreglos de NumPy (`xy_mean`, `xy_std`, códigos de comuna y barrio). Verificado: con `weights_only=True` la carga falla. `gazetteer.pkl` se carga con `pickle.load` (`gazetteer.py:949`), y `train_data.npz` con `allow_pickle=True`. Un archivo `pickle` o un checkpoint manipulado puede **ejecutar código arbitrario** al cargarse. Cargue únicamente artefactos de origen confiable y verificados.
   - Verifique la integridad del `model.pt` publicado comparando su SHA-256 (de la copia fuente, 13 527 959 bytes): `890b6d83eea29c3783e03416029d4b45682da0fadf411eb73f90b616954349df`. Los de `tuning.json` y `reliability.json` son `d0892cbe4410d701b9a926b032885bb92dc95359fd145be0808465af133166ef` y `7a5434ce1616ae839ad77c95a3cb5e3bc6c42d345d4ac28e2e6c20a4a910d1f0`.
   - `catastro_emb.pt` se carga con el modo predeterminado de `torch.load`; en torch 2.6 o posterior ese modo es `weights_only=True`, pero `pyproject.toml` admite `torch>=2.2`, donde el valor por defecto era `False`.
   - Mitigación recomendada: no apuntar `--artifacts-dir`, `CALI_ARTIFACTS_DIR` ni `ARTIFACTS_DIR` a directorios que no controle; no regenerar `gazetteer.pkl` desde límites de origen dudoso sin revisar el archivo.
2. **La API no tiene autenticación, limitación de tasa ni CORS configurado.** Si se despliega en una URL pública (`railway domain`), cualquiera puede enviar peticiones hasta `MAX_UPLOAD_MB` y `MAX_ROWS`. Colóquela detrás de un proxy o una pasarela con autenticación si el servicio no debe ser público.
3. **Cargas.** Los bytes recibidos se procesan en memoria y no se escriben a disco. Aun así, un `.zip` con shapefile se descomprime completo sin límite de tamaño descomprimido: no hay protección contra una bomba de compresión; no procese archivos de origen no confiable.
4. **Datos personales.** Las direcciones de entrada pueden ser datos personales. La API y la CLI no las registran en archivos por diseño, pero el registro del servidor y los resultados que usted escriba sí pueden contenerlas. Aplique su política de protección de datos.
5. **Fuentes externas.** El módulo de evaluación contra geocodificadores de terceros no se incluye en esta versión pública.
