# Guía de despliegue del normalizador de direcciones

Esta guía explica cómo configurar, operar y desplegar la imagen de `normalizador_produccion/` en cualquier
plataforma que ejecute contenedores. Todos los ejemplos de direcciones son inventados.

> **Estado: imagen candidata / staging, NO apta para producción hasta aprobación del titular.** La herramienta de
> despliegue (imagen, validaciones, verificación del índice, pruebas) es de nivel producción; los pesos de `modelo/`
> son el candidato del reentrenamiento 2026-10 (veredicto **NO APRUEBA**, no promovido). El servicio en producción
> conserva el modelo anterior. Las etiquetas de la imagen llevan `co.hiig.status=candidato-no-aprobado` y la etiqueta
> sugerida es `0.2.0-candidato`; quite ambas solo cuando el titular apruebe el modelo.

## 1. Cómo está armado

```
imagen (pública, sin datos)              paquete del índice (privado, lo aporta el usuario)
/app/modelo/  model.onnx, onnx_meta.json,   /data/indice/  catastro_docs.parquet
              decision.json, ...                           catastro_emb.npy
/app/src/     cali_address + servicio                      index_meta.json, ...
                     \                            /
                      python -m servicio  ->  /tmp/normalizador/artifacts (enlaces)  ->  uvicorn
```

Al arrancar, `python -m servicio`:

1. valida todas las variables de entorno (un valor mal escrito termina el proceso con código 2);
2. resuelve el backend (`onnx` por defecto; `onnx-int8` cae a `onnx` si falta su grafo, salvo que se desactive);
3. si faltan archivos del índice y hay `INDEX_URL`, lo descarga y verifica su sha256;
4. comprueba el índice: archivos obligatorios, filas de documentos = filas de embeddings, dimensión = la del modelo
   y, si trae `indice_manifest.json`, que fue construido para **este** modelo y que **cada** archivo coincide con el
   tamaño y el sha256 del manifiesto (un archivo alterado, aunque pese lo mismo, o uno presente pero no listado en el
   manifiesto, impide el arranque; si algo falla: código 3). Esta verificación lee el paquete completo una vez al
   arrancar (unos segundos con el índice de 401.860 documentos);
5. enlaza modelo e índice en `RUNTIME_DIR` y arranca uvicorn con `servicio.app:app`.

`GET /health` responde 503 mientras carga y 200 cuando el modelo y el índice están en memoria. Mientras carga, las
demás rutas responden 503 de inmediato con `Retry-After: 5` (la API original bloqueaba el bucle de eventos durante
la carga; la imagen añade esa compuerta).

## 2. El índice

### Qué contiene

| Archivo | Obligatorio | Origen |
|---|---|---|
| `catastro_docs.parquet` | sí | documentos de referencia (nomenclatura y respaldo catastral) |
| `catastro_emb.npy` | sí | embeddings fp16 de los documentos (n_docs x 256) |
| `index_meta.json` | sí | marcador del índice; activa la separación de letras del pre-normalizador |
| `catastro_own_centroids.parquet` | no | centroide del predio devuelto (sin él se usa el centroide de la dirección) |
| `interp_index_v1.*`, `catalog_keys_v1.*`, `gazetteer.pkl` | no | cachés de arranque (se reconstruyen si faltan; unos 20 s más) |
| `indice_manifest.json` | no, recomendado | modelo al que pertenece, tamaños y sha256 (lo escribe `preparar_indice.py`) |

Los mapas base IDESC (`barrios_veredas.geojson`, `comunas_corregimientos.geojson`) van aparte en `BASEMAPS_DIR` y
son opcionales: sin ellos el servicio funciona, pero no asigna barrio ni comuna por polígono.

Todo esto deriva de datos de terceros y **no se publica** en git ni en la imagen pública. El índice del candidato
final tiene 401.860 documentos y ocupa unos 290 MB (206 MB de embeddings).

### Construirlo: `scripts/preparar_indice.py`

El script parte de un directorio con al menos `catastro_docs.parquet` e `index_meta.json`, es decir, la salida de
`normalizador_direcciones/scripts/build_index.py` (que a su vez parte de la nomenclatura y el catastro que el
usuario descarga con `download_nomenclatura.py` / `download_catastro.py`; ver `normalizador_direcciones/docs/`).
Los embeddings se toman, en este orden:

1. `catastro_emb.npy` del origen (copia exacta);
2. `catastro_emb.pt` del origen (se convierte; requiere torch solo para este paso);
3. si no hay ninguno, o con `--recalcular-embeddings`, se calculan con `modelo/model.doc.onnx`, el lado de
   documentos del mismo modelo, **sin torch**.

```bash
pip install -r requirements.txt
python scripts/preparar_indice.py --origen /ruta/indice_construido --destino ./indice \
       --basemaps /ruta/basemaps --precalentar --empaquetar indice.tar.gz
```

| Opción | Efecto |
|---|---|
| `--precalentar` | construye en el paquete las cachés de interpolación y catálogo (y `gazetteer.pkl` con `--basemaps`): arranque más rápido |
| `--recalcular-embeddings` | ignora los embeddings del origen y los calcula con `model.doc.onnx` |
| `--enlazar` | enlaces duros en lugar de copias (mismo disco) |
| `--empaquetar archivo.tar.gz` | además escribe un `.tar.gz` plano e imprime su sha256 (para `INDEX_URL`) |
| `--hilos N` | hilos de ONNX Runtime para el cálculo de embeddings |

Si el origen trae `model.pt`, el script verifica que sea el modelo de `modelo/onnx_meta.json` y rechaza embeddings de
otro modelo. Sin `model.pt` solo advierte (`model_verified: false` en el manifiesto).

Equivalencia de los embeddings calculados con `model.doc.onnx` (medida sobre el índice del candidato final):
coseno mínimo 0,99971 y medio 0,999992 frente a los embeddings originales (calculados en GPU con bf16); en 2.000
consultas sintéticas (direcciones públicas corrompidas) por `normalize_strict`, 1.999 dieron exactamente el mismo
resultado (estado, predial, manzana, dirección normalizada) y la restante cambió solo el predial dentro de la misma
manzana y dirección. Tarda unos 6 minutos en CPU (16 hilos). Es equivalente, no idéntico byte a byte; para la
reproducción exacta use los embeddings originales.

### Entregarlo al contenedor (elija una)

1. **Volumen** (recomendado): monte el directorio en `/data/indice` en solo lectura.
2. **Descarga al arrancar**: publique `indice.tar.gz` en un almacenamiento privado con URL HTTPS (URL prefirmada
   de S3, URL firmada de GCS, recurso de una versión de GitHub de un repositorio privado servido por un proxy, etc.)
   y defina `INDEX_URL` e `INDEX_SHA256`. Se admite `.tar.gz` o `.zip`, plano o con una carpeta raíz; las entradas
   con rutas inseguras se rechazan. Si el sha256 no coincide no se instala nada. `INDEX_DIR` debe ser escribible
   (por ejemplo un `tmpfs` o un volumen efímero). Marcador para una versión publicada:
   `INDEX_URL=https://<almacenamiento-privado>/normalizador/indice-v2g-r3.tar.gz`.
3. **Imagen privada con el índice dentro** (para plataformas sin volúmenes):
   `docker build --target con-indice --build-context indice=./indice -t <registro-privado>/normalizador-privado:0.2.0-candidato .`
   Esa imagen (1,25 GB en disco, 421 MB comprimida, medido) **contiene datos derivados de terceros**: súbala solo a
   un registro privado.

Si falta el índice, el contenedor termina con código 3 y este mensaje:

```
[servicio] ERROR: faltan archivos del indice en INDEX_DIR=/data/indice: catastro_docs.parquet, catastro_emb.npy,
index_meta.json. Opciones: (1) monte un volumen ...; (2) defina INDEX_URL e INDEX_SHA256 ...; (3) construyalo con
scripts/preparar_indice.py. Ver GUIA_DESPLIEGUE.md, seccion 'El indice'.
```

Seguridad: **el paquete del índice es una entrada de confianza.** Las cachés `.pkl` (`interp_index_v1.pkl`,
`catalog_keys_v1.pkl`, `gazetteer.pkl`) se cargan con `pickle`, que puede ejecutar código arbitrario al leerlas. El
manifiesto detecta archivos modificados o corruptos, pero quien pueda escribir el paquete y el manifiesto juntos puede
falsificar ambos: use solo paquetes construidos por usted o de una fuente de confianza, no monte volúmenes escribibles
por terceros, y verifique siempre `INDEX_SHA256` en la descarga. Sin `indice_manifest.json` el servicio arranca con un
aviso, pero no puede verificar nada (incluidos los `.pkl`): en despliegues reales exija el manifiesto.

Descarga al arrancar: `INDEX_URL` debe ser `https://`; `file://` está deshabilitado salvo que se defina
`INDEX_ALLOW_FILE_URL=true` (solo pruebas locales). La descarga se aborta al superar `INDEX_MAX_DOWNLOAD_MB`
(4096 por defecto). Si `INDEX_DIR` (o su carpeta madre) no es escribible, el contenedor termina con código 3 y un
mensaje que lo explica; en ese caso monte el paquete ya descomprimido.

## 3. Variables de entorno

| Variable | Defecto | Valores | Efecto |
|---|---|---|---|
| `PORT` | `8000` | 1-65535 | Puerto HTTP (Cloud Run, Railway y otros lo inyectan) |
| `HOST` | `0.0.0.0` | interfaz | Interfaz de escucha |
| `NORMALIZER_BACKEND` | `onnx` | `onnx`, `onnx-int8` | Backend. `torch` se rechaza: la imagen no lo incluye |
| `NORMALIZER_BACKEND_FALLBACK` | `true` | `true`/`false` | Si `onnx-int8` no tiene grafo, usar `onnx` (con aviso) o fallar |
| `NORMALIZER_ONNX_THREADS` | `auto` | `auto`, `0`, 1-256 | Hilos de ONNX Runtime y de BLAS. `auto` usa la cuota de CPU del contenedor (cgroup), no los núcleos del anfitrión |
| `NORMALIZER_ONNX_INDEX` | `fp32` | `fp32`, `fp16` | Precisión de la matriz de documentos en memoria (fp16: unos 200 MB menos, unas 3,5 veces más latencia) |
| `NORMALIZER_EXCLUDE_FUENTES` | vacío | lista separada por comas | Excluye fuentes del índice en servicio (por ejemplo `catastro` desactiva el respaldo catastral sin reconstruir) |
| `UVICORN_WORKERS` | `1` | 1-16 | Procesos. Cada uno carga su propia copia del índice (unos 1,5 GB); escale con réplicas, no con procesos |
| `LOG_LEVEL` | `info` | `critical` ... `trace` | Nivel de log de uvicorn; con `debug`/`trace` se activa el log de accesos |
| `MODEL_DIR` | `/app/modelo` | ruta | Directorio del modelo (en la imagen) |
| `INDEX_DIR` | `/data/indice` | ruta | Paquete del índice |
| `BASEMAPS_DIR` | `/data/basemaps` | ruta | Mapas base IDESC (opcional) |
| `RUNTIME_DIR` | `/tmp/normalizador` | ruta escribible | Directorio de ejecución (enlaces y cachés reconstruidas) |
| `INDEX_URL` | vacío | `https://...` | Descarga el índice si falta (`file://` solo con `INDEX_ALLOW_FILE_URL=true`) |
| `INDEX_ALLOW_FILE_URL` | `false` | `true`/`false` | Permite `INDEX_URL=file://...`; solo para pruebas locales |
| `INDEX_MAX_DOWNLOAD_MB` | `4096` | > 0 | Tamaño máximo de la descarga; se aborta al superarlo |
| `DOCS_DIR` | `/app/docs` | ruta | Carpeta con `API.md` para `/docs-integracion` (404 con mensaje si falta) |
| `INDEX_SHA256` | vacío | 64 hexadecimales | Obligatorio con `INDEX_URL` |
| `INDEX_DOWNLOAD_TIMEOUT_S` | `600` | > 0 | Tiempo máximo de la conexión de descarga |
| `MAX_UPLOAD_MB` | `25` | > 0 y <= 1024 | Tamaño máximo del cuerpo (413 si se supera) |
| `MAX_ROWS` | `20000` | 1-1.000.000 | Filas máximas por petición (422 si se supera) |
| `WARMUP_TIMEOUT_S` | `300` | > 0 | Espera máxima de una petición a que termine la carga |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | IPs o `*` | Proxies de confianza para `X-Forwarded-*` (variable propia de uvicorn) |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | = hilos | entero | Si no se definen, se igualan a los hilos resueltos antes de importar numpy |

Códigos de salida de `python -m servicio`: 0 correcto, 2 configuración inválida, 3 modelo o índice ausente o
incompatible. `python -m servicio check` valida todo sin servir (útil como paso previo de un despliegue).

## 4. Uso de la API

| Método y ruta | Uso |
|---|---|
| `GET /health` | Estado (200 listo, 503 cargando o con error) |
| `POST /api/v1/normalize-address` | Lista de direcciones `{"addresses": [...]}` |
| `POST /api/v1/normalize-json` | `{"addresses": [...]}` o `{"records": [...], "address_field": "..."}` |
| `POST /api/v1/normalize` | Archivo (xlsx, xls, csv, geojson o zip con shapefile); devuelve el archivo normalizado |
| `POST /api/v1/inspect` | Columnas y muestra de un archivo antes de normalizar |
| `GET/POST .../findAddressCandidates`, `.../geocodeAddresses` | Compatibilidad aproximada con clientes ArcGIS |
| `GET /docs` | Documentación interactiva (OpenAPI) |

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"device":"cpu","catastro_size":401860,"gazetteer_loaded":true,"threshold":0.83,"detail":null}

curl -s -X POST http://127.0.0.1:8000/api/v1/normalize-address \
  -H 'content-type: application/json' \
  -d '{"addresses": ["CALLE 10 # 5 - 10", "Cra 11 #12-23", "AV 3N # 45-10"]}'

curl -s -X POST http://127.0.0.1:8000/api/v1/normalize-json \
  -H 'content-type: application/json' \
  -d '{"records": [{"id": 1, "dir": "Calle 5 con carrera 100"}], "address_field": "dir"}'

curl -s -X POST http://127.0.0.1:8000/api/v1/normalize -F "file=@direcciones.csv" -o normalizado.xlsx
```

Forma de la respuesta (extracto, valores ilustrativos):

```json
{
  "summary": {"OK": 1, "SIN_MATCH": 0, "NO_PARSEABLE": 0, "total": 1},
  "results": [{
    "direccion_entrada": "CALLE 10 # 5 - 10",
    "estado": "OK",
    "direccion_normalizada": "CL 10 # 5 - 10",
    "numero_predial_nacional": "...", "manzana": "...",
    "lat": 3.41, "lon": -76.52,
    "nivel_precision": "predio",
    "confiabilidad_calibrada": 0.99, "banda_confiabilidad": "alta"
  }],
  "warnings": [], "address_column": null
}
```

Comportamiento ante entradas difíciles (cubierto por las pruebas): lista vacía devuelve 200 con `results: []`;
cadenas vacías, `null`, solo símbolos, emojis o textos de 5.000 caracteres devuelven 200 con `SIN_MATCH` o
`NO_PARSEABLE` (nunca 500); tildes y eñes se aceptan; JSON mal formado o tipos incorrectos devuelven 422; más filas
que `MAX_ROWS` devuelven 422; un cuerpo mayor que `MAX_UPLOAD_MB` devuelve 413.

## 5. Recursos y dimensionamiento

Mediciones propias de esta imagen (Docker Desktop 29.4 sobre WSL2, Windows 11, AMD Ryzen 9 9950X; índice del
candidato final con 401.860 documentos montado desde el disco local; consultas sintéticas):

| Límite del contenedor | Backend | Listo (s) | p50 HTTP (ms) | p95 HTTP (ms) | Lote de 1.000 filas/s | Memoria tras la prueba |
|---|---|---|---|---|---|---|
| 4 vCPU, 4 GB | onnx fp32 | 3,7 | 53 | 81 | 233 | 1,35 GiB |
| 2 vCPU, 3 GB | onnx fp32 | 3,6 | 48 | 68 | 188 | 1,33 GiB |
| 2 vCPU, 3 GB | onnx-int8 | 3,6 | 46 | 66 | 207 | 1,48 GiB |
| 2 vCPU, 3 GB | onnx fp32, índice fp16 | 3,6 | 170 | 186 | 179 | 1,14 GiB |

Dentro del contenedor, sin HTTP, `normalize_strict` tarda 18,5 ms por dirección con 4 vCPU y 22,6 ms con 2 vCPU; la
diferencia con la columna HTTP es el reenvío de puertos de Docker Desktop, la serialización JSON y el cliente. En la
medición previa fuera de contenedor (`normalizador_direcciones/docs/onnx.md`), ONNX fp32 rinde 1,5 veces lo de torch
con 16 hilos y 3,8 a 4,1 veces con 4 hilos; int8 gana solo 3 a 14 %, usa unos 100 MB más y no es invariante al lote,
por eso el defecto es fp32. El pico de RSS medido fuera de contenedor fue 1,96 GB con 16 hilos.

Recomendación:

- **Mínimo**: 2 vCPU y 3 GB de memoria. **Recomendado**: 2 a 4 vCPU y 4 GB (margen para lotes grandes y archivos).
- Escale horizontalmente con réplicas; un proceso atiende las peticiones de una en una (los endpoints de la API son
  síncronos dentro del bucle de eventos), así que la concurrencia útil por réplica es 1. Lotes de hasta `MAX_ROWS`
  filas en una sola petición rinden mucho más que muchas peticiones de una dirección.
- Disco: la imagen (744 MB) más el índice (unos 290 MB).
- Arranque: unos 4 s con el índice local y cachés precalentadas; añada el tiempo de descarga si usa `INDEX_URL` y unos
  20 s si las cachés no están precalentadas.

## 6. Plataformas

En todas: construya y suba la imagen a un registro, entregue el índice por una de las tres vías de la sección 2,
apunte el health check a `GET /health` y dé al menos 120 s de gracia en el arranque.

### Máquina virtual (cualquier nube o servidor propio)

```bash
git clone https://github.com/Juanpgm/HIIG && cd HIIG/normalizador_produccion
# copie el paquete del índice a ./indice (scp, rsync, descarga privada) y opcionalmente ./basemaps
docker compose up -d --build
INDEX_DIR=$PWD/indice BASEMAPS_DIR=$PWD/basemaps bash scripts/smoke_docker.sh   # prueba de humo y latencia
```

Ponga delante un proxy con TLS (Caddy, nginx) y defina `FORWARDED_ALLOW_IPS` con la IP del proxy.

### AWS ECS / Fargate

- Tarea: 2 vCPU / 4 GB (o 1 vCPU / 3 GB si la carga es baja), plataforma `linux/amd64`.
- Índice: volumen EFS montado en `/data/indice` (solo lectura) o `INDEX_URL` con una URL prefirmada de S3
  (`INDEX_DIR` en un volumen efímero de la tarea).
- Health check del contenedor (además del de la imagen):
  `["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]`,
  `startPeriod` 120.
- Health check del balanceador: ruta `/health`, código 200.
- `readonlyRootFilesystem: true` funciona si monta un volumen efímero en `/tmp`.

```json
{
  "family": "normalizador-direcciones",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "2048", "memory": "4096",
  "containerDefinitions": [{
    "name": "normalizador",
    "image": "<cuenta>.dkr.ecr.<region>.amazonaws.com/normalizador-direcciones:0.2.0-candidato",
    "portMappings": [{"containerPort": 8000}],
    "environment": [{"name": "NORMALIZER_BACKEND", "value": "onnx"}],
    "mountPoints": [{"sourceVolume": "indice", "containerPath": "/data/indice", "readOnly": true}],
    "logConfiguration": {"logDriver": "awslogs", "options": {"awslogs-group": "/ecs/normalizador",
      "awslogs-region": "<region>", "awslogs-stream-prefix": "app"}}
  }],
  "volumes": [{"name": "indice", "efsVolumeConfiguration": {"fileSystemId": "<fs-id>", "rootDirectory": "/indice"}}]
}
```

### AWS App Runner

App Runner no monta volúmenes: use `INDEX_URL` + `INDEX_SHA256` (URL prefirmada de S3 con vigencia suficiente, o
un objeto servido por CloudFront firmado) o la imagen privada `con-indice` en ECR. Configuración: 2 vCPU / 4 GB,
puerto 8000, health check HTTP en `/health`. La descarga ocurre en cada arranque de instancia.

### Google Cloud Run

```bash
gcloud run deploy normalizador \
  --image <region>-docker.pkg.dev/<proyecto>/<repo>/normalizador-direcciones:0.2.0-candidato \
  --cpu 2 --memory 4Gi --concurrency 1 --cpu-boost --no-cpu-throttling \
  --min-instances 1 --timeout 300 \
  --add-volume name=indice,type=cloud-storage,bucket=<bucket-privado>,readonly=true \
  --add-volume-mount volume=indice,mount-path=/data/indice \
  --startup-probe httpGet.path=/health,initialDelaySeconds=5,periodSeconds=5,failureThreshold=30
```

Cloud Run inyecta `PORT`. `--concurrency 1` coincide con el procesamiento de una petición a la vez. El montaje de
Cloud Storage lee el índice por red: el primer arranque es más lento que con disco local (no se midió). Alternativa:
`INDEX_URL` con una URL firmada de GCS, o la imagen `con-indice` en Artifact Registry privado. `--min-instances 1`
evita arranques en frío en cada petición.

### Railway

1. Cree un servicio desde el repositorio con *Root Directory* `normalizador_produccion` (usa el `Dockerfile`).
2. Agregue un volumen montado en `/data/indice` y copie allí el paquete (o defina `INDEX_URL` e `INDEX_SHA256`;
   con un volumen, `INDEX_DIR` es escribible y la descarga ocurre una sola vez).
3. Health check path: `/health`; plan con al menos 4 GB de memoria. Railway inyecta `PORT`.

### Lo que no se probó

Desde este equipo no es posible desplegar en ECS, App Runner, Cloud Run ni Railway: esos comandos no se ejecutaron y
sus tiempos no se midieron. Solo se probó la imagen localmente con Docker Desktop (sección 5). Para repetir la prueba
en una VM Linux en la nube:

```bash
sudo apt-get update && sudo apt-get install -y docker.io python3 git
git clone https://github.com/Juanpgm/HIIG && cd HIIG/normalizador_produccion
# copie el paquete del índice a ./indice
sudo INDEX_DIR=$PWD/indice CPUS=2 MEMORY=4g bash scripts/smoke_docker.sh
```

El script construye la imagen, comprueba que sin índice termina con código 3 y que con una variable inválida termina
con código 2, arranca el contenedor con raíz de solo lectura y sin privilegios, mide el tiempo hasta `/health` = 200,
la latencia p50/p95 de 200 peticiones y el rendimiento de un lote de 1.000 direcciones inventadas, y lo elimina al final.

## 7. Operación

- **Logs**: las líneas `[servicio]` muestran la configuración resuelta y los avisos (falta de mapas base, índice sin
  manifiesto, caída de int8 a fp32). Las peticiones solo se registran con `LOG_LEVEL=debug`.
- **Privacidad**: los archivos subidos se procesan en memoria y no se escriben a disco.
- **Seguridad del contenedor**: usuario 10001 sin shell, sin pip ni compiladores en la imagen final, compatible con
  raíz de solo lectura (`/tmp` escribible), `cap_drop: ALL` y `no-new-privileges`.
- **Actualizar el modelo**: un modelo nuevo requiere un índice nuevo (los embeddings dependen del modelo).
  Reemplace `modelo/`, reconstruya el índice con `preparar_indice.py` y despliegue ambos juntos; el manifiesto impide
  combinar un índice con otro modelo.

### Fijar la imagen base (opcional)

El `Dockerfile` usa la etiqueta móvil `python:3.12-slim`: es cómoda pero no reproducible, porque el contenido cambia
con cada reconstrucción de la imagen oficial. Para una construcción reproducible, fije el digest:

```bash
docker buildx imagetools inspect python:3.12-slim      # copie el campo Digest (sha256:...)
```

luego sustituya las dos líneas `FROM python:${PYTHON_VERSION}-slim` por
`FROM python:3.12-slim@sha256:<digest>` y revise el digest cada cierto tiempo para recibir parches de seguridad.

## 8. Qué se publica y qué no

| En git / en la imagen pública | Fuera de git (lo aporta el usuario) |
|---|---|
| código, pruebas, documentación | `catastro_docs.parquet`, `catastro_emb.npy`, `index_meta.json` |
| `modelo/model.onnx` (13 MB), `model.int8.onnx` (3,6 MB), `model.doc.onnx` (13 MB, solo en git; no entra a la imagen) | cachés derivadas (`*.pkl`), centroides |
| `onnx_meta.json`, `tuning.json`, `decision.json`, `reliability_calibrated.json` (parámetros y agregados) | mapas base IDESC, catastro, nomenclatura |

Los pesos de `modelo/` se entrenaron con consultas derivadas de datos de terceros (catastro y nomenclatura
municipal). La procedencia y los términos de redistribución de esos datos **no están determinados** (ver `NOTICE`);
esta carpeta no toma una decisión legal sobre ellos. Antes de una distribución pública amplia de los pesos, el
titular debería confirmar los términos de las fuentes.
