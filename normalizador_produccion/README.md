# Normalizador de direcciones: imagen candidata para despliegue

> **Estado: imagen candidata / staging, NO apta para producción hasta aprobación del titular.** Los pesos de
> `modelo/` son el candidato del reentrenamiento 2026-10, cuyo veredicto de confiabilidad es **NO APRUEBA** (no
> promovido; ver `normalizador_direcciones/documentacion/14-reentrenamiento-2026-10.md`). El servicio en producción
> conserva el modelo anterior. Lo que está listo para producción es la herramienta de despliegue (imagen, validación,
> verificación del índice, pruebas); los pesos son un candidato.

Imagen de contenedor del normalizador de direcciones de Santiago de Cali. Sirve la misma API
HTTP que `normalizador_direcciones/` con el backend **ONNX Runtime** (fp32 por defecto), **sin torch**, como usuario
no root y con un `HEALTHCHECK`.

| | |
|---|---|
| Estado del modelo | **candidato, no aprobado** (veredicto NO APRUEBA; pendiente de decisión del titular) |
| Modelo | candidato final `v2g_cert_pcer_catfb_s42_r3_basecal` (exportado a ONNX, en `modelo/`) |
| Tamaño de la imagen (medido) | **744 MB en disco, 187 MB comprimida** (la estimación previa era 1,45 GB) |
| Arranque en frío (medido) | unos 4 s hasta `/health` = 200 con el índice en un volumen local |
| Latencia (medida, 4 vCPU) | 18 ms por dirección en el motor; 53 ms p50 por HTTP en Docker Desktop |
| Memoria en reposo (medida) | 1,3 a 1,5 GiB con el índice de 401.860 documentos |

La imagen **no trae el índice** (documentos y embeddings derivados del catastro y la nomenclatura). Se monta en
`/data/indice` o se descarga al arrancar. Si falta, el contenedor termina con código 3 y un mensaje que explica
cómo obtenerlo.

## Inicio rápido

```bash
# 1. Construya el paquete del índice a partir de sus datos (ver GUIA_DESPLIEGUE.md, "El índice")
python scripts/preparar_indice.py --origen /ruta/a/indice_construido --destino ./indice --precalentar

# 2. Construya y ejecute la imagen
docker build -t normalizador-direcciones:0.2.0-candidato .
docker run --rm -p 8000:8000 -v "$PWD/indice:/data/indice:ro" normalizador-direcciones:0.2.0-candidato

# 3. Pruebe
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/api/v1/normalize-address \
     -H 'content-type: application/json' \
     -d '{"addresses": ["CALLE 10 # 5 - 10", "Cra 11 #12-23"]}'
```

Con Docker Compose: `docker compose up -d --build` (monta `./indice` y, si existe, `./basemaps`).

La documentación interactiva de la API queda en `http://127.0.0.1:8000/docs`.

## Contenido de la carpeta

| Ruta | Qué es |
|---|---|
| `Dockerfile`, `.dockerignore`, `docker-compose.yml` | Imagen multietapa (sin compiladores ni pip en la etapa final) y despliegue local |
| `requirements.txt` / `requirements-dev.txt` | Dependencias fijadas del servicio / de las pruebas |
| `src/cali_address/` | Copia de los módulos de servicio de `normalizador_direcciones` (sin código de entrenamiento) |
| `src/servicio/` | Punto de entrada delgado: valida variables, verifica o descarga el índice, arma el directorio de ejecución y arranca uvicorn |
| `docs/API.md` | Guía de integración que sirve `/docs-integracion` |
| `modelo/` | Grafo ONNX fp32 e int8, metadatos y configuración de decisión (archivos pequeños) |
| `scripts/preparar_indice.py` | Construye el paquete del índice a partir de datos aportados por el usuario |
| `scripts/smoke_docker.sh`, `scripts/medir_servicio.py` | Prueba de humo y medición de latencia (local o en una VM) |
| `tests/` | Pruebas sin Docker (configuración, índice, API, punto de entrada) |
| `GUIA_DESPLIEGUE.md` | Configuración completa, ejemplos de la API, dimensionamiento y despliegue en cada plataforma |

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

No necesitan Docker ni datos reales: usan el modelo de `modelo/` y un índice sintético de 120 direcciones inventadas.

## El paquete del índice es una entrada de confianza

Al arrancar, el servicio verifica el sha256 de **cada** archivo del índice contra `indice_manifest.json` y se niega a
arrancar (código 3) si un archivo cambió, falta en el manifiesto o no coincide. Aun así, las cachés `.pkl` se cargan con
`pickle`, que puede ejecutar código arbitrario: trate el paquete como entrada de confianza, constrúyalo usted o
obténgalo de una fuente que controle, y no monte volúmenes escribibles por terceros. El manifiesto protege contra
corrupción y modificaciones accidentales; no sustituye la verificación de procedencia (`INDEX_SHA256` en la descarga).

## Licencia y procedencia

Código y documentación bajo Apache-2.0 (`LICENSE`, `NOTICE`). **Advertencia sobre los pesos:** los archivos de
`modelo/` provienen de entrenamientos con consultas derivadas de datos de terceros (catastro y nomenclatura
municipal); como indica `NOTICE`, su uso y redistribución pueden depender de los términos de esas fuentes, que el
titular no ha determinado. La licencia Apache-2.0 no concede derechos sobre datos de terceros. Este aviso no es
asesoría legal.
