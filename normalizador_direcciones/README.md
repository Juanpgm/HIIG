# Normalizador de direcciones de Cali

Sistema que convierte direcciones escritas como texto libre (`Cl. 5 # 34-12`, `Calle 5ta No 34 12`, `Cll 5 34 12`) en una referencia catastral de Santiago de Cali: dirección normalizada, Número Predial Nacional (NPN), manzana, coordenada del predio, barrio y comuna, con un estado explícito y un nivel de precisión por fila.

Un codificador Transformer a nivel de carácter (3 375 601 parámetros) recupera candidatos entre las direcciones del catastro; luego reglas estructurales y geográficas deciden si se acepta alguno. Cuando no hay una coincidencia aceptable, el sistema se abstiene en lugar de adivinar.

> **Estado: versión preliminar probada con datos de Cali.** No se ha medido fuera de Cali ni con fuentes distintas de las de la evaluación original. Lea [Advertencias de confiabilidad](#advertencias-de-confiabilidad) antes de usar los resultados.

## Contenido

1. [Alcance](#alcance)
2. [Qué se publica y qué no](#qué-se-publica-y-qué-no)
3. [Inicio rápido](#inicio-rápido)
4. [Columnas de salida, niveles de posición y reglas](#columnas-de-salida-niveles-de-posición-y-reglas)
5. [Advertencias de confiabilidad](#advertencias-de-confiabilidad)
6. [Requisitos resumidos](#requisitos-resumidos)
7. [Índice de documentación](#índice-de-documentación)
8. [Pruebas](#pruebas)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Licencias y datos de terceros](#licencias-y-datos-de-terceros)
11. [Licencia](#licencia)

## Alcance

- Normaliza direcciones **solo de Santiago de Cali**. El catálogo de referencia es el catastro urbano de Cali (330 387 direcciones únicas) y el modelo se entrenó con su gramática. Una dirección de otra ciudad se compararía con predios de Cali, lo cual es incorrecto.
- La entrada puede ser cualquier tabla (CSV, Excel, Parquet, JSON, GeoJSON, shapefile, SQL, ...). Se conservan las columnas originales y se añaden las de resultado.
- Interfaces: línea de comandos (`cali-address`), API HTTP (FastAPI), página de carga local y endpoints con forma Esri (no es un Locator certificado).

## Qué se publica y qué no

| Se publica | No se publica |
|------------|---------------|
| Código (`src/`, `scripts/`, `deploy/`, `Dockerfile`, `railway.toml`) | Índice catastral (`catastro_docs.parquet`, `catastro_own_centroids.parquet`) |
| Pesos entrenados del codificador: `artifacts/model.pt` (unos 13 MB) | Embeddings (`catastro_emb.pt`, 169 MB) |
| Configuración calibrada: `artifacts/tuning.json`, `artifacts/reliability.json`, `artifacts/reliability_v2.json` (solo coeficientes y metadatos) | Mapas base (polígonos de la infraestructura de datos municipal) |
| Pruebas automáticas (`tests/`, con direcciones sintéticas) y la guardia de publicación (`scripts/check_publishable.py`) | Datos de entrenamiento, evaluación y prueba, y resultados de ejecución (`strict_eval_*`, `eval_calibration.csv`, `artifacts/experiments/`, reportes de entrenamiento) |
| Documentación (`documentacion/` y `docs/`) | Contextos privados (registros con datos personales) y cachés regenerables (`interp_index_v1.*`, `catalog_keys_v1.*`, `gazetteer.pkl`) |

**Consecuencia: para predecir, el usuario debe construir el índice con su propio catastro de Cali** (instrucciones en [documentacion/05-uso-e-implementacion.md](documentacion/05-uso-e-implementacion.md), sección 1). El `model.pt` publicado es específico de Cali y no es adecuado para el catastro de otra ciudad ([documentacion/10-estandar-y-generalizacion.md](documentacion/10-estandar-y-generalizacion.md), sección 5). Detalle y motivos en [documentacion/07-requisitos.md](documentacion/07-requisitos.md), sección 5.

Seguridad: `model.pt` se carga con `torch.load(weights_only=False)` y el caché del gazetteer con `pickle`; ambos mecanismos pueden ejecutar código si el archivo está manipulado. Cargue solo artefactos de confianza y verifique su SHA-256 ([documentacion/05-uso-e-implementacion.md](documentacion/05-uso-e-implementacion.md), sección 9).

## Inicio rápido

Requiere Python 3.11 o superior.

```bash
# 1. Instalar (torch de CPU primero, para evitar la versión CUDA de unos 3 GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[api,excel]"
```

```python
# 2. Construir el índice con su catastro de Cali (esquema de columnas en documentacion/04-entrenamiento.md, sección 2.1)
import sys
sys.path.insert(0, "src")
import pandas as pd
from cali_address.dataset import build_docs, own_centroid_table

catastro = pd.read_parquet("artifacts/catastro.parquet")   # su tabla catastral
docs = build_docs(catastro)
docs.to_parquet("artifacts/catastro_docs.parquet", index=False)
own_centroid_table(docs, catastro).to_parquet("artifacts/catastro_own_centroids.parquet", index=False)
```

```bash
# 3. Regenerar catastro_emb.pt (no se versiona; rápido en GPU, más lento en CPU) y autoverificar
cali-address setup

# 4. Normalizar un archivo (escribe datos_normalizado.xlsx junto a la entrada)
cali-address normalize datos.xlsx
```

Opciones útiles:

```bash
cali-address inspect datos.csv                       # ver columnas y la columna de dirección sugerida
cali-address normalize datos.csv -o salida.csv --dry-run 20     # vista previa sin escribir
cali-address web                                     # página de carga en http://127.0.0.1:8000
```

Si el comando `cali-address` no está en su `PATH`, use `python -m cali_address ...`. Los límites geográficos (`basemaps/`) son opcionales: sin ellos el sistema funciona, pero sin detección de barrio ni comuna en el texto ni puerta geográfica.

## Columnas de salida, niveles de posición y reglas

Versión del paquete: **0.2.0** (`cali_address.__version__`, `pyproject.toml` y la API informan la misma). Se conservan las columnas originales y se añaden: `direccion_entrada`, `estado`, `motivo`, `direccion_normalizada`, `fuente_normalizacion`, `numero_predial_nacional`, `manzana`, `lat`, `lon`, `confianza`, `confiabilidad`, `confiabilidad_manzana`, `margen`, `nivel_precision`, `barrio_vereda`, `comuna_corregimiento`, `complemento_normalizado` y, **al final**, las diez columnas de posición, fidelidad y catálogo:

| Columna | Qué indica |
|---------|------------|
| `position_tier` | Qué tan bien ubicada quedó la fila: `predio`, `lote`, `tramo`, `manzana`, `via` o `none`. |
| `position_radius_m` | Radio de error geométrico del nivel, en metros (p90 derivado del catastro, sin GPS). |
| `position_source` | Origen de la coordenada: `parcel`, `interp` o `none`. |
| `selected_doc_id` | Documento catastral elegido (auditoría). |
| `override_rejected` | `True` si el candidato se rechazó por las reglas estructurales. |
| `direccion_catastral` | Dirección del predio coincidente tal como la escribe el catastro (solo filas `OK`). |
| `fidelity_risk`, `fidelity_flags` | Riesgo (`none`, `low`, `medium`, `high`) de que la dirección devuelta sea **otra** que la recibida, y las banderas que lo explican. |
| `catalog_status`, `address_verified` | `verified`, `ambiguous`, `door_not_catalogued`, `rejected_mismatch`, `unparsed` o `below_cutoff`; `address_verified` es `True` solo para `verified`. |

**Niveles de posición y radios geométricos** (derivados del propio catastro; no miden acierto frente a la realidad de campo):

| `position_tier` | Radio | Coordenada | Predial |
|-----------------|------:|------------|---------|
| `predio` | 15 m | Centroide del predio | Se conserva |
| `lote` | 44 m | Centroide del lote (una unidad de sitio difiere del catastro) | Retenido |
| `tramo` | 15 a 23 m | Interpolada entre predios vecinos del mismo costado (placa a 1 o 2 unidades, letras o `BIS` distintos) | Retenido |
| `manzana` | 56 m | Punto del mejor candidato (letra o cuadrante en un solo lado, dirección compartida) | Se conserva, salvo letra o cuadrante en un solo lado |
| `via` | 21 m | **Retenida** por defecto (`EXTRAPOLATED_COORDS_ENABLED = False`): placa más allá del último predio, parse truncado o fidelidad alta | Retenido (también la manzana) |
| `none` | - | - | - |

**Dirección en blanco.** `direccion_normalizada` (y `fuente_normalizacion`) queda `null` cuando el texto no es confiable: filas `unparsed`; filas `below_cutoff` o `rejected_mismatch` con un análisis inseguro (solo se emite con fidelidad `none` o `low` y parse completo); y **toda** fila con `fidelity_risk = high`, también las `OK`. La canalización de archivos recurre entonces al texto original.

**Medida de fidelidad.** Compara la entrada original con la dirección devuelta usando un tokenizador independiente del analizador de producción. Lo inofensivo (abreviaturas, mayúsculas, tildes, `No.` a `#`, `N` a `NORTE` en el mismo segmento) no cuenta; sí cuentan los números de puerta, las letras pegadas a ellos, el tipo de vía y el cuadrante (su posición es significativa). `high` limita la fila a `via` y la deja sin dirección; `medium`, a `manzana` sin predial.

**Reglas de la guarda estructural.** El cuadrante se compara **por posición** (vía, cruce, final): un cuadrante movido de posición se rechaza. `N` equivale a `NORTE` solo dentro del mismo segmento. **Una placa distinta nunca es una puerta verificada** (como máximo `tramo`). Un sufijo numérico en un solo lado (`KR 26 H` frente a `KR 26 H 3`) se rechaza con `regla: sufijo unilateral`.

**Cachés perezosas (no se versionan).** `artifacts/interp_index_v1.pkl` + `.json` (índice de interpolación, ~8 MB) y `artifacts/catalog_keys_v1.pkl` + `.json` (claves exactas del catastro, ~6 MB) se construyen la primera vez que se usan y se verifican por hash antes de leerlas. Para construirlas de antemano: `python deploy/prepare_artifacts.py` (`--no-warm` las omite).

**Confiabilidad y artefactos.** `artifacts/reliability.json` (v4) es el calibrador que se sirve; `artifacts/reliability_v2.json` (v5, ajustado solo con etiquetas consistentes con el texto) no se sirve por defecto y cambia la escala. **Las bandas de confiabilidad no están validadas**: ni la escala operativa ni los cortes se han medido fuera de Cali ni con otras fuentes ([documentacion/08-metricas-y-confiabilidad.md](documentacion/08-metricas-y-confiabilidad.md), [documentacion/09-advertencias-y-limitaciones.md](documentacion/09-advertencias-y-limitaciones.md)). `catastro_emb.pt` no se versiona: `cali-address setup` lo regenera en CPU.

Detalle campo por campo en [documentacion/06-predicciones.md](documentacion/06-predicciones.md), sección 10, y en [docs/API.md](docs/API.md).

## Advertencias de confiabilidad

Resumen; la definición de las métricas, el protocolo y los comandos para reproducirlas localmente están en [documentacion/08-metricas-y-confiabilidad.md](documentacion/08-metricas-y-confiabilidad.md) y las advertencias en [documentacion/09-advertencias-y-limitaciones.md](documentacion/09-advertencias-y-limitaciones.md). Los resultados de las corridas no se publican.

- **El resultado es una propuesta, no una verdad.** Solo una parte de las filas reales llega a `OK`, y de esas no todas aciertan la manzana ni el predio, medido contra una referencia por GPS que es ruidosa. El subconjunto cuya dirección escrita coincide con la del predio de referencia ("gold") da una cota optimista porque se elige por coincidencia textual.
- **`confianza` no es una probabilidad.** Una `confianza` alta no implica que el predio sea exacto.
- **La `confiabilidad` que se sirve (v4, `reliability.json`) casi no distingue filas correctas de incorrectas** y su escala no está calibrada: se ajustó con etiquetas GPS ruidosas. Hay un calibrador alternativo (v5, `reliability_v2.json`), ajustado solo con etiquetas consistentes con el texto; no se sirve por defecto, usa otra escala y no mejora la manzana. Verifique cualquier calibrador con su propia muestra (ver [documentacion/08-metricas-y-confiabilidad.md](documentacion/08-metricas-y-confiabilidad.md), sección 4).
- **Las métricas de una corrida dependen de la versión del código**; una auditoría posterior cambió reglas de decisión, así que mida de nuevo con el código vigente.
- **Las bandas de confiabilidad no están validadas** y los radios de posición son geométricos (derivados del catastro), no una medida de acierto.
- **No hay medición fuera de Cali**, en zona rural, ni con otras fuentes de texto.
- **No use el resultado sin verificación humana en decisiones jurídicas, catastrales o tributarias.**
- Escala operativa sugerida (aceptable con muestreo, revisión humana, rechazo): [documentacion/08-metricas-y-confiabilidad.md](documentacion/08-metricas-y-confiabilidad.md), sección 5.

## Validación con JEV (opcional)

Juez semántico externo e independiente. **Envía cadenas de dirección a un servicio de terceros**, por lo que su uso es una decisión explícita de quien lo ejecuta; consulte [`validacion_jev/README.md`](../validacion_jev/README.md) antes de usarlo.

## Requisitos resumidos

Tablas completas en [documentacion/07-requisitos.md](documentacion/07-requisitos.md).

### Requisitos para entrenamiento

| Categoría | Requisito |
|-----------|-----------|
| Software | Python 3.11 o superior; torch con CUDA; extra `train` (`scikit-learn`, `requests`, etc.) |
| Hardware | GPU NVIDIA con bf16; el pico de memoria de GPU depende del tamaño de lote (ver [documentacion/04-entrenamiento.md](documentacion/04-entrenamiento.md)) |
| Tiempo | Depende de la GPU; el presupuesto por defecto es `MAX_MINUTES` (24) |
| Datos | Tabla catastral cruda: `OBJECTID`, `numero_predial_nacional` (30 dígitos), `numero_predial_manzana`, `direccion`, `centroid_lon`, `centroid_lat` |
| Datos privados | Ninguno para entrenar ni para ajustar pesos y umbral; solo hacen falta filas reales con referencia para ajustar la confiabilidad |
| CRS | Centroides en EPSG:4326; cálculo métrico en EPSG:3116 |

### Requisitos para predicción

| Categoría | Requisito |
|-----------|-----------|
| Software | Python 3.11 o superior; torch de CPU; extras según uso (`api`, `excel`, `sql`, `geo`) |
| Hardware | CPU; al menos 2 GB de RAM por instancia (un límite de 1 GB o menos no alcanza) |
| Artefactos publicados | `model.pt`, `tuning.json`, `reliability.json` |
| Artefactos que debe generar | `catastro_docs.parquet`, `catastro_own_centroids.parquet`, `catastro_emb.pt` (con `cali-address setup`); opcionalmente `basemaps/` |
| Límites | API: `MAX_ROWS` 20 000 filas y `MAX_UPLOAD_MB` 25 por petición |
| Cobertura | Santiago de Cali, zona urbana |

## Índice de documentación

Toda la documentación nueva está en [documentacion/](documentacion/README.md), en español. Los documentos originales del proyecto están en `docs/`.

| Documento | Contenido |
|-----------|-----------|
| [01-teoria.md](documentacion/01-teoria.md) | Recuperación con codificador dual, InfoNCE, nivel de carácter, corrupción sintética, negativos duros, fusión, reglas y confiabilidad |
| [02-arquitectura.md](documentacion/02-arquitectura.md) | Módulos, diagramas de la red, formas de tensor, datos de entrenamiento, flujo de inferencia y despliegue |
| [03-configuracion.md](documentacion/03-configuracion.md) | Variables de entorno, opciones de la CLI, archivos de configuración, Docker y Railway |
| [04-entrenamiento.md](documentacion/04-entrenamiento.md) | Receta paso a paso, esquemas de datos, hiperparámetros, tiempos, ajuste de umbral y confiabilidad |
| [05-uso-e-implementacion.md](documentacion/05-uso-e-implementacion.md) | Instalación, CLI, API, formatos, despliegue, Esri y seguridad |
| [06-predicciones.md](documentacion/06-predicciones.md) | Entradas, columnas de salida, valores de `estado` y `motivo`, guía de interpretación |
| [07-requisitos.md](documentacion/07-requisitos.md) | Requisitos de entrenamiento y de predicción, esquemas, artefactos y política de publicación |
| [08-metricas-y-confiabilidad.md](documentacion/08-metricas-y-confiabilidad.md) | Definición de métricas, protocolo, cómo reproducirlas y escala operativa |
| [09-advertencias-y-limitaciones.md](documentacion/09-advertencias-y-limitaciones.md) | Advertencias numeradas, elementos específicos de Cali e inconsistencias |
| [10-estandar-y-generalizacion.md](documentacion/10-estandar-y-generalizacion.md) | Conformidad con el estándar nacional y cómo adaptar a otros territorios |

## Pruebas

```bash
pip install -e ".[dev,api,sql,excel]"
python -m pytest -q
```

Las pruebas usan modelos de juguete y direcciones **sintéticas**. Las marcadas `requires_artifacts` necesitan los artefactos reales de servicio (`model.pt`, `catastro_emb.pt`, `catastro_docs.parquet`) y se omiten si faltan; en un clon limpio faltan `catastro_emb.pt` y el índice catastral, de modo que esas pruebas se omiten hasta que se construyan (ver [Inicio rápido](#inicio-rápido)).

## Estructura del repositorio

| Ruta | Contenido |
|------|-----------|
| `src/cali_address/` | Paquete: analizador, modelo, inferencia, gazetteer, capa de decisión (`service`), `io/`, `cli.py`, `api/` |
| `scripts/` | Preparación de datos, entrenamiento, ajuste, evaluación, partición de datos, ajuste de confiabilidad y el escáner local de direcciones reales (`check_no_real_addresses.py`) |
| `tests/` | Suite de pytest con datos sintéticos |
| `artifacts/` | `model.pt`, `tuning.json`, `reliability.json`, `reliability_v2.json` y `catastro_meta.json` |
| `deploy/` | `prepare_artifacts.py`, que reúne los archivos de servicio para la imagen |
| `docs/` | Documentos originales del proyecto (API, integración Esri, despliegue, artefactos, experimentos, particiones) |
| `documentacion/` | Documentación completa en español (esta serie) |
| `Dockerfile`, `railway.toml`, `.dockerignore`, `.railwayignore` | Despliegue |
| `pyproject.toml`, `requirements.txt`, `uv.lock` | Dependencias |
| `CHANGELOG.md` | Registro de cambios |

## Licencias y datos de terceros

- **Licencia del código.** El repositorio todavía no incluye un archivo `LICENSE`: la licencia está pendiente de decisión del titular. Hasta que exista, no se concede ningún permiso de uso o redistribución más allá de los que otorgue la plataforma de alojamiento.
- **Datos de terceros que no se redistribuyen.** La capa catastral de Cali y los mapas base municipales (barrios, veredas, comunas y corregimientos) son datos de terceros y **no forman parte de este repositorio**. Cada usuario debe verificar los términos de uso y de redistribución de las fuentes que emplee antes de descargarlas, derivar productos de ellas o publicarlas.
- **Pesos entrenados.** `artifacts/model.pt` se entrenó con ruido sintético derivado de un catastro (no contiene filas del catastro). Aun así, su uso y redistribución dependen de los términos de la fuente catastral de la que deriva.
- **Cómo obtener los archivos que no se publican.** Genérelos localmente; el `.gitignore` ya los excluye (`*.parquet`, `*.pt` salvo `model.pt`, `*.geojson`, `basemaps/`) y no deben incorporarse al repositorio con `git add -f`:

| Archivo | Cómo se obtiene |
|---------|-----------------|
| `artifacts/catastro.parquet` | `python scripts/download_catastro.py` (descarga paginada y reanudable de la capa catastral; requiere la variable de entorno `CALI_CATASTRO_URL` con una fuente que usted esté autorizado a consultar, sin valor por defecto; verifique antes los términos del servicio) |
| `artifacts/catastro_docs.parquet` | `cali_address.dataset.build_docs` (paso 2 del inicio rápido) |
| `artifacts/catastro_own_centroids.parquet` | `own_centroid_table` (paso 2 del inicio rápido) o `python scripts/build_own_centroids.py` |
| `artifacts/catastro_emb.pt` | `cali-address setup` (regenera los embeddings a partir de `model.pt` y del índice) |
| `basemaps/barrios_veredas.geojson`, `basemaps/comunas_corregimientos.geojson` | No hay script: descárguelos del portal de datos abiertos o del geoportal municipal y guárdelos con esos nombres. Son opcionales; sin ellos no hay detección de barrio ni comuna en el texto |

- **Seguridad: deserialización.** `model.pt` se carga con `torch.load(..., weights_only=False)` y el caché del gazetteer (`gazetteer.pkl`) con `pickle`; ambos mecanismos pueden ejecutar código arbitrario si el archivo está manipulado. Cargue únicamente artefactos de origen confiable y compare su SHA-256 antes de usarlos ([documentacion/09-advertencias-y-limitaciones.md](documentacion/09-advertencias-y-limitaciones.md), advertencia 11).

## Licencia

Licencia: por definir (ver LICENSE en la raíz cuando exista).
