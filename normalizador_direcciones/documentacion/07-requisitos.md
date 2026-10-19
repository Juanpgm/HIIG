# 07. Requisitos

Requisitos de software, hardware, datos y artefactos para entrenar el modelo y para usarlo en predicción, más la política de artefactos publicados en este repositorio. Los pasos concretos están en [04-entrenamiento.md](04-entrenamiento.md) y [05-uso-e-implementacion.md](05-uso-e-implementacion.md).

## Contenido

1. [Requisitos para entrenamiento](#1-requisitos-para-entrenamiento)
2. [Requisitos para predicción](#2-requisitos-para-predicción)
3. [Esquemas de datos](#3-esquemas-de-datos)
4. [Artefactos requeridos al cargar](#4-artefactos-requeridos-al-cargar)
5. [Política de artefactos publicados](#5-política-de-artefactos-publicados)
6. [Acceso a red](#6-acceso-a-red)
7. [Qué es específico de Cali](#7-qué-es-específico-de-cali)

## 1. Requisitos para entrenamiento

| Categoría | Requisito | Detalle y fuente |
|-----------|-----------|------------------|
| Software | Python 3.11 o superior | `pyproject.toml` (`requires-python`). La imagen usa 3.12; el desarrollo se hizo con 3.14. |
| Software | torch con CUDA | `torch>=2.2` en `pyproject.toml`. Desarrollo con `torch 2.14.0+cu130`. El `Dockerfile` fija `torch==2.14.0` (CPU). |
| Software | Dependencias del núcleo | `numpy>=1.26`, `pandas>=2.1`, `pyarrow>=14`, `rapidfuzz>=3`, `shapely>=2`, `pyproj>=3.6`, `pyshp>=2.3`. |
| Software | Extra `train` | `scikit-learn>=1.3` (negativos espaciales), `requests>=2.31`, `nbformat>=5.9`, `nbclient>=0.9`, `unidecode>=1.3`. |
| Hardware | GPU NVIDIA con soporte bf16 | El pico de memoria de GPU depende del tamaño de lote. No se probó con menos memoria ni en CPU. |
| Hardware | RAM del sistema | No medida. `train_data.npz` contiene dos matrices de consultas de 48 bytes por fila más los índices; el resto es el DataFrame del catastro. |
| Tiempo | Depende de la GPU | Presupuesto por defecto `MAX_MINUTES` = 24 (el script detiene el entrenamiento al excederlo). |
| Datos | Tabla catastral cruda con el esquema de la sección 3.1 | Una fila por predio, con dirección, NPN de 30 dígitos, manzana y centroide. 338 311 filas en Cali. |
| Datos | **Ningún dato privado** para pasos 1 a 4 | La receta de producción es sintética. El ajuste de pesos y umbral usa solo el catastro. |
| Datos (opcional) | Filas reales con referencia conocida | Solo para ajustar `reliability.json` (2 o más fuentes, columnas descritas en [04](04-entrenamiento.md), sección 9) y para evaluar. |
| Datos (opcional) | Límites geográficos (2 GeoJSON, sección 3.3) | Para el gazetteer; necesarios para reproducir el comportamiento con puerta geográfica. |
| Red | Acceso a internet solo para obtener datos y paquetes | El script `download_catastro.py` consulta el servicio de entidades de ArcGIS indicado en `CALI_CATASTRO_URL` (sin valor por defecto); sustituible por cualquier otra fuente. |
| Sistema | Espacio en disco | Los artefactos de entrenamiento de Cali ocupan unos 415 MB sin experimentos; con los experimentos conservados, unos 2.1 GB. |

## 2. Requisitos para predicción

| Categoría | Requisito | Detalle y fuente |
|-----------|-----------|------------------|
| Software | Python 3.11 o superior | Imagen Docker: 3.12. |
| Software | torch (CPU basta) | `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Una instalación normal puede traer la versión CUDA (unos 3 GB). |
| Software | Núcleo más extras según el uso | `api` para el servicio; `excel` para `.xlsx`; `sql`; `geo`. Versiones fijadas para el servicio en `requirements.txt` (sección 9 de [03](03-configuracion.md)). |
| Hardware | CPU suficiente; GPU opcional | La imagen es de CPU. |
| Hardware | RAM de al menos 2 GB por instancia | Un límite de 1 GB o menos no alcanza; mida la memoria en su entorno. |
| Hardware | Disco | Embeddings 169 MB, `model.pt` 13.5 MB, índice 21 MB, centroides 8.3 MB, gazetteer 4.7 MB; la imagen Docker ocupa varios GB. |
| Datos | **Catastro propio de Cali** | Para construir `catastro_docs.parquet` y `catastro_own_centroids.parquet` (no se distribuyen). Esquema en la sección 3.1. |
| Datos | Entrada del usuario | Texto de dirección en una tabla o lista (sección 3.4). |
| Datos (opcional) | Límites geográficos | Sin ellos no hay detección de lugar ni puerta geográfica. |
| Red | No se necesita en la inferencia | El servicio no llama a servicios externos al normalizar. La URL de una fuente solo se usa si el usuario la entrega como entrada. |
| Límites | `MAX_ROWS` = 20 000 filas por petición (API); `MAX_UPLOAD_MB` = 25 | Configurables por variable de entorno. La CLI procesa por bloques de `--chunk-size` (20 000) sin tope de filas. |
| Límites | Longitud de texto | 40 caracteres útiles para la vista neuronal. |
| Cobertura | **Cali, zona urbana** | Ver sección 7. |

## 3. Esquemas de datos

### 3.1 Tabla catastral cruda (`catastro.parquet`)

Columnas obligatorias para `build_docs`:

| Columna | Tipo | Notas |
|---------|------|-------|
| `OBJECTID` | entero (`int64`) | Único; se usa para ordenar y desempatar. |
| `numero_predial_nacional` | texto | 30 dígitos numéricos continuos (estándar nacional del NPN). Se admiten registros anómalos; se marcan. |
| `numero_predial_manzana` | texto | Código de manzana (17 caracteres en la mayoría de los registros). |
| `direccion` | texto | Nomenclatura "vía # cruce - placa [complemento]". |
| `centroid_lon`, `centroid_lat` | decimal (`float64`) | Grados en EPSG:4326. |

Columnas opcionales: `geom_wkb` (polígono WKB EPSG:4326, para evaluación), `direccion_py`, `id_predio_sis`, `area_terreno`, `MAX_numero_pisos`, `Shape__Area`.

Codificación: texto Unicode. El analizador convierte a ASCII en mayúsculas; los símbolos fuera del vocabulario (por ejemplo `Ñ` en el texto catastral sin convertir) se mapean a `UNK` al codificar documentos.

### 3.2 Tabla de documentos (`catastro_docs.parquet`)

14 columnas; descripción completa en [04](04-entrenamiento.md), sección 2.2: `direccion`, `numero_predial_nacional`, `manzana`, `comuna`, `barrio`, `barrio_code`, `centroid_lon`, `centroid_lat`, `n_predios`, `manzana_nunique`, `objectid`, `x_m`, `y_m`, `doc_id`. El número de filas debe coincidir con el de `catastro_emb.pt` (el motor lo verifica con una aserción).

### 3.3 Límites geográficos (`basemaps/`)

Dos GeoJSON en EPSG:4326 (grados de longitud y latitud, sin `crs` declarado). Esquema que lee `gazetteer._read_basemaps`:

| Archivo | Geometría | Propiedades usadas |
|---------|-----------|--------------------|
| `barrios_veredas.geojson` | Polígonos o multipolígonos | Filas urbanas: `COMUNA` (no vacía), `BARRIO`, `ID_BARRIO`. Filas rurales: `COMUNA` vacía, `VEREDA`, `ID_VEREDA`, `CORREGIMIE`. En Cali, 440 elementos. |
| `comunas_corregimientos.geojson` | Polígonos | Comunas: `COMUNA` (número). Corregimientos: `CORREGIMIE`, `ID_CORREG`. En Cali, 37 elementos (22 comunas y 15 corregimientos). |

El rango de comunas válidas está fijo en 1 a 22 (`gazetteer.py:129`).

### 3.4 Entrada del usuario

Una cadena por dirección en la columna indicada. Tamaño: sin límite en la CLI (por bloques); en la API, 25 MB y 20 000 filas por defecto. Codificaciones de CSV admitidas por detección: UTF-8 (con o sin BOM), cp1252 y UTF-16. CRS de la entrada: irrelevante para la coincidencia (las coordenadas de la entrada nunca se usan); la salida está en EPSG:4326 y los cálculos internos de distancia en EPSG:3116.

## 4. Artefactos requeridos al cargar

"Obligatorio" indica lo que el motor no puede iniciar sin. "Distribuido" se refiere a **este repositorio publicado**.

| Archivo | Tamaño (copia fuente) | Función | Obligatorio | Distribuido aquí |
|---------|----------------------:|---------|:-----------:|:----------------:|
| `artifacts/model.pt` | 13 527 959 bytes (12.9 MiB) | Pesos del codificador, configuración, constantes de estandarización de coordenadas y códigos de comuna y barrio | sí | **sí** |
| `artifacts/tuning.json` | 5 424 bytes | Pesos de fusión, umbral y bloque opcional de decisión | no (hay valores por defecto en código) | **sí** |
| `artifacts/reliability.json` | 4 272 bytes | Modelo de confiabilidad calibrada | no (sin él, las columnas quedan `null`) | **sí** |
| `artifacts/reliability_v2.json` | 9 814 bytes | Calibrador alternativo de confiabilidad (v5, etiquetas consistentes con el texto); no se sirve por defecto | no | **sí** |
| `artifacts/train_report.json` | 5 232 bytes | Historia de entrenamiento (lo escribe `train.py`) | no | no (ya no se versiona) |
| `artifacts/interp_index_v1.pkl` + `.json`, `artifacts/catalog_keys_v1.pkl` + `.json` | 8 MB y 6 MB | Cachés perezosas (índice de interpolación y claves del catastro); se regeneran solas o con `deploy/prepare_artifacts.py` | no | **no** |
| `artifacts/catastro_docs.parquet` | 21 287 533 bytes | Índice de direcciones catastrales (330 387 filas) | sí | **no** |
| `artifacts/catastro_emb.pt` | 169 159 756 bytes (161.3 MiB) | Embeddings de los documentos, `(330 387, 256)` float16 | sí | **no** |
| `artifacts/catastro_own_centroids.parquet` | 8 309 094 bytes | Centroide del predio reportado | no (hay alternativas) | **no** |
| `artifacts/gazetteer.pkl` | 4 732 141 bytes | Caché de los límites ya procesados | no (se reconstruye desde `basemaps/`) | **no** |
| `basemaps/barrios_veredas.geojson`, `basemaps/comunas_corregimientos.geojson` | 8.2 y 3.5 MB | Límites de barrios, veredas, comunas y corregimientos | no (sin ellos se desactiva el gazetteer) | **no** |

Los tamaños en MiB frente a MB explican que la documentación original cite a veces 161 MB y a veces 169 MB para el mismo archivo `catastro_emb.pt`: son 161.3 MiB, es decir, 169.2 MB decimales.

El SHA-256 de `model.pt` en la copia fuente es `890b6d83eea29c3783e03416029d4b45682da0fadf411eb73f90b616954349df` (ver [05](05-uso-e-implementacion.md), sección 9).

## 5. Política de artefactos publicados

El repositorio publica:

- El **código** (`src/`, `scripts/`, `deploy/`, `Dockerfile`, `railway.toml`, `pyproject.toml`, `requirements.txt`, `uv.lock`).
- Los **pesos entrenados del codificador**, `artifacts/model.pt` (unos 13 MB).
- La **configuración calibrada**: `artifacts/tuning.json`, `artifacts/reliability.json` y `artifacts/reliability_v2.json` (solo coeficientes y metadatos; no contiene filas).
- **Resultados de ejecución no versionados.** Los resultados de ejecución (`strict_eval_*`, `eval_calibration.csv`, barridos de umbral, reportes de entrenamiento y de particiones, `artifacts/experiments/`) **ya no se versionan**: se regeneran con los scripts de `scripts/`.
- Esta documentación.

El repositorio **no publica**:

| No publicado | Motivo |
|--------------|--------|
| Índice catastral (`catastro_docs.parquet`) y centroides propios (`catastro_own_centroids.parquet`) | Son datos catastrales reales (direcciones y NPN de predios). La licencia de redistribución no está resuelta. |
| Embeddings (`catastro_emb.pt`, 169 MB) | Superan el límite por archivo de las plataformas de alojamiento y se pueden regenerar. |
| Mapas base (polígonos de la IDE municipal) | Licencia de redistribución no determinada. |
| Datos de entrenamiento, evaluación y prueba (`train_data.npz`, particiones, filas evaluadas) | Contienen direcciones reales o derivadas de fuentes con datos personales. |
| Contextos privados (registros de inmuebles, de personas afectadas y reportes de campo) | Datos personales. |

Consecuencia práctica: **la predicción exige que el usuario construya el índice con su propio catastro** (pasos en [05](05-uso-e-implementacion.md), sección 1). Con `model.pt` más un catastro de Cali propio basta; con el catastro de otra ciudad, `model.pt` no es adecuado (sección 7 y [10](10-estandar-y-generalizacion.md)).

Licencia: Apache-2.0 para el código y la documentación (ver `LICENSE` y `NOTICE` en la raíz; los datos de terceros y los pesos `model.pt` tienen las salvedades del `NOTICE`). Los datos que el usuario aporte conservan sus propias condiciones de uso.

## 6. Acceso a red

| Actividad | ¿Necesita red? | Detalle |
|-----------|:--------------:|---------|
| Instalar dependencias | Sí | PyPI e índice de ruedas de torch. |
| Obtener el catastro | Según la fuente | `download_catastro.py` consulta el servicio web indicado en `CALI_CATASTRO_URL`; con un archivo local no hace falta. |
| Entrenar | No | Después de tener los datos. |
| Normalizar con la CLI | No | Salvo que la entrada sea una URL o una base de datos remota. |
| Servir la API | No para normalizar | El contenedor solo necesita puerto de entrada. |
| Evaluación contra geocodificadores de terceros | No | El módulo de evaluación contra geocodificadores de terceros no se incluye en esta versión pública. |

## 7. Qué es específico de Cali

Resumen; la lista con archivo y línea está en [09-advertencias-y-limitaciones.md](09-advertencias-y-limitaciones.md), sección 2.

| Componente | Dependencia de Cali |
|------------|---------------------|
| `model.pt` | Entrenado con 330 387 direcciones de Cali. `comuna_head` (24 clases) y `barrio_head` (342 clases) y `coord_head` (media y desviación de coordenadas de Cali) son específicos. Los dos primeros no se usan al inferir; `coord_head` sí. |
| Índice y embeddings | Direcciones y NPN de Cali. |
| Parser | Convención de cuadrantes catastral y ruido de ciudad (`CALI`, `VALLE DEL CAUCA`). |
| Gazetteer | Esquema de los límites de Cali, comunas 1 a 22. |
| Generador de ruido | Sufijos de ciudad y barrios de Cali. |
| Umbrales | 0.73, holguras de 1000 y 300 m, `tau` de 600 m, ajustados con datos de Cali. |
| Puerta municipal | Solo reconoce Cali (`76001`). |
| Proyección | EPSG:3116 fija. |

Se recomienda reentrenar para cualquier otro municipio. Evaluación y recomendaciones de reutilización de `model.pt` en [10-estandar-y-generalizacion.md](10-estandar-y-generalizacion.md).
