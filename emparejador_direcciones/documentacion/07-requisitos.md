# 07 - Requisitos

Dos conjuntos de requisitos independientes: el de **predicción** (usar el comparador) y el de **evaluación y calibración** (medir su calidad con su catastro). El modelo no tiene fase de entrenamiento ([04](04-evaluacion-y-calibracion.md)).

Documentos relacionados: [03 configuración](03-configuracion.md), [05 uso](05-uso-e-implementacion.md), [06 predicciones](06-predicciones.md).

## Requisitos para predicción

| Categoría | Requisito | Detalle |
|---|---|---|
| Software | Python 3.12 o superior | El README original indica que las versiones anteriores no se admiten; el proyecto se ejecutó en 3.14 |
| Software | Solo biblioteca estándar | Se usan `re`, `unicodedata`, `dataclasses` y, en los scripts, `argparse`, `csv`, `json`, `pathlib` |
| Software | Sin paquete instalable | No existe `pyproject.toml`; los scripts agregan `src/` a `sys.path` |
| Sistema operativo | Cualquiera con Python 3.12 o superior | Probado en Windows; no se verificó en otros sistemas (no hay CI) |
| Hardware | CPU de un solo hilo | No se requiere GPU, aceleradores ni archivos de pesos |
| Hardware | Memoria RAM | La memoria del índice crece linealmente con el número de registros; mídala con su propia base (sección H de `tools/eval_catastro.py`) |
| Hardware | Disco | Solo el código (menos de 300 KB de fuentes de runtime y herramientas); sin artefactos |
| Entrada: comparación de un par | Dos cadenas `str` en forma canónica | Ver [06](06-predicciones.md), sección 2 |
| Entrada: vinculación | Base CSV | Delimitador coma; codificación UTF-8 por defecto (se acepta otra con `--encoding`; BOM UTF-8 admitido); encabezado obligatorio con una columna de id y otra de dirección; las comillas se interpretan en modo estricto; las filas sin id se omiten |
| Entrada: tamaños | Sin tope explícito en el código | La base se carga completa en memoria; el rendimiento debe medirse con su propio volumen. La cola de complemento se limita a 40 caracteres y la placa a 6 dígitos significativos por dirección |
| Precondición aguas arriba | Normalizador | Las direcciones deben llegar ya normalizadas (en este ecosistema, `normalizador_direcciones`). Sin ese paso la mayoría no se podrá leer |
| Datos | Base de referencia | Una lista de pares (id, dirección canónica) de su territorio; el proyecto no incluye ninguna |
| No se requiere | GPU, pesos, entrenamiento, conexión a internet, variables de entorno, base de datos, servicios externos | - |

## Requisitos para evaluación y calibración

| Categoría | Requisito | Detalle |
|---|---|---|
| Software | Python 3.12 o superior | Igual que predicción |
| Software | `pyarrow==25.0.1` | Solo `tools/*.py`; se importa dentro de `main`. Está en `requirements-dev.txt` |
| Software | `pytest==9.0.3` | Solo para ejecutar la suite de pruebas del proyecto original; está en `requirements.txt`. La suite no se incluye en esta publicación |
| Software | Entorno virtual separado | Recomendado, para no mezclar `pyarrow` con el runtime |
| Hardware | CPU de un solo hilo; sin GPU | - |
| Hardware | Memoria RAM | Mida el pico de memoria del arnés con su propio catastro y planifique con holgura (el arnés mantiene además columnas del parquet) |
| Hardware | Tiempo | Depende del tamaño de la base y de la máquina; la corrida completa de `eval_catastro.py` es lenta |
| Datos: catastro (`eval_catastro.py`) | Archivo parquet | Columna `direccion` (texto) obligatoria y `numero_predial_nacional` (texto de dígitos) opcional; los nombres se pueden cambiar con `--address-col` y `--parcel-col` |
| Datos: catastro (`build_groundtruth.py`) | Archivo parquet | Columnas de texto `direccion`, `numero_predial_nacional`, `manzana`, `comuna`, `barrio` |
| Datos: etiquetas humanas | CSV o JSONL | Obligatorias `pair_id` y `same_door` (`yes`/`no`/`unsure`); opcional `same_unit`; UTF-8 |
| Datos: texto real etiquetado (`eval_realworld.py`) | Parquet | `raw_address` (o columna normalizada), `dataset`, `gt_doc_id`, `gt_quality` (solo en `--dev`); catastro con `doc_id`, `direccion`, `numero_predial_nacional` |
| Entrada: formato | Direcciones en forma canónica del modelo | Si las suyas siguen el estándar nacional sin `#` ni `-`, aplique primero el adaptador de [10](10-estandar-y-generalizacion.md) |
| Personas | Etiquetadores con acceso a evidencia catastral | Para las etiquetas humanas de los estratos difíciles (ver [04](04-evaluacion-y-calibracion.md)) |
| No se requiere | GPU, pesos, entrenamiento | - |

## Qué no es necesario

- No hay red neuronal: ni GPU, ni PyTorch, ni TensorFlow, ni archivos de pesos (ver [01](01-teoria.md)).
- No hay entrenamiento ni conjuntos de entrenamiento, validación y prueba en el sentido de aprendizaje automático.
- No hay gazetteer ni lista de barrios o ciudades: el código no contiene nombres de barrios ni de ciudades (verificado con búsqueda en `src/` y `scripts/`; solo aparecen el alias `BARRIO` y la palabra Cali en una cadena de documentación).
- No hay servicio ni base de datos: es una biblioteca y dos scripts.

## Lo que sí debe preparar

1. Un paso de normalización previo, con su propia validación.
2. Una base de referencia en forma canónica (para vinculación).
3. Si va a evaluar: un entorno con `pyarrow` y su catastro en parquet, además de etiquetas humanas para obtener métricas reales (ver [08](08-metricas-y-confiabilidad.md)).
