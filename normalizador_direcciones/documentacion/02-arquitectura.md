# 02. Arquitectura

Este documento describe los módulos del paquete `cali_address`, la red neuronal (con sus formas de tensor verificadas contra `src/cali_address/model.py` y contra el checkpoint `artifacts/model.pt`), el flujo de datos de entrenamiento, el flujo completo de inferencia y la topología de despliegue. Los fundamentos conceptuales están en [01-teoria.md](01-teoria.md); los parámetros de configuración, en [03-configuracion.md](03-configuracion.md).

## Contenido

1. [Mapa de módulos](#1-mapa-de-módulos)
2. [Red neuronal](#2-red-neuronal)
   - [a) Diagrama de bloques](#a-diagrama-de-bloques-de-la-red)
   - [b) Formas de tensor por capa](#b-formas-de-tensor-por-capa)
   - [Conteo de parámetros](#conteo-de-parámetros)
   - [Discrepancia entre los valores por defecto de `model.py` y el checkpoint](#discrepancia-entre-los-valores-por-defecto-de-modelpy-y-el-checkpoint)
3. [c) Canalización de datos de entrenamiento](#3-c-canalización-de-datos-de-entrenamiento)
4. [d) Flujo de inferencia de extremo a extremo](#4-d-flujo-de-inferencia-de-extremo-a-extremo)
5. [e) Diagrama de despliegue](#5-e-diagrama-de-despliegue)
6. [Superficies de uso (CLI y API)](#6-superficies-de-uso-cli-y-api)

## 1. Mapa de módulos

Todos los módulos están en `src/cali_address/`. Se listan por capa.

| Capa | Módulo | Responsabilidad |
|------|--------|-----------------|
| Gramática | `parser.py` | Normaliza texto (mayúsculas, ASCII, separadores) y produce `ParsedAddress` (tipo de vía, número, letras, sufijo, BIS, cuadrante, cruce, placa, complemento). Nunca lanza excepciones: una entrada no reconocida produce `parse_ok=False`. |
| Gramática | `complement_lexicon.py` | Léxico de complementos (torre, bloque, apartamento, etc.) usado por la etapa de complemento. |
| Territorio | `gazetteer.py` | Detecta barrio, vereda, comuna y corregimiento en el texto, los retira antes del análisis gramatical y filtra candidatos geográficamente (`gate_candidates`). Se alimenta de dos archivos GeoJSON de límites (no incluidos en este repositorio). |
| Territorio | `geo.py` | Proyección a EPSG:3116, distancia haversine y búsqueda punto-en-polígono (`ParcelIndex`). |
| Modelo | `model.py` | `AddressEncoder`: codificador Transformer a nivel de carácter con tres cabezas auxiliares. Vocabulario y codificación a enteros. |
| Entrenamiento | `augment.py` | Corrupción sintética: 18 operadores. |
| Entrenamiento | `realistic_noise.py`, `realpairs.py` | Variantes experimentales: ruido calibrado con cadenas reales y pares reales. No se usan en la receta de producción. |
| Entrenamiento | `dataset.py` | Construye el índice catastral (`build_docs`), el centroide propio de cada predio (`own_centroid_table`), los tensores de entrenamiento y los negativos duros (espaciales y de placa). |
| Entrenamiento | `train.py` | Bucle de entrenamiento, `embed_documents`, `recall_at_k`, guardado y carga de checkpoints. |
| Entrenamiento | `experiment.py`, `paths.py` | Infraestructura de experimentos aislados y resolución de directorios de artefactos. |
| Inferencia | `inference.py` | `AddressNormalizer`: carga el checkpoint y los embeddings, busca los 20 vecinos más cercanos por vista y calcula los componentes de puntaje (`score_components`). |
| Decisión | `service.py` | `normalize_strict`: puerta geográfica, umbral, reglas estrictas, ambigüedad, nivel de precisión y columnas de salida. |
| Decisión | `reliability.py` | Carga `reliability.json` y calcula `confiabilidad` y `confiabilidad_manzana`. |
| E/S | `tables.py`, `io/` | Lectura de tablas (CSV, Excel, Parquet, JSON, GeoJSON, shapefile, GeoPackage, SQL), mapeo de columnas, escritura atómica y canalización por bloques. |
| Interfaces | `cli.py`, `__main__.py` | Comandos `setup`, `normalize`, `web`, `inspect`, `formats`. |
| Interfaces | `api/main.py`, `api/schemas.py`, `api/esri.py` | Servicio FastAPI y router con forma Esri. |
| Evaluación | `evaluate.py`, `labels.py`, `eval_data.py` | Métricas estrictas y calidad de etiquetas. Solo para evaluación fuera de línea. |
| Evaluación | `evaluate.py` | Métricas de evaluación contra datos de referencia. El módulo de evaluación contra geocodificadores de terceros no se incluye en esta versión pública. |
| Arranque | `bootstrap.py` | Regenera `catastro_emb.pt` y ejecuta la autoverificación de `cali-address setup`. |

## 2. Red neuronal

El modelo es un **codificador dual con pesos compartidos**: un único Transformer a nivel de carácter embebe tanto la consulta (texto ruidoso) como el documento (dirección catastral canónica). Un embedding de rol (`role`, 0 para consulta y 1 para documento) indica al codificador de qué lado proviene la cadena. La recuperación es una búsqueda por similitud coseno en un espacio único de 256 dimensiones.

Los valores de esta sección se verificaron cargando `artifacts/model.pt` (diccionario con claves `state_dict`, `config`, `n_comuna`, `n_barrio`, `xy_mean`, `xy_std`, `comuna_codes`, `barrio_codes`) y comparando con `model.py`.

### a) Diagrama de bloques de la red

```mermaid
flowchart TB
    IN["Texto (consulta o documento)<br/>ASCII en mayúsculas, máx. 40 caracteres"]
    TKN["Tokenización por carácter<br/>vocabulario 48 = 46 símbolos + PAD + UNK"]
    EMB["Suma de tres embeddings<br/>tok (48x256) + pos (40x256) + role (2x256)"]
    DRP["Dropout 0.1"]
    subgraph ENC["Codificador: 4 capas TransformerEncoderLayer, pre-LN"]
        direction TB
        L1["LayerNorm"] --> MHA["Autoatención multicabeza<br/>8 cabezas, d_cabeza = 32<br/>máscara de relleno"]
        MHA --> R1["Suma residual"]
        R1 --> L2["LayerNorm"]
        L2 --> FFN["Feed-forward: Linear 256-1024, GELU, Linear 1024-256"]
        FFN --> R2["Suma residual"]
    end
    POOL["Promedio sobre posiciones no PAD<br/>(B, 40, 256) a (B, 256)"]
    LN["LayerNorm 256"]
    PROJ["Linear 256 a 256"]
    L2N["Normalización L2"]
    OUT["Embedding de recuperación (B, 256)"]
    H1["coord_head<br/>Linear 256-128, GELU, Linear 128-2<br/>salida (B, 2), x/y estandarizados"]
    H2["comuna_head<br/>Linear 256-24<br/>logits (B, 24)"]
    H3["barrio_head<br/>Linear 256-342<br/>logits (B, 342)"]

    IN --> TKN --> EMB --> DRP --> ENC --> POOL --> LN
    LN --> PROJ --> L2N --> OUT
    LN --> H1
    LN --> H2
    LN --> H3
```

Notas de lectura:

- Las tres cabezas auxiliares leen la salida de `LayerNorm` (vector agrupado), no la proyección final.
- **En inferencia solo se usan el embedding de recuperación y `coord_head`.** `comuna_head` y `barrio_head` existen únicamente para dar señal durante el entrenamiento (verificado: ningún módulo fuera de `model.py` y `train.py` las referencia).
- El `TransformerEncoder` se construye sin normalización final por token; la `LayerNorm` posterior se aplica una sola vez, sobre el vector agrupado.
- Una fila completamente rellena haría que el softmax produjera NaN; `AddressEncoder.pool` mantiene viva la posición 0 para evitarlo.
- `logit_scale` es un parámetro aprendible (escalar). Se inicializa en `ln(1/0.05) = ln(20)` y se limita a `exp(.) <= 100` durante el entrenamiento.

### b) Formas de tensor por capa

Con `B` = tamaño de lote y `L` = 40 (longitud máxima del checkpoint), `d` = 256, `h` = 8, `d_h` = 32, `d_ff` = 1024.

| Etapa | Operación | Forma de entrada | Forma de salida | Parámetros |
|-------|-----------|------------------|-----------------|-----------:|
| 0 | Identificadores de carácter | texto | `(B, 40)` int64 | 0 |
| 1 | `tok`: Embedding(48, 256), `padding_idx=0` | `(B, 40)` | `(B, 40, 256)` | 12 288 |
| 2 | `pos`: Embedding(40, 256) | `(1, 40)` | `(1, 40, 256)` (difusión) | 10 240 |
| 3 | `role`: Embedding(2, 256), fila 0 o 1 | escalar | `(1, 1, 256)` (difusión) | 512 |
| 4 | Suma `tok + pos + role` y Dropout 0.1 | tres tensores | `(B, 40, 256)` | 0 |
| 5 | Por capa (x4): `LayerNorm` (norm1) | `(B, 40, 256)` | `(B, 40, 256)` | 512 |
| 5 | Por capa: `in_proj` de atención (Q, K, V juntos) | `(B, 40, 256)` | `(B, 40, 768)`, reorganizado a 3 x `(B, 8, 40, 32)` | 197 376 |
| 5 | Por capa: puntajes de atención `QK^T / sqrt(32)`, máscara, softmax | `(B, 8, 40, 32)` x2 | `(B, 8, 40, 40)` | 0 |
| 5 | Por capa: contexto y `out_proj` | `(B, 8, 40, 32)` | `(B, 40, 256)` | 65 792 |
| 5 | Por capa: suma residual | dos tensores | `(B, 40, 256)` | 0 |
| 5 | Por capa: `LayerNorm` (norm2) | `(B, 40, 256)` | `(B, 40, 256)` | 512 |
| 5 | Por capa: `linear1` 256 a 1024 y GELU | `(B, 40, 256)` | `(B, 40, 1024)` | 263 168 |
| 5 | Por capa: `linear2` 1024 a 256 | `(B, 40, 1024)` | `(B, 40, 256)` | 262 400 |
| 5 | Por capa: suma residual | dos tensores | `(B, 40, 256)` | 0 |
| 6 | Promedio sobre posiciones no PAD | `(B, 40, 256)` | `(B, 256)` | 0 |
| 7 | `norm`: LayerNorm(256) | `(B, 256)` | `(B, 256)` | 512 |
| 8 | `proj`: Linear(256, 256) | `(B, 256)` | `(B, 256)` | 65 792 |
| 9 | Normalización L2 | `(B, 256)` | `(B, 256)` unitario | 0 |
| 10a | `coord_head`: Linear(256, 128), GELU, Linear(128, 2) | `(B, 256)` | `(B, 2)` | 33 154 |
| 10b | `comuna_head`: Linear(256, 24) | `(B, 256)` | `(B, 24)` | 6 168 |
| 10c | `barrio_head`: Linear(256, 342) | `(B, 256)` | `(B, 342)` | 87 894 |
| - | `logit_scale` | - | escalar | 1 |

La capa del paso 5 pesa 789 760 parámetros y se repite cuatro veces (3 159 040).

Tensores del lado del índice y de la inferencia:

| Tensor | Forma y tipo | Notas |
|--------|--------------|-------|
| `d_tokens` (almacenado) | `(330 387, 48)` uint8 | Se trunca a 40 al entrenar y al embeber. |
| `q_tokens`, `q_tokens_raw` (almacenados) | `(991 161, 48)` uint8 | Dos vistas de cada consulta; se truncan a 40. |
| `doc_xy` | `(330 387, 2)` float32 | Coordenadas EPSG:3116 estandarizadas con `xy_mean` y `xy_std`. |
| `spatial_negatives.npy` | `(330 387, 8)` int64 | Ocho negativos espaciales por documento. |
| `catastro_emb.pt` | `(330 387, 256)` float16 | 169 159 756 bytes. Derivable de `model.pt` y `catastro_docs.parquet`. |
| Logits de entrenamiento | `(B, U)` | `U` = documentos únicos del lote (positivos más negativos duros), con `B` = 2048. |
| Candidatos por consulta | hasta 40 | 20 por vista, unidos y sin duplicados. |

### Conteo de parámetros

| Bloque | Parámetros |
|--------|-----------:|
| Embeddings (`tok`, `pos`, `role`) | 23 040 |
| 4 capas del codificador | 3 159 040 |
| `norm` y `proj` | 66 304 |
| `coord_head` | 33 154 |
| `comuna_head` | 6 168 |
| `barrio_head` | 87 894 |
| `logit_scale` | 1 |
| **Total** | **3 375 601** |

El total coincide con `n_params` de `artifacts/train_report.json` y con la suma de los tensores del `state_dict`. El camino de recuperación (sin `comuna_head` ni `barrio_head`) usa 3 281 539 parámetros.

### Discrepancia entre los valores por defecto de `model.py` y el checkpoint

La clase `AddressEncoder` define valores por defecto que **no coinciden** con el modelo publicado. Lo que se carga es siempre la configuración almacenada en el checkpoint (`inference.AddressNormalizer` y `train.load_checkpoint` construyen el modelo a partir de `ckpt["config"]`).

| Parámetro | Valor por defecto en `model.py` | Valor en el checkpoint | Efecto |
|-----------|--------------------------------|------------------------|--------|
| `n_layers` | 5 | **4** | Un modelo construido con los valores por defecto no puede cargar el `state_dict`. |
| `max_len` / `MAX_LEN` | 48 (constante del módulo) | **40** | Las matrices de tokens se almacenan a 48 y se truncan a 40; `pos` tiene 40 filas. |
| `n_barrio` | 400 | **342** | Solo afecta a `barrio_head`. |
| `n_comuna` | 24 | 24 | Coincide. |
| Vocabulario | 48 | 48 | Coincide (46 símbolos más PAD y UNK). |

Recomendación para quien extienda el código: construir siempre el modelo desde `config` y no confiar en los valores por defecto del constructor.

El `config` del checkpoint no contiene `label_smoothing`, `init_temperature`, `pct_start`, `div_factor` ni `final_div_factor`. Los valores de la receta (0.02, 0.05, 0.15, 25 y 100) provienen de los valores por defecto actuales de `TrainConfig`; no se pudo verificar que fueran idénticos al momento del entrenamiento original.

## 3. c) Canalización de datos de entrenamiento

El modelo se entrena **solo con datos sintéticos**: cada dirección del catastro se corrompe con los 18 operadores de `augment.py` para simular el texto que escriben las personas. La receta de producción no usa pares reales (el `config` del checkpoint no contiene `real_frac`).

```mermaid
flowchart TB
    CAT["Tabla catastral cruda<br/>(catastro.parquet)"]
    BD["build_docs<br/>una fila por dirección única<br/>comuna = NPN[9:11], barrio_code = NPN[9:13]<br/>centroide mediano, x_m / y_m en EPSG:3116"]
    DOCS["catastro_docs.parquet<br/>330 387 x 14"]
    OWN["own_centroid_table<br/>catastro_own_centroids.parquet"]
    NEG["build_spatial_negatives<br/>8 negativos por documento"]
    SPLIT["Partición por documento<br/>4 % de documentos a validación"]
    AUG["corrupt_many (3 variantes por dirección)<br/>variante 0 con intensidad 0.45<br/>variantes 1 y 2 con intensidad 1.0"]
    OPS["18 operadores de corrupción<br/>(ver tabla)"]
    QT["query_text_for<br/>vista canónica si el parser tiene éxito,<br/>si no, texto preprocesado<br/>y vista cruda preprocesada"]
    ENC["encode_batch<br/>q_tokens, q_tokens_raw, d_tokens<br/>(N, 48) uint8, luego truncado a 40"]
    NPZ["train_data.npz<br/>991 161 consultas"]
    LOOP["Bucle de entrenamiento (train.py)<br/>lote 2048, vista cruda con p = 0.25,<br/>1 negativo duro por ancla"]
    LOSS["Pérdida = InfoNCE (suavizado 0.02)<br/>+ 0.5 SmoothL1 (coordenadas)<br/>+ 0.2 CE (comuna) + 0.2 CE (barrio)"]
    OPT["AdamW lr 3e-3, wd 0.01, betas (0.9, 0.98)<br/>OneCycleLR, bf16, recorte de gradiente 1.0"]
    CKPT["model.pt<br/>train_report.json<br/>catastro_emb.pt"]

    CAT --> BD --> DOCS
    BD --> OWN
    DOCS --> NEG
    DOCS --> AUG
    OPS --> AUG
    AUG --> QT --> ENC --> NPZ
    DOCS --> SPLIT --> NPZ
    NPZ --> LOOP
    NEG --> LOOP
    LOOP --> LOSS --> OPT --> CKPT
```

Los 18 operadores (`augment.OPERATIONS`), con la probabilidad base de cada uno en `corrupt` (se multiplican por la intensidad: 0.45 en la primera variante y 1.0 en las demás):

| Operador | Efecto | Probabilidad base |
|----------|--------|-------------------|
| `expand_via_type` | `KR` a `CARRERA`, `Cra.`, `K`, etc. | 0.75 |
| `expand_cross_type` | Tipo de la vía de cruce expandido | 0.4 |
| `hash_variant` | `#` reemplazado por `No.`, `Nro`, `N°`, vacío, etc. | 0.65 |
| `plate_separator` | Separador de placa variado (`-`, espacio, `_`, etc.) | 0.6 |
| `glue_letters` | `12 A` a `12A` | 0.35 |
| `glue_all` | Pegado de `#` y de tipo con número (dos pasos independientes) | 0.25 y 0.12 |
| `lowercase` | Minúsculas (30 %) o formato título (12 % adicional) | 0.30 y 0.12 (escalado por intensidad) |
| `city_suffix` | `, Cali, Valle del Cauca` y variantes | 0.18 |
| `barrio_suffix` | Sufijo de barrio (lista fija de barrios reales de Cali) | 0.10 |
| `add_complement` | Añade una unidad (`APTO 301`, `TORRE 2`, etc.) | 0.22 sin complemento, 0.20 si ya hay uno |
| `drop_complement` | Elimina el complemento existente | 0.55 |
| `typo` | Intercambio, borrado, duplicado o sustitución de caracteres | 0.22 |
| `ordinal` | `1` a `1ra`, `7` a `7ma` (solo si el número de vía tiene un dígito) | 0.06 |
| `quadrant_abbrev` | `NORTE` a `Nte.`, `N`, etc. | 0.55 |
| `unpad_plate` | Quita el cero a la izquierda de la placa | 0.30 |
| `drop_plate` | Elimina la placa | 0.05 |
| `whitespace_jitter` | Duplica un espacio | 0.10 |
| `duplicate_address` | Repite la dirección completa | 0.04 |

Aspectos que conviene conocer:

- Los comentarios del código indican que los operadores se eligieron tras revisar cinco libros de evaluación reales. Esos libros no forman parte de este repositorio. El generador sintético incorpora, por tanto, conocimiento del tipo de ruido observado en datos de Cali.
- `CITY_SUFFIXES`, `BARRIO_SUFFIXES` y `COMPLEMENTS` contienen nombres de barrios y ciudad específicos de Cali (`augment.py:32-39`).
- La partición de validación es por documento: ninguna dirección catastral de validación aparece como positivo en entrenamiento. La validación sintética mide qué tan bien se recupera el documento correcto bajo el **mismo** generador de ruido con que se entrenó; no equivale a texto real (ver [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md)).
- Los negativos duros espaciales (`build_spatial_negatives`) llenan los dos primeros de ocho espacios con direcciones de la misma manzana cuando existen; el resto proviene de los centroides más cercanos. El bucle toma uno de los ocho al azar para cada ancla (`n_hard_neg = 1`).

## 4. d) Flujo de inferencia de extremo a extremo

El flujo siguiente corresponde a `service.normalize_strict` (ruta de la CLI, de la API y de la canalización de archivos). Los números son los valores por defecto verificados en el código y en `tuning.json`.

```mermaid
flowchart TB
    A["Entrada: texto de dirección"] --> B{"is_unspecified?<br/>vacío, marcador o sin letras<br/>y menos de 3 dígitos"}
    B -- "sí" --> S0["SIN_MATCH<br/>motivo: dato vacio o marcador sin informacion"]
    B -- "no" --> C["Gazetteer.detect (opcional)<br/>barrio, vereda, comuna, corregimiento<br/>cleaned_text sin el lugar"]
    C --> D["parse_address(cleaned_text)<br/>si falla, reintenta con el texto original"]
    D --> E{"parse_ok?"}
    E -- "no" --> NP["NO_PARSEABLE<br/>motivo: notas del parser"]
    E -- "sí" --> F["Dos vistas de la consulta<br/>1: forma canónica del parser<br/>2: texto preprocesado crudo<br/>ambas truncadas a 40 caracteres"]
    F --> G["Codificador (fp32, rol 0)<br/>dos embeddings de 256"]
    G --> H["Búsqueda coseno en el índice<br/>catastro_emb.pt (330 387 x 256)<br/>top 20 por vista, unión de hasta 40"]
    F --> XY["coord_head: coordenada predicha<br/>(EPSG:3116)"]
    H --> I["Componentes por candidato<br/>sim (máx. entre vistas)<br/>fuzz = token_set_ratio / 100<br/>struct = acuerdo estructural ponderado<br/>geo = exp(-d / 600 m)"]
    XY --> I
    I --> J["Puntaje fusionado<br/>0.7 sim + 0.1 fuzz + 0.1 struct + 0.1 geo"]
    J --> K{"Puerta geográfica<br/>lugar detectado?"}
    K -- "sí" --> K2["Filtra candidatos por polígono<br/>barrio (+1000 m), si queda vacío<br/>comuna (+300 m), luego corregimiento (+300 m)"]
    K -- "no" --> L
    K2 --> K3{"Quedan candidatos?"}
    K3 -- "no" --> S1["SIN_MATCH<br/>motivo: candidatos fuera de ..."]
    K3 -- "sí" --> L["Mejor candidato por puntaje fusionado"]
    L --> M{"confianza >= 0.73?"}
    M -- "no" --> S2["SIN_MATCH<br/>motivo: confianza X < umbral 0.73"]
    M -- "sí" --> N{"Reglas estrictas<br/>violación dura, o más de<br/>1 violación blanda, o<br/>struct < 0.6?"}
    N -- "sí" --> S3["SIN_MATCH<br/>motivo: regla violada<br/>(y marca sufijo unilateral si aplica)"]
    N -- "no" --> O{"Ambigüedad<br/>margen_manzana < 0.02?"}
    O -- "sí" --> S4["SIN_MATCH<br/>motivo: ambiguo: N candidatos cercanos"]
    O -- "no" --> P["OK<br/>nivel_precision, motivo con notas,<br/>punto del predio, zona oficial"]
    P --> Q["confiabilidad y confiabilidad_manzana<br/>(reliability.json, modelo híbrido)"]
```

Detalles verificados que el diagrama no muestra:

- Los componentes `sim`, `fuzz`, `struct` y `geo` se calculan en `inference.AddressNormalizer.score_components`; la fusión y las decisiones, en `service._build_row`.
- `geo` usa la distancia entre el centroide del documento (`x_m`, `y_m`) y la coordenada predicha por `coord_head`, desestandarizada con `xy_mean` y `xy_std` del checkpoint.
- Un candidato que solo falla la regla de sufijo numérico presente en un solo lado sigue contando como competidor al calcular `margen`.
- Las violaciones blandas por defecto son: letra de vía presente en un solo lado y diferencia de placa de 1 a 2 unidades. Una fila con una violación blanda permanece `OK` pero con `nivel_precision` limitado (`esquina` o `manzana`).
- Una dirección compartida por varios predios devuelve el primer predio (menor `OBJECTID`) y degrada el nivel a `manzana` o `direccion`.
- La salida `confiabilidad` no interviene en la decisión de `estado`.
- Si no hay `gazetteer.pkl` ni límites en `basemaps/`, el gazetteer se desactiva (`load_gazetteer` devuelve `None` con una advertencia): no hay detección de lugar ni puerta geográfica.

## 5. e) Diagrama de despliegue

```mermaid
flowchart LR
    subgraph CLIENTES["Clientes"]
        U1["Usuario por terminal<br/>cali-address normalize / inspect / setup"]
        U2["Navegador<br/>página de carga (/)"]
        U3["Sistemas externos<br/>HTTP JSON y multipart"]
        U4["ArcGIS Pro u otros clientes<br/>forma Esri (no es un Locator certificado)"]
    end
    subgraph LOCAL["Ejecución local"]
        CLI["cli.py + io/pipeline.py<br/>lectura por bloques, escritura atómica"]
        WEB["cali-address web<br/>uvicorn en 127.0.0.1:8000"]
    end
    subgraph CONT["Contenedor Docker (python:3.12-slim, torch CPU)"]
        API["uvicorn cali_address.api.main:app<br/>1 worker, puerto $PORT (8000 por defecto)"]
        R1["/health, /api/v1/inspect, /api/v1/normalize,<br/>/api/v1/normalize-json, /api/v1/normalize-address,<br/>/docs, /docs-integracion"]
        R2["Router Esri<br/>/arcgis/rest/services/CaliNormalizador/GeocodeServer"]
        ENGINE["Motor: AddressNormalizer + Gazetteer<br/>carga en un hilo en segundo plano"]
        ART["/app/artifacts: 7 archivos de servicio<br/>/app/basemaps: 2 GeoJSON"]
    end
    subgraph PLAT["Plataforma Railway"]
        RW["Construcción con Dockerfile<br/>healthcheck /health, timeout 300 s<br/>1 réplica, reinicio ON_FAILURE x3"]
    end

    U1 --> CLI
    U2 --> WEB
    WEB --> ENGINE
    CLI --> ENGINE
    U3 --> API
    U4 --> R2
    API --- R1
    API --- R2
    R1 --> ENGINE
    R2 --> ENGINE
    ENGINE --> ART
    RW --> CONT
```

Puntos de diseño del despliegue:

- **Un solo worker.** Cada proceso trabajador mantiene su propia copia del modelo y de los 330 387 embeddings; un segundo trabajador duplicaría aproximadamente la memoria residente.
- **Carga diferida en segundo plano.** El puerto se abre de inmediato; `/health` responde 503 con `status: loading` hasta que el modelo y el índice están en memoria, y los endpoints de normalización esperan hasta `WARMUP_TIMEOUT_S` segundos.
- **Imagen de CPU.** `NORMALIZER_DEVICE=cpu` por defecto; torch se instala en una capa propia desde el índice de ruedas de CPU.
- **Sin persistencia de cargas.** Los bytes recibidos por `multipart/form-data` se procesan en memoria y no se escriben a disco.
- El contenedor requiere que `deploy/artifacts/` exista en el momento de construir. En este repositorio **no se distribuyen** los archivos que dependen de datos catastrales; ver [07-requisitos.md](07-requisitos.md) y [05-uso-e-implementacion.md](05-uso-e-implementacion.md).

## 6. Superficies de uso (CLI y API)

| Superficie | Punto de entrada | Descripción |
|------------|------------------|-------------|
| CLI | `cali-address` o `python -m cali_address` | Subcomandos `setup`, `normalize`, `web`, `inspect`, `formats`. Códigos de salida 0, 1, 2, 3, 4 y 130. |
| Biblioteca | `cali_address.service.normalize_strict` | Misma ruta que CLI y API. Devuelve un `DataFrame` con las columnas de `OUTPUT_COLUMNS`. |
| Biblioteca | `cali_address.inference.AddressNormalizer.normalize_batch` | Ruta antigua, siempre devuelve el mejor candidato aunque la entrada sea basura. Se conserva para scripts de ablación y evaluación; no es la ruta de producción. |
| API HTTP | `GET /health` | Estado del servicio (200 o 503). |
| API HTTP | `POST /api/v1/inspect` | Lee un archivo sin normalizar; devuelve formato, columnas, columna sugerida y vista previa. |
| API HTTP | `POST /api/v1/normalize` | Normaliza un archivo (`multipart/form-data`); salida `json`, `xlsx`, `csv` o `geojson`. |
| API HTTP | `POST /api/v1/normalize-json` | Normaliza registros en memoria (`records` y `address_field`) o una lista (`addresses`). |
| API HTTP | `POST /api/v1/normalize-address` | Alias mínimo que solo acepta `addresses`. |
| API HTTP | `GET /docs`, `GET /docs-integracion` | Swagger UI generada por FastAPI y la guía de integración renderizada desde `docs/API.md`. |
| Forma Esri | `GET` y `POST .../GeocodeServer/findAddressCandidates`, `POST .../geocodeAddresses` | Reproduce solo la forma JSON de ArcGIS REST. No es un Locator registrado. |

Diferencia relevante entre superficies: la CLI y la canalización de archivos pueden devolver, además de `OK`, `SIN_MATCH` y `NO_PARSEABLE`, los estados `FUERA_DE_AREA` (con `--municipality-col`) y `ERROR`; la API HTTP expone solo los tres primeros.

Ejemplos de uso en [05-uso-e-implementacion.md](05-uso-e-implementacion.md); interpretación de cada columna en [06-predicciones.md](06-predicciones.md).
