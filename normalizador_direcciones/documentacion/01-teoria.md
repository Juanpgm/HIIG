# 01. Fundamentos teóricos

Este documento explica por qué el sistema está construido como está: qué problema resuelve, qué técnica de aprendizaje usa y dónde la técnica se apoya en reglas deterministas. Está escrito para quien necesita juzgar si el enfoque es adecuado antes de adoptarlo. Los detalles de implementación están en [02-arquitectura.md](02-arquitectura.md); la evidencia empírica, en [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md).

## Contenido

1. [El problema](#1-el-problema)
2. [Recuperación con codificador dual](#2-recuperación-con-codificador-dual)
3. [Aprendizaje contrastivo (InfoNCE)](#3-aprendizaje-contrastivo-infonce)
4. [Por qué a nivel de carácter](#4-por-qué-a-nivel-de-carácter)
5. [Corrupción sintética](#5-corrupción-sintética)
6. [Negativos duros espaciales](#6-negativos-duros-espaciales)
7. [Cabezas multitarea](#7-cabezas-multitarea)
8. [Puntaje de fusión y decaimiento geográfico](#8-puntaje-de-fusión-y-decaimiento-geográfico)
9. [Capa de reglas estrictas](#9-capa-de-reglas-estrictas)
10. [Confiabilidad calibrada: `confianza` frente a `confiabilidad`](#10-confiabilidad-calibrada-confianza-frente-a-confiabilidad)
11. [Qué garantiza y qué no garantiza el enfoque](#11-qué-garantiza-y-qué-no-garantiza-el-enfoque)

## 1. El problema

Una dirección urbana colombiana es un identificador espacial fácil de leer para una persona y frágil para un sistema: se escribe como `Cl. 5 # 34-12`, `Calle 5ta No 34 12` o `Cll 5 34 12`. Para cruzar registros administrativos con el catastro se necesita convertir ese texto en una referencia estable (el Número Predial Nacional, NPN, y la manzana).

El sistema resuelve el problema como **recuperación de información**, no como clasificación: en lugar de predecir una de 330 387 clases, mide la similitud entre el texto de entrada y cada dirección del catastro y elige la mejor. La ventaja es que el catálogo puede cambiar (otra ciudad, otra fecha de actualización) sin reentrenar un clasificador con un número fijo de clases; basta recalcular los embeddings del catálogo.

El sistema combina tres componentes:

1. Un **analizador gramatical** basado en reglas que reconoce la estructura `vía # cruce - placa [complemento]`.
2. Un **codificador neuronal** que mapea cadenas a vectores, para tolerar errores de escritura.
3. Una **capa de decisión** con reglas estrictas y umbrales que decide cuándo abstenerse.

## 2. Recuperación con codificador dual

En un codificador dual (también llamado bi-encoder o torre siamesa), una consulta `q` y un documento `d` se codifican de forma independiente con la misma red y se comparan con una función de similitud simple, aquí el producto interno de vectores unitarios (similitud coseno):

```
sim(q, d) = <f(q, rol=0), f(d, rol=1)>      con ||f(.)|| = 1
```

Propiedades relevantes:

- **Los documentos se codifican una sola vez.** Los 330 387 vectores de 256 dimensiones se guardan en `catastro_emb.pt` (float16, unos 169 MB). Consultar es codificar la consulta y hacer una multiplicación matricial seguida de un `top-k`.
- **Escala bien.** La búsqueda exacta sobre 330 387 vectores es una multiplicación de `(2, 256)` por `(256, 330 387)` por consulta; no requiere un índice aproximado.
- **Pesos compartidos con rol.** Un solo codificador procesa ambos lados; el embedding de rol le indica si lee una cadena ruidosa (rol 0) o una dirección catastral canónica (rol 1). Esto permite que el espacio sea común sin duplicar parámetros.
- **Limitación.** Al no haber interacción entre consulta y documento durante la codificación, el modelo es menos preciso en diferencias finas (una letra, un dígito de la placa) que un modelo de atención cruzada. Esa carencia se compensa con la fusión de puntajes y con la capa de reglas (secciones 8 y 9).

## 3. Aprendizaje contrastivo (InfoNCE)

El entrenamiento usa una pérdida contrastiva de tipo InfoNCE. Para un lote de consultas `q_i` con su documento positivo `d_i`, y un conjunto de documentos del lote `D` (los positivos más los negativos duros, sin repetidos), se calcula:

```
logits_ij = escala * sim(q_i, d_j)             escala = exp(logit_scale), aprendible, limitada a 100
L_nce     = promedio_i  CE( softmax_j(logits_ij), objetivo = indice de d_i en D )
```

con suavizado de etiquetas de 0.02 (`label_smoothing`). La temperatura es aprendible: `logit_scale` se inicializa en `ln(20)`, es decir, una temperatura inicial de 0.05.

Los documentos del mismo lote que no son el positivo de una consulta actúan como negativos gratuitos (negativos "en el lote"). Con lote de 2048 y un negativo duro por ancla, cada consulta compite contra hasta 4096 documentos. Esto empuja a la red a separar direcciones distintas y a acercar las variantes ruidosas de una misma dirección.

## 4. Por qué a nivel de carácter

Las direcciones con ruido tienen propiedades que perjudican a los modelos basados en palabras o subpalabras:

| Propiedad | Ejemplo | Efecto en un vocabulario de palabras |
|-----------|---------|--------------------------------------|
| Tokens pegados | `Calle12a#57-04` | Palabra desconocida |
| Abreviaturas y variantes ortográficas | `KR`, `Cra`, `CRA.`, `Krr` | Muchos tokens sin relación aparente |
| Errores tipográficos | `Carrea 26` | Palabra desconocida |
| Identificadores numéricos con letras | `31 F 3`, `73 BIS` | Combinatoria enorme, un dígito cambia la identidad |
| Texto corto | 18 caracteres de mediana, 31 en el percentil 99 | No hay contexto largo que aproveche subpalabras |

Un modelo a nivel de carácter con un vocabulario de 48 símbolos (letras mayúsculas A-Z, dígitos y los signos `# - . , / ' ( ) |` y espacio, más relleno y desconocido) no tiene tokens fuera de vocabulario de letras y dígitos ASCII, y aprende por sí mismo qué variaciones son equivalentes. Costos y límites:

- La entrada se convierte a mayúsculas y ASCII (los acentos se eliminan) antes de codificar; cualquier símbolo fuera del vocabulario se mapea a `UNK`.
- La longitud máxima del checkpoint es de **40 caracteres**. En el catastro de Cali, unas pocas direcciones indexadas superan 40 caracteres; se truncan al embeber, y una consulta larga se trunca igual. Un complemento largo puede perderse en la vista neuronal (la vista de reglas conserva el texto completo hasta la etapa de decisión).
- Con secuencias de 40 posiciones, la atención completa es barata (`40 x 40` por cabeza).

## 5. Corrupción sintética

No existe un corpus público de pares (texto escrito por personas, dirección catastral correcta) con tamaño suficiente y sin datos personales. El sistema genera sus propios pares: parte de cada dirección catastral canónica y aplica operadores de corrupción que imitan los errores observados (los 18 operadores están en [02-arquitectura.md](02-arquitectura.md), sección 3).

Se generan tres variantes por dirección: la primera con intensidad 0.45 (ruido suave) y las otras dos con intensidad 1.0. El resultado son 991 161 consultas de entrenamiento y validación sobre 330 387 documentos.

Consecuencias que el lector debe tener presentes:

- **El modelo conoce el generador, no el mundo.** La validación sintética (recall@k sobre consultas sintéticas) mide la recuperación bajo el mismo proceso de ruido con el que se entrenó. Es una condición necesaria, no una garantía sobre texto real.
- **Brecha entre sintético y real.** En evaluación con texto real (ver [08](08-metricas-y-confiabilidad.md)) las precisiones son bastante menores que las sintéticas.
- **El generador es específico de Cali** en sus listas de sufijos de ciudad, barrios y convenciones de cuadrante. Otro territorio requiere revisar los operadores.

## 6. Negativos duros espaciales

El error típico de este problema no es confundir `CL 9 # 41 - 26` con una calle remota, sino con su vecina: `KR 31 F 2 # 84 - 15` frente a `KR 31 F # 84 - 15`, o la placa `15` frente a la `17` en la misma cuadra. Los negativos en el lote casi nunca son tan parecidos, así que se añaden **negativos duros**: para cada documento se precalculan ocho documentos confundibles (`build_spatial_negatives`):

1. Dos espacios con direcciones de la **misma manzana catastral**, cuando existen.
2. El resto, de los **centroides más cercanos** (búsqueda kd-tree sobre coordenadas métricas).

En cada paso se toma uno al azar por ancla (`n_hard_neg = 1`). La infraestructura de experimentos incluye además negativos de placa (misma cara de manzana con placa a 1-20 unidades) y mezcla con pares reales; esas variantes no superaron el ruido entre semillas y **no se usan** en el modelo publicado (ver `docs/experiments.md` y la tabla de experimentos en [08](08-metricas-y-confiabilidad.md)).

## 7. Cabezas multitarea

Además de la pérdida contrastiva, el codificador aprende tres tareas auxiliares a partir del mismo vector agrupado:

| Cabeza | Objetivo | Pérdida | Peso | Fuente de la etiqueta |
|--------|----------|---------|-----:|-----------------------|
| `coord_head` | Coordenada del centroide del predio (EPSG:3116, estandarizada) | SmoothL1 (`beta` 0.2) | 0.5 | Centroide de la dirección catastral |
| `comuna_head` | Comuna (24 clases) | Entropía cruzada | 0.2 | Dígitos 10-11 del NPN |
| `barrio_head` | Barrio (342 clases) | Entropía cruzada | 0.2 | Dígitos 10-13 del NPN |

Pérdida total:

```
L = L_nce + 0.5 * L_coord + 0.2 * L_comuna + 0.2 * L_barrio
```

La idea es inyectar en el embedding un **prior espacial y jerárquico**: direcciones cercanas o del mismo barrio deben tener representaciones compatibles. Después del entrenamiento:

- `coord_head` **sí se usa** en inferencia: su predicción alimenta el término geográfico del puntaje.
- `comuna_head` y `barrio_head` **no se usan** en inferencia; son solo señal de entrenamiento. Sus dimensiones (24 y 342 clases) son específicas de Cali, pero eso no afecta a la inferencia.

El código de comuna y barrio se obtiene del NPN por posición. En Cali, los dígitos 8-9 del NPN (el "sector" del estándar nacional) valen `00` en todo el índice, y la comuna aparece en los dos primeros dígitos del campo de barrio (posiciones 10-11). Es una convención local; ver [10-estandar-y-generalizacion.md](10-estandar-y-generalizacion.md).

## 8. Puntaje de fusión y decaimiento geográfico

La similitud neuronal sola no basta para decidir. Cada candidato (hasta 40 por consulta: 20 por vista) recibe cuatro componentes en `[0, 1]`:

| Componente | Definición | Peso |
|------------|------------|-----:|
| `sim` | Similitud coseno del codificador; el máximo entre las dos vistas de la consulta (forma canónica del parser y texto crudo preprocesado) | 0.7 |
| `fuzz` | `rapidfuzz.token_set_ratio` entre la forma canónica sin complemento y la dirección catastral, dividido por 100 | 0.1 |
| `struct` | Acuerdo estructural ponderado entre los campos analizados de ambos lados (tipo de vía, números, letras, cuadrante, BIS, placa). Pesos: número de vía 2.0, número de cruce 2.0, placa 1.5, tipo de vía 1.0, letras 1.0, cuadrantes 0.75, BIS 0.75, tipo de cruce 0.5; la placa a 1-2 unidades da crédito parcial (0.4) | 0.1 |
| `geo` | `exp(-d / 600 m)`, con `d` la distancia entre el centroide del candidato y la coordenada predicha por `coord_head` | 0.1 |

```
confianza = 0.7 * sim + 0.1 * fuzz + 0.1 * struct + 0.1 * geo
```

Los pesos y el umbral de 0.73 se eligieron en `scripts/tune_threshold.py` con documentos catastrales retenidos y cadenas basura sintéticas (ver [04-entrenamiento.md](04-entrenamiento.md)); `tuning.json` indica `evaluation_rows_used: none` (no se usaron filas de evaluación real para esos dos elementos). La escala `tau = 600 m` del decaimiento geográfico está fijada a priori (`GEO_TAU_M`), no se ajustó.

El término geográfico actúa como un amortiguador: un candidato que coincide textualmente pero queda a kilómetros de donde el modelo "espera" la dirección pierde hasta 0.1 de puntaje. Se impuso un peso mínimo de 0.1 para `geo` antes del ajuste, porque con solo datos de validación sintética el óptimo sin restricción lo reducía a 0.05 y anulaba `fuzz` (la recuperación neuronal ya es del 94 % en esa distribución, así que no hay nada que corregir). Esa restricción es una decisión de diseño documentada en `tuning.json` y en `scripts/tune_threshold.py`; su costo medido en top-1 es de 0.0018 (0.83 errores estándar).

## 9. Capa de reglas estrictas

El puntaje fusionado decide si hay un candidato "suficientemente parecido". Las reglas deciden si el candidato es **admisible**. Un candidato que supera el umbral pero viola una regla de identidad no se entrega: la fila queda como `SIN_MATCH` con el motivo explícito, sin ningún campo catastral.

| Regla | Naturaleza | Comportamiento por defecto |
|-------|------------|-----------------------------|
| Tipo de vía, número de vía, número de cruce, tipo de cruce | Dura | Rechazo si ambos lados traen valor y difieren |
| Placa | Dura o blanda | Distinta: rechazo; con diferencia de 1 a 2 unidades es blanda (la fila puede salir `OK` con nivel limitado a `manzana`) |
| Sufijo numérico de vía o cruce (`KR 31 F` frente a `KR 31 F 3`) | Dura | Rechazo, también si está presente en un solo lado (se marca con `regla: sufijo unilateral`) |
| Letra de vía | Dura si difiere en ambos lados; blanda si está en un solo lado | Fila `OK` con nivel limitado a `manzana` |
| Letra de cruce, BIS | Dura | Rechazo |
| Cuadrante | Dura si hay desacuerdo; presente en un solo lado se acepta con nivel limitado a `manzana` | - |
| Coincidencia estructural mínima | Dura | `struct >= 0.6` |
| Puerta geográfica | Dura | Si se detecta un lugar en el texto, el candidato debe caer dentro de su polígono más una holgura |
| Ambigüedad | Abstención | Si hay un competidor que cumple las reglas en **otra manzana** con diferencia de puntaje menor que 0.02, la fila pasa a `SIN_MATCH` |

Por qué reglas y no solo aprendizaje: una letra, un dígito o un sufijo no son "ruido" cuando cambian la identidad de la calle. Dejar que el modelo los trate como similitud fuzzy produciría coincidencias plausibles pero incorrectas. La contrapartida está medida: la regla de sufijo unilateral rechazó filas que eran el lote correcto (24 de 79 en el conjunto de referencia del registro de cambios; ver [08](08-metricas-y-confiabilidad.md)). Es un compromiso deliberado entre precisión y cobertura.

## 10. Confiabilidad calibrada: `confianza` frente a `confiabilidad`

Son dos magnitudes distintas y no deben confundirse.

| Campo | Qué es | Cómo se obtiene | Qué significa |
|-------|--------|-----------------|---------------|
| `confianza` | Puntaje de fusión (0-1) | Fórmula de la sección 8 | Qué tan parecido es el candidato a la consulta. **No es una probabilidad.** |
| `confiabilidad` | Probabilidad estimada de que el NPN entregado sea el correcto | `reliability.json` (modelo híbrido) a partir de `confianza` y del tipo de fila | Estimación calibrada, con las salvedades de la sección 11 |
| `confiabilidad_manzana` | Probabilidad estimada de que la manzana sea la correcta | Idem | Siempre mayor o igual que `confiabilidad` |

El modelo `hybrid` de `reliability.json`:

1. `P(manzana | confianza)`: regresión isotónica (función escalonada monótona, interpolada linealmente entre puntos y acotada a `[0.02, 0.98]`).
2. `P(predio) = P(manzana) * q_tipo`, donde `q_tipo` es la tasa suavizada (Bayes empírico) de acertar el predio dado que la manzana es correcta, por tipo de fila (`exacto`, `relajada`, `compartida`, `parcial`).

Ejemplo de por qué `confianza` no es una probabilidad: una banda alta de `confianza` (por ejemplo 0.9-1.0) no acierta el predio exacto en todos los casos; mida el acierto por banda con su propia muestra (ver [08](08-metricas-y-confiabilidad.md)).

## 11. Qué garantiza y qué no garantiza el enfoque

Lo que el diseño garantiza:

- **Abstención explícita.** Una fila sin coincidencia aceptable no recibe un candidato "de relleno"; el campo `motivo` explica el rechazo.
- **Degradación honesta del nivel.** Una coincidencia con alguna aproximación no se etiqueta como `predio`.
- **Determinismo en la ruta estricta.** El codificador y la cabeza de coordenadas corren en fp32 y la similitud se recalcula en fp32; el resultado no depende del tamaño del lote.

Lo que el diseño no garantiza:

- **Corrección del predio.** Se observó que `confiabilidad` discrimina muy poco entre filas `OK`: el calibrador servido (v4) se ajustó con etiquetas GPS ruidosas; un calibrador alternativo (v5, `reliability_v2.json`, no servido por defecto) ordena bien sobre etiquetas consistentes con el texto. Ver [08](08-metricas-y-confiabilidad.md) y [09](09-advertencias-y-limitaciones.md).
- **Generalización fuera de Cali.** El índice, el generador de ruido, el gazetteer, los códigos de comuna y barrio y el umbral están ajustados a un solo municipio ([10](10-estandar-y-generalizacion.md)).
- **Uso sin supervisión humana en decisiones jurídicas o catastrales.** La herramienta apoya la normalización y el cruce de datos; no sustituye la verificación.
