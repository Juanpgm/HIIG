# 10. Estándar nacional y generalización

Este documento contrasta lo que el sistema produce con el estándar colombiano de direcciones y de identificación predial, y da un procedimiento para adaptarlo a otras ciudades colombianas o a otros países que usen un estándar similar, con un protocolo de validación antes de confiar en los resultados de un territorio nuevo.

La referencia normativa es el documento de síntesis del proyecto sobre lineamientos técnicos y estándares para la gestión de datos catastrales en Colombia (`docs_standard.md`, que cita circulares y resoluciones de la autoridad catastral, de la autoridad estadística y del modelo LADM_COL). Es una **síntesis secundaria**: antes de adoptar una exigencia como obligatoria hay que contrastarla con los instrumentos primarios vigentes.

## Contenido

1. [Conformidad con el estándar](#1-conformidad-con-el-estándar)
2. [Pruebas realizadas con el analizador](#2-pruebas-realizadas-con-el-analizador)
3. [Lista de verificación para adaptar a otro territorio](#3-lista-de-verificación-para-adaptar-a-otro-territorio)
4. [Protocolo de validación antes de confiar en un territorio nuevo](#4-protocolo-de-validación-antes-de-confiar-en-un-territorio-nuevo)
5. [¿Es reutilizable `model.pt` fuera de Cali?](#5-es-reutilizable-modelpt-fuera-de-cali)

## 1. Conformidad con el estándar

Estado: **cumple**, **parcial** o **no cumple**. "Producto" es lo que realmente emiten el analizador (`parser.canonical`) y la salida del normalizador; la columna "Acción" es una recomendación.

| # | Requisito del estándar | Qué produce el sistema (verificado) | Estado | Acción |
|--:|------------------------|-------------------------------------|:------:|--------|
| 1 | Orden de componentes: tipo de vía, número de vía, prefijo o cuadrante, número de vía generadora, prefijo o cuadrante, placa | Orden `TIPO NUM [LETRAS] [SUFIJO] [BIS] [CUADRANTE] # CRUCE ... - PLACA [COMPLEMENTO]`; ej. `CL 11 SUR # 23 A BIS - 50` | Cumple (el orden es el exigido) | Ninguna |
| 2 | Letras mayúsculas sostenidas | La forma por reglas siempre sale en mayúsculas. En filas `OK` se devuelve la dirección catastral tal cual: algunas direcciones del índice de Cali contienen minúsculas | Parcial | Aplicar `upper()` a `direccion_normalizada` o limpiar el catastro |
| 3 | Sin tildes | El analizador elimina acentos. El texto catastral de 172 direcciones contiene `Ñ` y se devuelve sin cambios | Parcial | Convertir a ASCII la salida de origen catastral |
| 4 | Sin signos de puntuación | La salida usa `#` y ` - ` como separadores (`KR 12 # 34 - 56`), y guion en complementos de varias partes (`AP 3 - 02`). El ejemplo del propio estándar los elimina (`CL 11 SUR 23 A BIS 50`) | **No cumple** (por diseño: replica el formato del catastro de Cali) | Añadir un formateador final, o una opción de estilo, que elimine `#` y `-` |
| 5 | Sin el símbolo `#` ni `No.`, `N°` | Las variantes de entrada `No`, `Nro`, `N°`, `Num` se reconocen y se unifican, pero la salida emite `#` | **No cumple** en la salida; cumple en la entrada | Igual que la fila 4 |
| 6 | Sin ordinales (`7ma`, `1ra`) | Se descartan en la entrada: `Cra 7ma # 45-10` produce `KR 7 # 45 - 10` | Cumple | Ninguna |
| 7 | `AU` Autopista | `Autopista 3 # 10-20` produce `AU 3 # 10 - 20` (con número; `Autopista` sin número no se analiza) | Cumple | Ninguna |
| 8 | `AV` Avenida | Cumple (`Avenida 6 Norte` produce `AV 6 NORTE`) | Cumple | Ninguna |
| 9 | `AC` Avenida Calle | `Avenida Calle 9` produce `AC 9` | Cumple | Ninguna |
| 10 | `BL` Bulevar | `Boulevard 5 # 12-30` no se analiza (`NO_PARSEABLE`). `BL` está registrado como alias de **bloque** (`BLQ`) en complementos, no como tipo de vía | **No cumple** (y el alias entra en conflicto) | Registrar `BULEVAR` y `BL` como tipo de vía y renombrar el alias de bloque |
| 11 | `CL` Calle | Cumple | Cumple | Ninguna |
| 12 | `DG` Diagonal | Cumple | Cumple | Ninguna |
| 13 | `TV` Transversal | Cumple (`Transv`, `Trv`, `T` suelta aceptados en entrada) | Cumple | Ninguna |
| 14 | `CO` Conjunto Residencial | Se normaliza a `CONJ` (`Conjunto Los Pinos` produce `CONJ LOS PINOS`), no a `CO` | **No cumple** | Cambiar el código canónico a `CO` o añadir una tabla de salida |
| 15 | `KR` Carrera (y no `Cra`, `Crr`, `Krr`) | Cumple (`Carrera`, `Cra`, `CR`, `CRR`, `KRA`, `K` producen `KR`) | Cumple | Ninguna |
| 16 | Tipos de vía adicionales | El sistema añade `AK` (avenida carrera), `PJ`, `PS`, `CV`, `CT` | No requerido | Contrastar con el diccionario oficial de abreviaturas |
| 17 | En la salida de origen catastral, códigos de vía canónicos | Una parte de las direcciones del índice empieza por una letra antigua (`K`, `C`, `A`, `D`, `T`); se devuelven sin normalizar en filas `OK`. Las filas por reglas sí usan `KR`, `CL`, etc. | Parcial | Reaplicar `canonical()` al texto catastral antes de devolverlo |
| 18 | NPN de 30 posiciones numéricas | Casi todos los documentos tienen 30 caracteres; unos pocos tienen 29 o 28 y alguno es nulo. Las anomalías se conservan y se marcan (`NPN anomalo en catastro`); un NPN nulo baja el nivel a `direccion` | Cumple en general; anomalías marcadas | Corregir en origen |
| 19 | Códigos DIVIPOLA en el NPN (departamento 2, municipio 3) | El índice de Cali contiene `76` y `001` salvo un registro. El sistema no extrae ni valida departamento y municipio; solo la puerta municipal de la CLI reconoce `76001`, `76001000` y el postal `7600xx` | Parcial | Validar `NPN[0:5]` frente al municipio objetivo al construir el índice |
| 20 | Estructura del NPN (sector 8-9, barrio 10-13, manzana 14-17, terreno 18-21, condición 22, torre 23-26, unidad 27-30) | `comuna = NPN[9:11]`, `barrio_code = NPN[9:13]`, `manzana` = código de 17 posiciones (NPN[:17]). En Cali el sector (8-9) es `00` en todo el índice: la comuna se lee en las posiciones 10-11 | Parcial: convención local, no la del estándar | Ver sección 3, paso 2 |
| 21 | Zona rural o urbana (posiciones 6-7: `01` urbana) | El índice tiene `01` en 330 368 documentos y `00` en 18; no se usa la zona para decidir | Parcial | Filtrar por zona al construir el índice |
| 22 | Modelo LADM_COL (`CR_Predio`, unidad administrativa básica) | Solo se devuelve el NPN y la manzana. No hay exportación ni correspondencia con clases de LADM_COL (terreno, unidad de construcción, derechos, interesados) | **No cumple** (fuera de alcance) | Si se necesita, añadir un adaptador que mapee las columnas de salida a `CR_Predio` |
| 23 | Marco Geoestadístico Nacional (MGN): código de manzana censal de 22 posiciones | El sistema usa la manzana **catastral** (17 posiciones) y polígonos de la IDE municipal; no usa ni cruza con códigos del MGN | **No cumple** (fuera de alcance) | Añadir una tabla de correspondencia manzana catastral a manzana censal si se requiere |
| 24 | Seis componentes mínimos de la dirección | Se reconocen y validan los seis. Una dirección sin placa baja a `esquina`; sin cruce, a `via` | Cumple | Ninguna |
| 25 | Entrada escrita en la forma del propio estándar (sin `#` ni `-`) | `CL 11 SUR 23 A BIS 50` se acepta en la entrada y `parser.canonical` la emite como `CL 11 SUR # 23 A BIS - 50` (verificado). A diferencia del emparejador, que no lee esa forma ([10 del emparejador](../../emparejador_direcciones/documentacion/10-estandar-y-generalizacion.md)) | Cumple en la entrada | Ninguna |

## 2. Pruebas realizadas con el analizador

Se ejecutó `parser.parse_address` y `parser.canonical` sobre direcciones **inventadas**, en modo de solo lectura (sin escribir archivos). Resultados representativos:

| Entrada inventada | Resultado de `canonical` |
|-------------------|--------------------------|
| `Calle 11 Sur Número 23 A BIS – 50` (ejemplo del estándar) | `CL 11 SUR # 23 A BIS - 50` (el estándar espera `CL 11 SUR 23 A BIS 50`) |
| `Cl. 5 # 34-12` | `CL 5 # 34 - 12` |
| `Calle 5ta No 34 12` | `CL 5 # 34 - 12` |
| `Cll 5 34 12` | `CL 5 # 34 - 12` |
| `Cra 7ma # 45-10 apto 301` | `KR 7 # 45 - 10 AP 301` |
| `Avenida Calle 9 # 12-30` | `AC 9 # 12 - 30` |
| `Diagonal 23 # 40-15` | `DG 23 # 40 - 15` |
| `Transv 3 # 4-5` | `TV 3 # 4 - 05` (la placa se rellena a dos dígitos, convención del catastro de Cali) |
| `Autopista 3 # 10-20` | `AU 3 # 10 - 20` |
| `Boulevard 5 # 12-30` | no analizable |
| `Calle 9 # 41-26 Conjunto Los Pinos` | `CL 9 # 41 - 26 CONJ LOS PINOS` |
| `Carrera 100 bis # 15-30` | `KR 100 BIS # 15 - 30` |

Observaciones:

- Quitar `#` y ` - ` de la forma canónica reproduce el ejemplo del estándar (`CL 11 SUR 23 A BIS 50`). El formateador de la fila 4 es, por tanto, una transformación simple, no un cambio del analizador.
- El relleno de la placa a dos dígitos y el uso de `CONJ`, `BLQ` y `AK` provienen del catastro de Cali, no del estándar.
- Las pruebas de analizador no validan la corrección de la recuperación, solo la forma sintáctica.

## 3. Lista de verificación para adaptar a otro territorio

### 3.1 Otra ciudad colombiana

Los números de paso corresponden a la estructura de los demás documentos. Las ubicaciones de código están en el inventario de [09-advertencias-y-limitaciones.md](09-advertencias-y-limitaciones.md), sección 2.

1. **Catastro del municipio.** Obtener la tabla predial (con el esquema de [04-entrenamiento.md](04-entrenamiento.md), sección 2.1): dirección, NPN de 30 posiciones, manzana, centroide. Confirmar derechos de uso y redistribución.
2. **Etiquetas jerárquicas del NPN.** Verificar qué posiciones del NPN están pobladas en el municipio (conteos por posición). Elegir el nivel auxiliar con pocas decenas de clases. En Cali, el sector (8-9) es `00` y se usan las posiciones 10-11 como comuna; en otro municipio puede ser el sector. Modificar los cortes en `dataset.py:44-46`, `evaluate.py:246` y `scripts/build_splits.py:104`. Verificar `NPN[0:5]` frente al código DIVIPOLA del municipio.
3. **Gramática de direcciones.** Medir qué fracción de las direcciones del catastro local se analiza correctamente y reproduce su forma original (ida y vuelta); el cuaderno del proyecto reporta esa prueba en Cali. Si el catastro local usa otra nomenclatura (por ejemplo, vías con nombre, sin grilla), el enfoque de grilla no es adecuado.
4. **Analizador y léxico.** Ajustar en `parser.py` los tipos de vía, la convención de cuadrantes (`parser.py:88-98`), el ruido de ciudad (`parser.py:151-154`) y los complementos del léxico.
5. **Generador de ruido.** Sustituir en `augment.py` los sufijos de ciudad y de barrio y los complementos (`augment.py:32-39`). Si se dispone de cadenas reales sin etiquetar, usar `--realistic-noise` para recalibrar las probabilidades de los operadores.
6. **Límites geográficos.** Preparar los dos GeoJSON con el esquema de [07-requisitos.md](07-requisitos.md), sección 3.3, o adaptar `_read_basemaps`. Ajustar `MIN_COMUNA`, `MAX_COMUNA` (`gazetteer.py:129`), los nombres de archivo y el ruido de ciudad (`gazetteer.py:139-142`). Sin ellos, desactivar el gazetteer (`--no-gazetteer`) y asumir que se pierde la puerta geográfica.
7. **Proyección métrica.** Elegir un CRS proyectado adecuado al territorio (`geo.py:17`). EPSG:3116 es la zona Bogotá de MAGNA-SIRGAS; para un uso nacional conviene revisar el sistema proyectado oficial vigente del país. Actualizar la caja de saneamiento (`eval_data.py:168-171`).
8. **Puerta municipal.** Sustituir `_DANE_CALI`, los conjuntos de tokens y el postal en `io/pipeline.py:118-159` por el código DIVIPOLA y los nombres del municipio objetivo.
9. **Datos de entrenamiento y entrenamiento.** Ejecutar la receta de [04](04-entrenamiento.md). No reutilizar `model.pt` (sección 5).
10. **Pesos y umbral.** Ejecutar `tune_threshold.py` con el catastro local. No copiar el 0.73 de Cali.
11. **Constantes de la capa de reglas.** Revisar la holgura de barrio (1000 m), la de zona (300 m), la escala geográfica (600 m), `min_struct` y `ambiguity_delta` con datos locales (esas constantes se fijaron con datos de Cali).
12. **Referencias para validar.** Construir un conjunto de filas reales con referencia (sección 4).
13. **Confiabilidad.** Ajustarla con datos locales y verificar su AUC, o dejarla desactivada.
14. **Nombres y textos.** Cambiar el título de la API, el nombre del servicio Esri (`api/esri.py:48`, `api/main.py:446`) y los mensajes.
15. **Pruebas.** Reconstruir las pruebas de regresión del analizador con direcciones inventadas del territorio.

### 3.2 Otro país con un estándar de direcciones propio

Los elementos reutilizables son la arquitectura (codificador dual a nivel de carácter), el método de entrenamiento (corrupción sintética, negativos duros, fusión de puntajes) y la capa de decisión (abstención, degradación honesta del nivel). Todo lo demás es local:

| Componente | Qué se reemplaza |
|------------|------------------|
| Gramática y vocabulario de vías | Tipos de vía y abreviaturas del estándar nacional del país; convención de cuadrantes o equivalentes |
| Identificador de parcela | Estructura del identificador catastral y sus cortes jerárquicos; el sentido de `comuna`, `barrio` y `manzana` |
| Códigos administrativos | Reemplazo de DIVIPOLA por el código del país; puerta municipal |
| Sistema de referencia | CRS proyectado local; caja de saneamiento |
| Vocabulario del codificador | Si el alfabeto del país no es ASCII (tildes, otras escrituras), ampliar `CHARS` y reentrenar |
| Longitud máxima | Revisar el percentil 99 de la longitud de las direcciones catastrales |
| Idioma | Mensajes de motivo, nombres de columnas y documentación |
| Datos | Catastro, límites y referencias propios, con sus licencias |

## 4. Protocolo de validación antes de confiar en un territorio nuevo

Es una **recomendación** del autor de esta documentación, no un procedimiento del proyecto original.

### 4.1 Preparación

1. Construir un conjunto de referencia **independiente del texto de la dirección**, por ejemplo el predio verificado en campo o por trámite. Una referencia derivada del mismo texto (como la etiqueta "gold" de Cali) infla las cifras.
2. Separar tres conjuntos disjuntos por clave de dirección: **ajuste** (para constantes y confiabilidad), **desarrollo** y **prueba final** (se usa una sola vez).
3. Registrar la **calidad de la referencia** (distancia entre la referencia y el predio real, tasa de referencias inciertas) y publicarla.
4. Muestrear de forma **estratificada** por fuente de datos, tipo de vía, zona (comuna o localidad) y tipo de dirección (con y sin complemento, con y sin barrio).

### 4.2 Tamaños mínimos de muestra

Con una proporción esperada `p` y un error absoluto `e` (95 % de confianza), `n = 1.96² p (1 - p) / e²`:

| Proporción esperada | Error `e` | `n` mínimo |
|--------------------:|----------:|-----------:|
| 0.90 | 0.05 | 139 |
| 0.90 | 0.03 | 385 |
| 0.50 (peor caso) | 0.05 | 385 |
| 0.50 (peor caso) | 0.03 | 1 068 |

Condiciones prácticas:

- El `n` es **por estrato** si se quieren cifras por estrato, y de filas que llegan a `OK` (no de filas totales) para estimar precisión.
- Si la tasa de `OK` es de alrededor del 50 %, hacen falta aproximadamente el doble de filas totales.
- Para estimar la confiabilidad calibrada se necesitan cientos de filas `OK` con referencia por cada fuente; el modelo publicado se ajustó con 529 filas y su AUC resultó cercano a 0.5.
- Documentar el tamaño real y los intervalos de Wilson: a 289 filas, una fila equivale a 0.35 puntos.

### 4.3 Métricas a reportar

| Métrica | Definición | Nivel |
|---------|-----------|-------|
| Distribución de `estado` | Proporción de `OK`, `SIN_MATCH`, `NO_PARSEABLE` | Global y por estrato |
| Cobertura | `OK` entre filas analizables | Global y por estrato |
| Precisión de manzana y de predio | Aciertos entre filas `OK` con referencia | Global, por `nivel_precision` y por estrato; con intervalo de Wilson |
| Precisión con `nivel_precision == predio` | Idem, solo en ese nivel | Con intervalo |
| Correctas perdidas | Filas con respuesta correcta descartadas por umbral o reglas | Por motivo |
| Calibración | Brier, ECE y AUC de `confiabilidad` y de `confianza` | Con validación cruzada por fuente |
| Errores graves | Proporción de filas `OK` a más de 100 m de la referencia | Distribución |
| Efecto de la referencia | Resultados con referencia estricta y con referencia ruidosa | Ambas |

### 4.4 Puertas de aceptación (propuesta)

Cada puerta debe cumplirse con el **límite inferior del intervalo de confianza**, no con la estimación puntual. Los valores numéricos son una propuesta de partida; cada institución debe fijar los suyos según el costo de un error.

| Puerta | Condición propuesta | Si no se cumple |
|--------|---------------------|-----------------|
| G1. Calidad de la referencia | Documentada y de tamaño suficiente (4.2) | No se puede concluir |
| G2. Manzana | Límite inferior de la precisión de manzana en filas `OK` mayor o igual que 0.90 para uso estadístico | Limitar el uso a lo que la cifra respalda o reentrenar |
| G3. Predio | Límite inferior de la precisión de predio en filas `OK` con `nivel_precision == predio` mayor o igual que el objetivo del uso; para uso individual con efectos sobre personas, exigir verificación humana en todos los casos | No usar el NPN individual sin revisión |
| G4. Abstención | La fracción de `OK` con error grave (más de 100 m) está por debajo de un máximo acordado | Subir umbral o reglas |
| G5. Calibración | `confiabilidad` con AUC claramente mayor que 0.5 (con intervalo) si se piensa usarla para ordenar | No usarla para ordenar o filtrar |
| G6. Estabilidad | Resultados reproducibles al repetir la corrida con distinto tamaño de lote | Revisar la ruta de ejecución |
| G7. Cobertura por estrato | Ningún estrato relevante con cobertura o precisión sustancialmente peor que el resto | Informar la limitación o excluir el estrato |
| G8. Prueba final | Se ejecutó una sola vez, sobre el candidato ya elegido | Repetir con datos nuevos |

Si alguna puerta no se cumple, el territorio no debe tratarse como validado.

## 5. ¿Es reutilizable `model.pt` fuera de Cali?

**Juicio: no para producción; sí como código y como punto de partida experimental con cambios.**

Hechos verificados:

1. `model.pt` contiene un `state_dict` de un codificador a nivel de carácter con vocabulario ASCII genérico. La arquitectura no depende de Cali.
2. Los pesos se entrenaron únicamente con 330 387 direcciones del catastro de Cali y con el ruido de sus operadores.
3. Las cabezas `comuna_head` (24 clases) y `barrio_head` (342 clases) son específicas de Cali, pero **no se usan en inferencia**; por sí solas no impedirían usar el codificador.
4. `coord_head` **sí se usa** en inferencia, y el checkpoint guarda `xy_mean` y `xy_std` de los centroides de Cali (media de aproximadamente 729 887 y 871 633 m en EPSG:3116, con desviaciones de 2 903 y 3 333 m). En una ciudad lejana, la coordenada predicha estaría en Cali y el término `geo = exp(-d / 600 m)` valdría prácticamente cero para todos los candidatos (derivado del código, no medido).
5. El umbral 0.73 y los pesos se ajustaron con la presencia de ese término geográfico y con el ruido de Cali.
6. `catastro_emb.pt` debe calcularse con el índice de la nueva ciudad (`cali-address setup` lo hace para cualquier `catastro_docs.parquet` válido), pero el espacio de embeddings se aprendió con direcciones de Cali.
7. Continuar el entrenamiento desde `model.pt` (`--mode finetune`) falla con `ValueError: checkpoint comuna/barrio heads do not match the training data` si el número de clases difiere (`train.py`). Habría que modificar ese código, por ejemplo reinicializando las cabezas.

Qué haría falta para un piloto fuera de Cali (no probado):

- Desactivar el gazetteer o aportar límites propios, y desactivar el término geográfico. El `tuning.json` publicado trae una variante lista, `neural + rerank` (pesos `sim 0.7778`, `fuzz 0.1111`, `struct 0.1111`, `geo 0`, umbral 0.80), pero su umbral se ajustó con datos sintéticos de Cali.
- Reconstruir el índice y los embeddings.
- Recalibrar el umbral con las direcciones de la nueva ciudad.
- Validar con el protocolo de la sección 4 antes de usar cualquier resultado.

No se probó ninguna de estas opciones en una ciudad distinta de Cali; no existe evidencia de su desempeño. La recomendación es **reentrenar** con el catastro del territorio objetivo siguiendo [04-entrenamiento.md](04-entrenamiento.md), que cuesta unos 16 minutos de GPU en el hardware de referencia más la preparación de datos.
