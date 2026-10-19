# 10 - Estándar nacional y generalización

Este documento compara la gramática del modelo con el estándar colombiano de nomenclatura de direcciones y con la estructura del Número Predial Nacional (NPN), documenta con honestidad las diferencias, propone cómo cerrarlas sin cambiar el código y explica cómo adaptar el modelo a otras ciudades de Colombia y a otros países.

**Hallazgo principal:** el estándar prohíbe el carácter `#` y la puntuación, y su forma normalizada no lleva separadores (`CL 11 SUR 23 A BIS 50`). La gramática canónica de este modelo, en cambio, **exige** `#` entre la vía y el cruce y `-` antes de la placa. Una dirección en formato estándar puro **no se lee** (`falta_separador_hash`). Esta discrepancia se explica en la sección 3.

## Contenido

1. [Qué establece el estándar](#1-qué-establece-el-estándar)
2. [Tabla de conformidad](#2-tabla-de-conformidad)
3. [La discrepancia de `#` y `-`: opciones](#3-la-discrepancia-de--y---opciones)
4. [Qué es genérico y qué es específico de Cali](#4-qué-es-genérico-y-qué-es-específico-de-cali)
5. [Adaptar a otras ciudades colombianas](#5-adaptar-a-otras-ciudades-colombianas)
6. [Adaptar a otros países](#6-adaptar-a-otros-países)
7. [Protocolo de validación en un territorio nuevo](#7-protocolo-de-validación-en-un-territorio-nuevo)

Relacionados: [01 teoría](01-teoria.md), [03 configuración](03-configuracion.md), [04 evaluación](04-evaluacion-y-calibracion.md), [09 limitaciones](09-advertencias-y-limitaciones.md).

## 1. Qué establece el estándar

Resumen del documento de lineamientos técnicos usado como referencia (que cita la Circular 300 de 2001 del IGAC, la Resolución 166 de 2004 del Ministerio de Educación Nacional y especificaciones del DANE):

- **Seis componentes en orden fijo:** tipo de vía, número o nombre de la vía principal, prefijo o cuadrante de la vía principal, número de la vía generadora, prefijo o cuadrante de la vía generadora, y número de placa o distancia.
- **Limpieza:** eliminar caracteres especiales y partículas innecesarias (`#`, `No.`, `N°`, indicadores ordinales como `7ma`), prohibir tildes y signos de puntuación y usar mayúsculas sostenidas.
- **Ejemplo oficial:** `Calle 11 Sur Número 23 A BIS - 50` se normaliza como `CL 11 SUR 23 A BIS 50`.
- **Abreviaturas obligatorias (algunas de las más críticas):** Autopista `AU`, Avenida `AV`, Avenida Calle `AC`, Bulevar `BL`, Calle `CL`, Diagonal `DG`, Transversal `TV`, Conjunto Residencial `CO`, Carrera `KR` (descartando `Cra`, `Crr`, `Krr`).
- **NPN de 30 posiciones:** departamento (1-2), municipio (3-5), zona (6-7), sector o comuna (8-9), barrio (10-13), manzana o vereda (14-17), terreno (18-21), condición del predio (22), torre o edificio de propiedad horizontal (23-26), unidad predial (27-30). El sistema anterior usaba 20 posiciones.
- **DIVIPOLA:** códigos del DANE para departamento (2 dígitos) y municipio (3 dígitos); por ejemplo 76 y 001 para Santiago de Cali.
- **LADM_COL:** perfil colombiano de la norma ISO 19152; el predio (`CR_Predio`) lleva el NPN y la dirección.

## 2. Tabla de conformidad

Cada fila se verificó ejecutando `parse_canonical` / `render` del proyecto sobre ejemplos inventados, salvo donde se indica lectura del código.

| Regla del estándar | Qué hace el modelo (verificado) | Estado | Acción |
|---|---|---|---|
| Orden de los seis componentes: tipo, vía, prefijo o cuadrante, cruce, prefijo o cuadrante, placa | El orden de la gramática es el mismo (`TIPO NUM [LETRA] [sufijo] [BIS] [CUADRANTE] # cruce - PLACA`). Añade número y letra de sufijo y el complemento tras la placa, que el estándar resumido no describe | Cumple (con extensiones) | Documentar las extensiones; ver [01](01-teoria.md) |
| Número **o nombre** de la vía principal | Exige un número de vía: `AV CIRCUNVALAR # 10 - 20` produce `numero_via_faltante` | Parcial | Para vías con nombre, el normalizador previo debe asignar un número o usar otra rama de gramática (sección 5) |
| Mayúsculas sostenidas | `clean_text` pasa a mayúsculas; `CL 5 # 10 - 20` en minúsculas se lee igual; la salida `render` está en mayúsculas | Cumple | Ninguna |
| Sin tildes | NFKD elimina las marcas combinantes (`Ñ` pasa a `N`) | Cumple | Ninguna |
| Sin signos de puntuación | Quita puntos finales de token y unifica guiones; **pero** la coma produce fallo (`CL 5, # 10 - 20` es `numero_via_faltante`), y `#` y `-` son obligatorios | Parcial | Eliminar comas en el paso previo |
| Eliminar `#` | La forma canónica del modelo **usa `#`** como separador obligatorio (`parser.py:6-7`); `render` lo emite siempre | No cumple (divergencia deliberada) | Adaptador previo o cambio de código (sección 3) |
| Eliminar `No.` y `N°` | Los acepta en la entrada y los convierte en `#` (`NO`, `NRO`, `NUM`, `N°`; `NUMERO` en posición específica) | Parcial (acepta, pero no elimina) | Ninguna si se mantiene la gramática |
| Eliminar ordinales (`7ma`, `5ta`) | `CL 5ta # 10 - 20` produce `token_inesperado:TA` | No cumple (no los depura) | El paso previo debe eliminarlos |
| La placa sin separador (`... 23 A BIS 50`) | La placa va tras `-`; sin `-` ni `#`, la dirección no se lee salvo la forma `TIPO N N - N` | No cumple | Adaptador (sección 3) |
| Cuadrantes (`SUR`, `NORTE`, `ESTE`, `OESTE`) | Se reconocen las palabras completas en cualquier posición válida | Cumple | Ninguna |
| Cuadrante abreviado a una letra | `S`, `N`, `E`, `O`, `W` son cuadrante solo tras letra, sufijo o BIS; junto al número de vía son letra de calle (`CL 11 S # 23 - 50` queda con letra `S`) | Parcial | Escribir la palabra completa en el paso previo |
| `A BIS` separado por espacios | Lo lee; también lee `ABIS` pegado (que emite el normalizador hermano) | Cumple | Ninguna |
| Abreviatura `AU` | Se lee (`AU 5 # 10 - 20`) | Cumple | Ninguna |
| Abreviatura `AV` | Se lee | Cumple | Ninguna |
| Abreviatura `AC` | Se lee | Cumple | Ninguna |
| Abreviatura `CL` | Se lee; alias `CALLE`, `CLL`, `C` | Cumple | Ninguna |
| Abreviatura `DG` | Se lee | Cumple | Ninguna |
| Abreviatura `TV` | Se lee | Cumple | Ninguna |
| Abreviatura `KR` | Se lee; `CRA`, `KRA`, `CARRERA`, `CR`, `K` se convierten a `KR` | Cumple | Ninguna |
| Abreviatura `BL` (Bulevar) | **No** es tipo de vía. `BL` es alias de bloque (`BLQ`) en los complementos: `BL 5 # 10 - 20` produce `numero_via_faltante` | No cumple (conflicto de significado) | Decidir qué significa `BL` en su dominio; añadir un tipo de vía distinto (por ejemplo `BV`) en `VIA_TYPES` y evitar la colisión |
| Abreviatura `CO` (Conjunto Residencial) | **No** es tipo de vía; el modelo representa el conjunto como complemento `CONJ`. `CO 5 # 10 - 20` produce `numero_via_faltante` | No cumple | Normalizar `CO` a `CONJ` en el paso previo si es un complemento |
| Distancia en lugar de placa | Solo hay placa numérica (hasta 6 dígitos) con letra opcional | Parcial | Una distancia numérica se lee como placa; no se distingue semánticamente |
| NPN de 30 posiciones | El comparador no usa el NPN. Las herramientas leen `numero_predial_nacional` como texto y toman los primeros 21 caracteres como "parcela de terreno" (`tools/eval_realworld.py:100`) | Parcial (solo herramientas) | Coherente con 30 posiciones (1-21 son hasta el terreno); un NPN legado de 20 posiciones no tendría clave |
| DIVIPOLA (municipio) | El modelo no maneja municipio. `AddressIndex` **no separa municipios**: dos direcciones iguales de municipios distintos serían el mismo bloque | No cumple (no incorporado) | Construir un índice por código DIVIPOLA de 5 dígitos, o filtrar por municipio antes de vincular |
| LADM_COL | No lo implementa ni depende de él; los ids de registro son opacos | No aplica | Use el NPN de `CR_Predio` como `record_id` para vincular de vuelta al modelo |

Recuento de las 25 filas: 12 cumple (incluida la que cumple con extensiones), 6 parcial, 6 no cumple y 1 no aplica.

## 3. La discrepancia de `#` y `-`: opciones

Hecho verificado: `parse_canonical("CL 11 SUR 23 A BIS 50")` devuelve `parse_ok=False` con la nota `falta_separador_hash`; `parse_canonical("CL 11 SUR # 23 A BIS - 50")` se lee correctamente. El ejemplo oficial del estándar, escrito como `Calle 11 Sur Número 23 A BIS – 50`, sí se lee (porque `Número` hace de `#`) y se reconstruye como `CL 11 SUR # 23 A BIS - 50`. Es decir, el modelo contiene la información, pero su forma canónica añade dos separadores.

Por qué existe: la gramática se diseñó para el catastro de Cali, que almacena `#` y `-`, y para la salida del normalizador hermano. Los separadores eliminan la ambigüedad entre vía y cruce cuando hay número de sufijo (`KR 31 E 3 # 84 - 15`).

Opciones (no se modificó el código):

| Opción | Descripción | Ventajas | Riesgos |
|---|---|---|---|
| A. Adaptador previo (recomendada) | Un paso que convierte la forma estándar a la gramática del modelo antes de comparar | No toca el código verificado; las métricas siguen siendo válidas | Debe segmentar vía y cruce; ver ambigüedad abajo |
| B. Salida del normalizador en la gramática del modelo | Que el normalizador emita `#` y `-` (como ya hace en este ecosistema) | Cero ambigüedad, porque el normalizador conoce los componentes | Hay que mantener dos formatos si se exporta el estándar |
| C. Cambio de código | Añadir un modo estándar a `parse_canonical` (segmentación de seis componentes) y una opción `style="standard"` en `render` | Entrada y salida conformes | Cambia el comportamiento verificado; exige volver a medir todo; la ambigüedad del sufijo numérico sigue |

### Esbozo del adaptador (no forma parte del proyecto; ejecutado)

```python
import re
import unicodedata

TIPOS = {"AU", "AV", "AC", "AK", "CL", "DG", "TV", "KR", "CV", "PJ", "PS", "CT"}
CUADRANTES = {"NORTE", "SUR", "ESTE", "OESTE"}


def _consumir_segmento(tokens, i):
    """NUM [LETRA] [BIS] [CUADRANTE]. Devuelve (tokens_del_segmento, nuevo_i) o None."""
    if i >= len(tokens) or not tokens[i].isdigit():
        return None
    seg = [tokens[i]]
    i += 1
    if i < len(tokens) and re.fullmatch(r"[A-Z]", tokens[i]):
        seg.append(tokens[i]); i += 1
    if i < len(tokens) and tokens[i] == "BIS":
        seg.append("BIS"); i += 1
    if i < len(tokens) and tokens[i] in CUADRANTES:
        seg.append(tokens[i]); i += 1
    return seg, i


def estandar_a_modelo(texto: str) -> str | None:
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c)).upper()
    tokens = t.split()
    if len(tokens) < 4 or tokens[0] not in TIPOS:
        return None
    via = _consumir_segmento(tokens, 1)
    if via is None:
        return None
    cruce = _consumir_segmento(tokens, via[1])
    if cruce is None:
        return None
    i = cruce[1]
    if i >= len(tokens) or not tokens[i].isdigit():
        return None
    placa, resto = tokens[i], tokens[i + 1:]
    return " ".join([tokens[0], *via[0], "#", *cruce[0], "-", placa, *resto])
```

Resultados ejecutados:

| Entrada (estándar) | Salida del adaptador | `parse_ok` del modelo |
|---|---|---|
| `CL 11 SUR 23 A BIS 50` | `CL 11 SUR # 23 A BIS - 50` | Sí; `match` contra la forma canónica da 1,0 |
| `KR 31 G 68 14` | `KR 31 G # 68 - 14` | Sí |
| `AV 15 OESTE 9 OESTE 137` | `AV 15 OESTE # 9 OESTE - 137` | Sí |
| `CL 5 10 20 AP 301` | `CL 5 # 10 - 20 AP 301` | Sí |
| `AV CIRCUNVALAR 10 20` | `None` (vía con nombre) | - |

**Advertencia de ambigüedad.** El estándar resumido no contempla un número de sufijo. Con la convención de Cali, `KR 31 E 3 84 15` (vía `KR 31 E 3`, cruce `84`, placa `15`) el adaptador produce `KR 31 E # 3 - 84 15`, que el modelo **lee sin error pero de forma distinta a la prevista** (cruce 3, placa 84, complemento opaco `15`). Por eso un adaptador sobre cadenas solo es seguro si el origen no usa números de sufijo; en caso contrario, hay que segmentar a partir de campos estructurados (opción B). Este caso es el mismo motivo por el que `_restore_missing_hash` solo acepta la forma `TIPO N N - N`.

## 4. Qué es genérico y qué es específico de Cali

| Componente | Genérico (reutilizable) | Específico (debe revisarse) |
|---|---|---|
| Idea de puntaje por producto de penalizaciones con vetos | Sí | Las constantes (`_PENALTIES`) son de diseño; ver [01](01-teoria.md) |
| Índice con bloqueo por campos vetables | Sí, mientras la clave sea un conjunto de campos que son veto duro | La clave (vía, cruce, placa) supone nomenclatura de malla |
| `clean_text` (Unicode, mayúsculas, guiones, tokens pegados) | En buena parte | `NUMERO`, `NRO`, `NUM`, `N°` son del español |
| Gramática de segmento y de dirección | Válida para nomenclatura colombiana de malla | `#`, `-`, `BIS`, cuadrantes en español |
| `VIA_TYPES` (`parser.py:58`) | - | Tipos y alias de Cali |
| `COMPLEMENT_KINDS` (`parser.py:139`) | - | Léxico de complementos en español y de Cali |
| `QUADRANTS` (`parser.py:116`), `_QUADRANT_ABBREVIATIONS` (`122`) y `_LETTER_WORD_RE` (`388`) | - | Palabras de cuadrante |
| Topes (`_MAX_TAIL_CHARS` 40, `MAX_PLATE_DIGITS` 6) | - | Calibrados con el catastro de Cali |
| `BUILDING_IGNORED_KINDS` y niveles | - | Derivados de una medición en Cali |
| Mensajes y motivos | - | Español |
| Herramientas de evaluación | Protocolos sí | Nombres de columnas y clave de 21 caracteres |

## 5. Adaptar a otras ciudades colombianas

Las ciudades colombianas comparten el estándar nacional y, probablemente, buena parte de la gramática, pero **esto no está validado**: el proyecto solo se midió con el catastro de Cali.

### Lista de verificación paso a paso

1. **Conseguir una muestra representativa** de direcciones del catastro local (miles de filas) y ponerlas en la forma canónica del modelo (sección 3).
2. **Medir la cobertura de lectura** con `tools/eval_catastro.py` (sección A del informe) para su catastro. Las "causas de fallo" y las firmas de forma indican qué escrituras locales faltan.
3. **Extender los tipos de vía** en `VIA_TYPES` (`parser.py:58`) y, si hay alias de dos palabras, en `_TWO_WORD_VIA_ALIASES` (`109`). Evite colisiones de significado (por ejemplo `BL`). No agregue alias ambiguos; el proyecto dejó fuera `CIR`/`CIRC` por esa razón.
4. **Extender los tipos de complemento** en `COMPLEMENT_KINDS` (`parser.py:139`). Cada alias nuevo cambia el puntaje de cadenas que antes eran opacas; vuelva a medir.
5. **Revisar los cuadrantes** (`QUADRANTS`, abreviaturas y `_LETTER_WORD_RE`); si su ciudad usa otras palabras, deben actualizarse las tres definiciones de forma coherente.
6. **Revisar los topes** (`_MAX_TAIL_CHARS`, `MAX_PLATE_DIGITS`) con la distribución de sus datos.
7. **Rurales**: el modelo no cubre direcciones rurales, circulares ni basadas en kilómetros; aparecen como fallo de lectura. Si son relevantes, hace falta una rama de gramática nueva (por ejemplo vereda o corregimiento como identificador principal).
8. **Separar por municipio** (DIVIPOLA): un índice por municipio o un filtro previo (sección 2).
9. **Repetir la medición de los tipos de complemento que se ignoran en `building`** con la clave de parcela de su catastro (los primeros 21 caracteres del NPN de 30 posiciones). No asuma que la lista de Cali sirve.
10. **Reejecutar** las evaluaciones, etiquetar pares y aplicar el protocolo de la sección 7.

## 6. Adaptar a otros países

El estándar colombiano no se aplica fuera de Colombia. Aplican solo las ideas de diseño. Para un país con otro sistema de direcciones:

1. **Documentar el modelo de dirección del país**: componentes, orden, separadores, códigos postales, ejes de numeración (calle y número, malla, manzana y lote, etc.) y el identificador oficial de parcela.
2. **Reescribir la gramática y las tablas** (`parser.py` completo): tipos de vía, abreviaturas, ordinales, idioma y alfabetos (la limpieza actual asume letras latinas y dígitos ASCII).
3. **Redefinir los campos de veto y blandos** en `matcher.py`: hoy son `via_type`, `via_number`, `cross_number` y `plate` (vetos) y letras, sufijos, cuadrantes y BIS (blandos). Un país con "número de calle + nombre de calle + código postal" necesita otros campos.
4. **Adaptar la clave de bloqueo** de `index.py` a los nuevos campos de veto duro; la propiedad de no pérdida solo vale si la clave contiene exclusivamente campos vetables.
5. **Reemplazar el mensaje y los avisos** por el idioma local.
6. **Reconstruir el arnés de evaluación** con el identificador de parcela y las columnas locales (sección 5.1 a 5.4 de [04](04-evaluacion-y-calibracion.md)).
7. Aplicar el protocolo de la sección 7.

Para un país distinto es probable que sea más rápido reescribir el parser conservando el patrón (veto duro, producto de penalizaciones, índice por bloques) que adaptar la gramática actual.

## 7. Protocolo de validación en un territorio nuevo

No confíe en los resultados en un territorio nuevo hasta cumplir lo siguiente:

1. **Cobertura de lectura**: al menos el 99 % de las filas de su catastro leídas, con las causas de fallo restantes documentadas (el criterio del proyecto para Cali es mayor o igual a 99,0 %).
2. **Invariantes**: cero violaciones de autocoincidencia y simetría (secciones B del arnés).
3. **Falsos vínculos internos**: cero pares con placa distinta que superen el umbral (E1) y revisión de los pares de tipo de cruce (E3).
4. **Ambigüedad inherente**: contar las cadenas canónicas compartidas por varias parcelas (E2) para dimensionar el porcentaje de resultados ambiguos.
5. **Etiquetas humanas**: construir con `build_groundtruth.py` y etiquetar pares (al menos los seis estratos), con evidencia catastral o de campo; calcular precisión y recall por estrato con intervalos de confianza.
6. **Texto real**: si hay texto sucio de usuarios, medir con `eval_realworld.py` en modo `normalized` y en modo `raw`; reportar ambos.
7. **Umbral y tolerancia**: elegirlos con una tabla como la de [03](03-configuracion.md), sección 7, **reproducida con sus datos**, y justificar cualquier valor distinto de 0,90 y 0.
8. **Escala de confiabilidad**: reconstruir la de [08](08-metricas-y-confiabilidad.md), sección 5, con sus propias tasas de error.
9. **Registro de resultados**: documentar números, protocolo, fecha, versión del código y limitaciones, y repetir ante cualquier cambio en las tablas o en la gramática.
10. **Revisión humana obligatoria** en todo uso con efectos legales, tributarios o catastrales.
