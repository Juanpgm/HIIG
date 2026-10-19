# 06 - Predicciones: entrada, salida e interpretación

Este documento define qué entrada exige el comparador, qué devuelve campo por campo y cómo interpretar cada resultado, con ejemplos inventados ejecutados sobre el código real. Para la teoría ver [01](01-teoria.md); para la confiabilidad ver [08](08-metricas-y-confiabilidad.md).

## Contenido

1. [Precondición: entrada normalizada](#1-precondición-entrada-normalizada)
2. [Formato de entrada](#2-formato-de-entrada)
3. [Esquema de salida](#3-esquema-de-salida)
4. [Cadenas de motivo (reason)](#4-cadenas-de-motivo-reason)
5. [Avisos (warnings)](#5-avisos-warnings)
6. [Cómo interpretar puntajes y ambigüedad](#6-cómo-interpretar-puntajes-y-ambigüedad)
7. [Ejemplos resueltos](#7-ejemplos-resueltos)

## 1. Precondición: entrada normalizada

> **Este modelo espera direcciones ya normalizadas por un paso previo.** No normaliza texto libre. Un normalizador externo (en este ecosistema, el proyecto hermano `normalizador_direcciones`) debe producir las cadenas canónicas antes de llamar al comparador. Si usted pasa texto crudo, la mayoría de las direcciones no se podrán leer y se puntuarán como veto (0,0).

Evidencia: con texto crudo de personas, la tasa de lectura y el recall son mucho menores que con texto ya normalizado. Los resultados de las corridas no se publican; mida ambos escenarios con `tools/eval_realworld.py` (ver [08](08-metricas-y-confiabilidad.md)).

El comparador solo hace higiene superficial (mayúsculas, tildes, variantes de guion y de `#`, tokens pegados). No repara errores tipográficos, no elimina el nombre de la ciudad, no separa varias direcciones y no adivina separadores ausentes salvo la forma estricta `TIPO N N - N`.

## 2. Formato de entrada

- **Tipo:** cadena de texto (`str`). Un valor que no es texto (`None`, número) produce `entrada_no_texto`; una cadena vacía o solo con espacios produce `entrada_vacia`.
- **Gramática:**

```text
SEGMENT := [TYPE] NUMBER [LETTER] [SUFFIX_NUMBER [SUFFIX_LETTER]] [BIS] [QUADRANT]
ADDRESS := SEGMENT_VIA "#" [SEGMENT_CROSS] ["-" [PLATE]] [COMPLEMENT...]
```

- **Ejemplos válidos (inventados):** `KR 31 F 2 # 84 - 15`, `CL 14 # 52 BIS - 27`, `AV 15 OESTE # 9 OESTE - 137`, `CL 25 NORTE # AV 6 - 30`, `CL 12 A # 52 - 64 AP 301 TO 2`, `KR 96F # 96 - 66`.
- **Formas crudas también aceptadas** (porque el catastro de Cali las contiene): tipos de vía de una letra (`K`, `C`, `D`, `T`, `A`), placa final vacía (`C 1 # 74 -`), complemento en la posición de placa (`KR 69 # 33 - LT 20`), separadores duplicados (`# #`, `- -`), cuadrantes de una letra tras una letra de calle (`C 70 B N`), colas no registradas que se conservan como complemento opaco (`8 C`, `GASS 5`), `NUMERO` en lugar de `#` y la forma `TIPO N N - N` sin `#`.
- **Límites:**

| Límite | Valor | Resultado si se excede |
|---|---|---|
| Cola de complemento (tokens unidos con un espacio) | 40 caracteres | `complemento_demasiado_largo` |
| Dígitos significativos de la placa | 6 | `placa_invalida` |
| Varias direcciones (`;`, un segundo `#`, un tipo de vía seguido de número en la cola) | no admitido | `multiples_direcciones` |
| Cuadrante suelto como primer token de la cola | no admitido | `token_inesperado:<token>` |
| Caracteres en la cola | solo letras, dígitos y `/` | `token_inesperado:<token>` |
| Longitud total de la entrada | sin tope explícito; el costo es lineal | - |

- **Codificación:** el comparador trabaja con cadenas Unicode; aplica NFKD y elimina tildes. Los scripts leen CSV con `--encoding` (UTF-8 por defecto). Los dígitos de otros sistemas de escritura sin descomposición Unicode (por ejemplo árabe-índicos) se rechazan; los de compatibilidad (superíndices, anchos completos) se pliegan a ASCII.
- **Notas de fallo de lectura** (`CanonicalAddress.notes`, `LinkResult.reason`): `entrada_no_texto`, `entrada_vacia`, `multiples_direcciones`, `falta_separador_hash`, `numero_via_faltante`, `token_inesperado:<token>`, `complemento_demasiado_largo`, `placa_invalida`.

Una dirección que sigue el estándar nacional sin `#` ni `-` (por ejemplo `CL 11 SUR 23 A BIS 50`) **no se lee** (`falta_separador_hash`). Ver [10](10-estandar-y-generalizacion.md).

## 3. Esquema de salida

### 3.1 `MatchResult` (función `match`)

| Campo | Tipo | Descripción |
|---|---|---|
| `is_match` | `bool` | Verdadero si no hay veto y `score >= threshold` |
| `score` | `float` | De 0,0 a 1,0 con 4 decimales; 0,0 si hay veto o fallo de lectura |
| `threshold` | `float` | Umbral usado |
| `reason` | `str` | Explicación en español (ver sección 4) |
| `parsed_a`, `parsed_b` | `CanonicalAddress` | Los lados leídos; ver [02](02-arquitectura.md) |
| `fields` | `tuple[FieldComparison, ...]` | Detalle por campo; vacío si hubo fallo de lectura |
| `warnings` | `tuple[str, ...]` | Ver sección 5 |

`FieldComparison`: `field`, `a`, `b`, `status` (`ausente_en_ambos`, `igual`, `equivalente`, `solo_en_uno`, `difiere`, `dentro_tolerancia`), `penalty` (0,0 a 0,30), `veto` (booleano).

Orden de los campos en `fields`: `via_type`, `via_number`, `cross_number`, `plate`, `complement`, `via_letters`, `via_suffix`, `via_suffix_letters`, `via_quadrant`, `cross_letters`, `cross_suffix`, `cross_suffix_letters`, `cross_quadrant`, `via_bis`, `cross_bis`, `cross_type`.

### 3.2 JSON de `emparejar.py --json`

Se obtiene de `MatchResult.to_dict()` con tres renombrados y dos claves nuevas:

| Clave JSON | Origen |
|---|---|
| `match` | `is_match` |
| `resultado` | `"MATCH"` o `"NO_MATCH"` |
| `score`, `threshold`, `reason`, `fields`, `warnings` | iguales |
| `plate_tolerance` | tolerancia aplicada |
| `a`, `b` | `parsed_a`, `parsed_b` convertidos |

Cada `a` / `b` tiene: `input` (texto original), `parse_ok`, `canonical` (forma reconstruida, o el texto original si no se pudo leer), `fields` (los 16 campos de `CanonicalAddress`; `complement` es una lista de pares `[tipo, valor]`) y `notes`.

### 3.3 `LinkResult` (índice y `vincular.py`)

| Campo | Descripción |
|---|---|
| `query`, `parse_ok`, `reason` | Consulta leída; si no se leyó, `reason` explica por qué |
| `candidates` | Solo registros con puntaje sobre el umbral; mejor primero; un candidato por id distinto |
| `best` | Primer candidato o `None` |
| `n_compared` | Registros realmente puntuados (incluye los que quedaron bajo el umbral) |
| `ambiguous` | Más de un id distinto empata en el mejor puntaje |
| `near_ambiguous` / `alternatives` | Otro id distinto también supera el umbral aunque no empate |
| `truncated` | `max_block_scan` cortó una cubeta; puede faltar un candidato mejor |

JSON de `vincular.py`: ver las claves en [03](03-configuracion.md), sección 3. Los puntajes de los candidatos van en la clave `puntaje`, no `score`.

## 4. Cadenas de motivo (reason)

| Situación | Texto de `reason` |
|---|---|
| Sin diferencias con penalización | `coincidencia exacta` |
| Coincide con penalizaciones | `coincide; diferencias menores: <campos separados por coma>` (nombres de campo del código, por ejemplo `plate, complement, cross_type`) |
| No coincide sin veto | `puntaje 0.8500 < umbral 0.9: <campo> solo en un lado` o `<campo> difiere` |
| Veto de campo ausente de un lado | `veto: <etiqueta> presente solo en un lado` |
| Veto de valor distinto | `veto: <etiqueta> difiere (<a> vs <b>)` |
| Fallo de lectura | `veto: dirección A no canónica (<notas>)` y/o `dirección B no canónica (...)` |
| Índice, consulta ilegible | `consulta no canónica (<notas>)` |
| Índice, sin candidatos | `sin candidatos sobre el umbral` |

Etiquetas de veto: `tipo de vía`, `número de vía`, `número de vía generadora`, `placa`, `complemento`. Un complemento se muestra como pares `TIPO VALOR` (por ejemplo `AP 301 vs AP 302`); un valor con varios tokens se muestra con `/` entre los números (por ejemplo `ED LOS/ARBOLES`) porque así lo guarda el parser.

## 5. Avisos (warnings)

| Aviso | Cuándo se emite | Qué implica |
|---|---|---|
| `sin_placa` | Ninguna de las dos direcciones tiene placa | La comparación se apoya en menos estructura (esquina o dirección incompleta) |
| `sin_cruce` | Ninguna de las dos direcciones tiene número de cruce | Ídem |
| `complemento_no_estructurado` | Cualquiera de las dos tiene un fragmento opaco (tipo no registrado, o tipo registrado sin valor útil) | El complemento solo se comparó por igualdad textual |
| `cadena_identica` | Las dos cadenas son iguales (tras `clean_text`) y la gramática solo las rechaza por complemento demasiado largo o por varias direcciones | `MATCH` con puntaje 1,0 sin que la dirección se haya podido leer entera; revise manualmente |

Los avisos no cambian el puntaje; son una señal para decidir si revisar. Se calculan sobre los complementos originales, incluso en los niveles `building` y `structure`.

## 6. Cómo interpretar puntajes y ambigüedad

El puntaje no es una probabilidad ([01](01-teoria.md), sección 9). Tome cada valor como una categoría:

| Puntaje | Significado | Acción sugerida |
|---|---|---|
| 1,0000 | Misma estructura en todos los campos | Aceptable, salvo que `near_ambiguous` sea verdadero o haya avisos |
| 0,9800 | Una dirección tiene un complemento conocido con valor y la otra no | Mismo edificio o predio; no necesariamente la misma unidad. Revisar si importa la unidad |
| 0,9700 | Tipo de la vía de cruce presente en un solo lado | Coincide por diseño; el catastro suele omitir el tipo de cruce. Puede ser un predio distinto (ver [08](08-metricas-y-confiabilidad.md)) |
| 0,9600 | Placa dentro de la tolerancia | Solo ocurre con `plate_tolerance > 0`; revisar |
| 0,9126 a 0,9506 | Combinación de los anteriores | Revisar |
| 0,8500 | Una diferencia estructural de un lado (letra, BIS, cuadrante, sufijo ausente) o cola opaca en un lado | No coincide con el umbral por defecto |
| 0,8330 | 0,85 por 0,98 | No coincide |
| 0,7000 | Un conflicto explícito (por ejemplo letras distintas) | No coincide |
| 0,5950 o menos | Varias diferencias | No coincide |
| 0,0000 | Veto o fallo de lectura | No coincide; ver `reason` |

Reglas de ambigüedad:

- `ambiguous = True`: más de un id distinto empata en el mejor puntaje. Ocurre cuando varias parcelas comparten la misma dirección canónica en la base (268 grupos en el catastro de Cali evaluado) o cuando varias unidades empatan.
- `near_ambiguous = True`: otro id distinto también supera el umbral aunque sin empate. Un 1,0 puede tener detrás una unidad del mismo edificio con 0,98 o una variante de cruce con 0,97.
- `truncated = True`: no confíe en que el mejor candidato sea el mejor de la base.
- `parse_ok = False` en una consulta: **no** es "sin candidato"; es una dirección que no se pudo leer. Trátela por separado.

## 7. Ejemplos resueltos

Todos con direcciones inventadas; las salidas son reales.

### MATCH por formato distinto (1,0)

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20" "Calle 5 No. 10-20"
MATCH  puntaje=1.0000  umbral=0.9
A: CL 5 # 10 - 20
B: CL 5 # 10 - 20
motivo: coincidencia exacta
```

### MATCH con complemento en un solo lado (0,98)

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # 10 - 20 AP 301"
MATCH  puntaje=0.9800  umbral=0.9
motivo: coincide; diferencias menores: complement
```

### MATCH con variante de tipo de cruce (0,97)

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # KR 10 - 20"
MATCH  puntaje=0.9700  umbral=0.9
motivo: coincide; diferencias menores: cross_type
```

### MATCH con tolerancia de placa (0,96)

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # 10 - 21" --tolerancia-placa 1
MATCH  puntaje=0.9600  umbral=0.9
motivo: coincide; diferencias menores: plate
```

### NO_MATCH por cada tipo de veto

| Veto | A | B | Resultado y motivo |
|---|---|---|---|
| Tipo de vía | `CL 5 # 10 - 20` | `KR 5 # 10 - 20` | 0,0 `veto: tipo de vía difiere (CL vs KR)` |
| Tipo de vía ausente en un lado | `5 # 10 - 20` | `CL 5 # 10 - 20` | 0,0 `veto: tipo de vía presente solo en un lado` |
| Número de vía | `CL 5 # 10 - 20` | `CL 6 # 10 - 20` | 0,0 `veto: número de vía difiere (5 vs 6)` |
| Número de cruce | `CL 5 # 10 - 20` | `CL 5 # 11 - 20` | 0,0 `veto: número de vía generadora difiere (10 vs 11)` |
| Cruce ausente en un lado | `CL 5 # - 20` | `CL 5 # 10 - 20` | 0,0 `veto: número de vía generadora presente solo en un lado` |
| Placa distinta | `CL 5 # 10 - 20` | `CL 5 # 10 - 21` | 0,0 `veto: placa difiere (20 vs 21)` |
| Letra de placa | `CL 5 # 10 - 20 A` | `CL 5 # 10 - 20` | 0,0 `veto: placa difiere (20A vs 20)` |
| Placa en un solo lado | `CL 5 # 10 - 20` | `CL 5 # 10` | 0,0 `veto: placa presente solo en un lado` |
| Complemento distinto | `CL 5 # 10 - 20 AP 301` | `CL 5 # 10 - 20 AP 302` | 0,0 `veto: complemento difiere (AP 301 vs AP 302)` |
| Torre distinta | `CL 5 # 10 - 20 TO A` | `CL 5 # 10 - 20 TO B` | 0,0 `veto: complemento difiere (TO A vs TO B)` |
| Fallo de lectura | `hola mundo` | `CL 5 # 10 - 20` | 0,0 `veto: dirección A no canónica (falta_separador_hash)` |
| Varias direcciones | `CL 5 # 10 - 20; CL 6 # 11 - 21` | `CL 5 # 10 - 20` | 0,0 `... (multiples_direcciones)` |
| Placa de más de 6 dígitos | `CL 5 # 10 - 20` | `CL 5 # 10 - 1234567` | 0,0 `... (placa_invalida)` |

### NO_MATCH sin veto (puntaje bajo el umbral)

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 A # 10 - 20"
NO_MATCH  puntaje=0.8500  umbral=0.9
motivo: puntaje 0.8500 < umbral 0.9: via_letters solo en un lado

$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # 10 - 20 8 C"
NO_MATCH  puntaje=0.8500  umbral=0.9
B: CL 5 # 10 - 20 8C
motivo: puntaje 0.8500 < umbral 0.9: complement solo en un lado
avisos: complemento_no_estructurado
```

El segundo ejemplo muestra por qué la cola opaca `8C` no se vincula automáticamente con la dirección sin cola. Con `--umbral 0.85` el mismo par sí coincide (`MATCH puntaje=0.8500`), lo que ilustra el riesgo de bajar el umbral.

### Complemento conocido sin valor (se trata como opaco)

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20 LT" "CL 5 # 10 - 20"
NO_MATCH  puntaje=0.8500  umbral=0.9
motivo: puntaje 0.8500 < umbral 0.9: complement solo en un lado
avisos: complemento_no_estructurado
```

### Dirección sin placa en ambos lados (aviso)

```text
$ python scripts/emparejar.py "CL 5 # 10" "CL 5 # 10"
MATCH  puntaje=1.0000  umbral=0.9
motivo: coincidencia exacta
avisos: sin_placa
```

### Ambigüedad en un índice

Con una base de tres registros `P001 = CL 5 # 10 - 20`, `P002 = CL 5 # 10 - 20 AP 301`, `P003 = CL 5 # 10 - 20 AP 302`, la consulta `CL 5 # 10 - 20` devuelve P001 (1,0) y dos alternativas con 0,98: `ambiguous=False`, `near_ambiguous=True`, `alternatives=2`. La consulta `CL 5 # 10 - 20 AP 301` devuelve P002 (1,0) y P001 (0,98).
