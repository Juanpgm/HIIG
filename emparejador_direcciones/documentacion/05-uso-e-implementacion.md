# 05 - Uso e implementación

Guía práctica: instalación, uso como biblioteca y por línea de comandos, vinculación por lotes, patrón de integración en un servicio y notas de producción. Todos los ejemplos usan direcciones **inventadas**. Las salidas se obtuvieron ejecutando los scripts reales del proyecto (Python 3.14.4 en Windows) sin modificarlos.

## Contenido

1. [Instalación](#1-instalación)
2. [Comparar dos direcciones (línea de comandos)](#2-comparar-dos-direcciones-línea-de-comandos)
3. [Uso como biblioteca](#3-uso-como-biblioteca)
4. [Vincular contra una base (vincular.py)](#4-vincular-contra-una-base-vincularpy)
5. [Vinculación por lotes](#5-vinculación-por-lotes)
6. [Patrón de servicio](#6-patrón-de-servicio)
7. [Notas de producción](#7-notas-de-producción)

Antes de empezar, recuerde la precondición: la entrada debe venir de un normalizador ([06](06-predicciones.md)). Para qué hacer con los resultados, ver también [08](08-metricas-y-confiabilidad.md) y [09](09-advertencias-y-limitaciones.md).

## 1. Instalación

No hay paquete instalable (no existe `pyproject.toml`). El runtime solo usa la biblioteca estándar de Python 3.12 o superior.

```bash
git clone <url-del-repositorio>
cd emparejador_direcciones
python --version        # debe ser 3.12 o superior
python scripts/emparejar.py "KR 1 # 4 - 80" "kr 1 #4-80"
```

Los scripts agregan `src/` a `sys.path` por su cuenta. Si usa la biblioteca desde su propio código, agregue `src/` a `sys.path` o a `PYTHONPATH`:

```python
import sys
sys.path.insert(0, "ruta/al/proyecto/src")
from emparejador import match
```

Dependencias opcionales (solo para evaluación o pruebas): `pip install -r requirements.txt` (instala `pytest==9.0.3`) y `pip install -r requirements-dev.txt` (agrega `pyarrow==25.0.1`). Detalle en [07](07-requisitos.md).

## 2. Comparar dos direcciones (línea de comandos)

Coincidencia exacta con formato distinto:

```text
$ python scripts/emparejar.py "KR 1 # 4 - 80" "kr 1 #4-80"
MATCH  puntaje=1.0000  umbral=0.9
A: KR 1 # 4 - 80
B: KR 1 # 4 - 80
motivo: coincidencia exacta
[código de salida 0]
```

Placa distinta sin tolerancia (veto) y con tolerancia 1:

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # 10 - 21"
NO_MATCH  puntaje=0.0000  umbral=0.9
A: CL 5 # 10 - 20
B: CL 5 # 10 - 21
motivo: veto: placa difiere (20 vs 21)
[código de salida 1]

$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # 10 - 21" --tolerancia-placa 1
MATCH  puntaje=0.9600  umbral=0.9
A: CL 5 # 10 - 20
B: CL 5 # 10 - 21
motivo: coincide; diferencias menores: plate
[código de salida 0]
```

Salida JSON (recortada; el objeto completo incluye `fields` con 16 entradas y los campos de `a` y `b`):

```text
$ python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # KR 10 - 21 AP 301" --tolerancia-placa 1 --json
{"score": 0.9126, "threshold": 0.9, "reason": "coincide; diferencias menores: plate, complement, cross_type",
 "fields": [ ... ], "warnings": [], "match": true, "resultado": "MATCH", "plate_tolerance": 1,
 "a": {"input": "CL 5 # 10 - 20", "parse_ok": true, "canonical": "CL 5 # 10 - 20", "fields": {...}, "notes": []},
 "b": {"input": "CL 5 # KR 10 - 21 AP 301", "parse_ok": true, "canonical": "CL 5 # KR 10 - 21 AP 301", "fields": {...}, "notes": []}}
```

(En la salida real todo va en una sola línea.)

Entrada que no se puede leer, con su motivo:

```text
$ python scripts/emparejar.py "hola mundo" "CL 5 # 10 - 20"
NO_MATCH  puntaje=0.0000  umbral=0.9
A: hola mundo
B: CL 5 # 10 - 20
motivo: veto: dirección A no canónica (falta_separador_hash)
[código de salida 1]
```

## 3. Uso como biblioteca

Script ejecutado (se antepone la ruta de `src/` a `sys.path`):

```python
from emparejador import match, parse_canonical, render, clean_text, AddressIndex

# Comparación en nivel por defecto
r = match("CL 5 # 10 - 20", "Calle 5 No. 10-20")
print(r.is_match, r.score, r.reason)        # True 1.0 coincidencia exacta   (ejecutado)

# Niveles
a, b = "CL 5 # 10 - 20 TO A AP 301", "CL 5 # 10 - 20 TO A AP 402"
print(match(a, b, level="structure").score)  # 1.0   (ejecutado)
print(match(a, b).score)                     # 0.0   (veto de complemento, nivel unit)

# Análisis sintáctico y reconstrucción canónica
p = parse_canonical("CL 5 A BIS NORTE # KR 10 B - 20 AP 301")
print(p.via_type, p.via_number, p.via_letters, p.via_bis, p.via_quadrant)  # CL 5 A True NORTE
print(render(p))                    # CL 5 A BIS NORTE # KR 10 B - 20 AP 301
print(clean_text("Calle 5 Nº 10-20"))  # CALLE 5 # 10 - 20  (solo higiene: no abrevia el tipo)
```

Salidas ejecutadas de las líneas anteriores: `True 1.0 coincidencia exacta`; `1.0`; `0.0`; la reconstrucción canónica `CL 5 A BIS NORTE # KR 10 B - 20 AP 301`; y `CALLE 5 # 10 - 20` para `clean_text`. Tenga en cuenta que `clean_text` no convierte `CALLE` en `CL`: la conversión de alias ocurre en `parse_canonical`.

Manejo de errores (ejecutado):

```python
match("a", "b", threshold=2)        # InvalidThresholdError: umbral fuera de rango [0,1]: 2.0
match("a", "b", level="x")          # InvalidLevelError: level must be one of ('unit', 'building', 'structure'): 'x'
match("a", "b", plate_tolerance=-1) # InvalidToleranceError: tolerancia de placa negativa: -1
match(None, "CL 5 # 10 - 20").reason  # 'veto: dirección A no canónica (entrada_no_texto)'
```

Índice (ejecutado sobre una base inventada de tres registros legibles y uno ilegible):

```python
idx = AddressIndex.from_records([
    ("P001", "CL 5 # 10 - 20"),
    ("P002", "CL 5 # 10 - 20 AP 301"),
    ("P003", "CL 5 # 10 - 20 AP 302"),
    ("P009", "basura"),
])
idx.stats()   # {'records': 3, 'blocks': 1, 'rejected': 1, 'max_block_size': 3}

res = idx.find("CL 5 # 10 - 20")
[(c.record_id, c.score) for c in res.candidates]
# [('P001', 1.0), ('P002', 0.98), ('P003', 0.98)]
res.ambiguous, res.near_ambiguous, res.alternatives, res.truncated, res.n_compared
# (False, True, 2, False, 3)

res = idx.find("CL 5 # 10 - 20 AP 301")
[(c.record_id, c.score) for c in res.candidates]   # [('P002', 1.0), ('P001', 0.98)]

idx.find("###").reason   # 'consulta no canónica (numero_via_faltante)'
```

Observe que una consulta sin complemento obtiene `ambiguous=False` pero `near_ambiguous=True`: otras dos unidades del mismo edificio superan el umbral con 0,98. Ese es el caso que `near_ambiguous` existe para señalar.

## 4. Vincular contra una base (vincular.py)

Base inventada (`base.csv`, UTF-8, separada por comas):

```text
id,direccion
P001,CL 5 # 10 - 20
P002,CL 5 # 10 - 20 AP 301
P003,CL 5 # 10 - 20 AP 302
P004,CL 5 # KR 10 - 21
P005,KR 7 # 3 - 15
P006,texto libre sin estructura
```

Ejecución real:

```text
$ python scripts/vincular.py --base base.csv --id-column id --address-column direccion "CL 5 # 10 - 20"
MATCH  candidatos=3  comparados=3  umbral=0.9
consulta: CL 5 # 10 - 20
base: 5 registros indexados, 1 rechazados
alternativas: 2
nota: otras parcelas también superan el umbral (mismo predio con complemento conocido o variante de tipo de cruce); revise las alternativas antes de vincular
1. P001  puntaje=1.0000  CL 5 # 10 - 20
2. P002  puntaje=0.9800  CL 5 # 10 - 20 AP 301
3. P003  puntaje=0.9800  CL 5 # 10 - 20 AP 302
[código de salida 0]
```

Con `--json` para una consulta sin alternativas:

```text
$ python scripts/vincular.py --base base.csv --id-column id --address-column direccion "KR 7 # 3 - 15" --json
{"consulta": "KR 7 # 3 - 15", "canonica": "KR 7 # 3 - 15", "parse_ok": true, "motivo": "", "umbral": 0.9, "tolerancia_placa": 0, "n_comparados": 1, "n_candidatos": 1, "ambiguo": false, "casi_ambiguo": false, "alternativas": 0, "truncado": false, "mejor": {"id": "P005", "puntaje": 1.0, "canonica": "KR 7 # 3 - 15", "avisos": []}, "candidatos": [{"id": "P005", "puntaje": 1.0, "canonica": "KR 7 # 3 - 15", "avisos": []}], "registros": 5, "rechazados": 1, "ids_vacios": 0}
```

Sin coincidencia:

```text
$ python scripts/vincular.py --base base.csv --id-column id --address-column direccion "CL 9 # 9 - 9"
SIN_COINCIDENCIA  candidatos=0  comparados=0  umbral=0.9
consulta: CL 9 # 9 - 09
base: 5 registros indexados, 1 rechazados
[código de salida 1]
```

(La línea `consulta:` muestra la forma canónica reconstruida, con la placa rellenada a dos dígitos.)

Límite de `--top` y tolerancia:

```text
$ python scripts/vincular.py ... "CL 5 # 10 - 21" --tolerancia-placa 1 --top 2
MATCH  candidatos=4  comparados=4  umbral=0.9
...
1. P004  puntaje=0.9700  CL 5 # KR 10 - 21
2. P001  puntaje=0.9600  CL 5 # 10 - 20
... y 2 candidatos más
```

Un efecto a tener en cuenta: con tolerancia 1 la coincidencia exacta de la placa no se prioriza por encima de la variante de cruce, porque el orden es por puntaje (0,97 supera a 0,96).

## 5. Vinculación por lotes

`vincular.py` recibe **una** consulta por invocación y reconstruye el índice cada vez. Para muchas consultas, construya el índice una sola vez desde Python. Ejemplo ejecutado (con `base.csv` como arriba y un archivo `consultas.csv` con columnas `consulta_id,direccion`):

```python
import csv
import sys
from collections import Counter

sys.path.insert(0, "src")  # ejecutar desde la raíz del proyecto
from emparejador import AddressIndex

index = AddressIndex(plate_tolerance=0)
with open("base.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        index.add(row["id"], row["direccion"])
print("base:", index.stats())

resumen = Counter()
with open("consultas.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        r = index.find(row["direccion"])
        if not r.parse_ok:
            estado = "consulta_ilegible"       # NO es lo mismo que "sin candidato"
        elif r.best is None:
            estado = "sin_candidato"
        elif r.near_ambiguous:
            estado = "revisar"
        else:
            estado = "enlazado"
        resumen[estado] += 1
        print(row["consulta_id"], estado, r.best.record_id if r.best else "")
print(dict(resumen))
```

Salida obtenida con cinco consultas inventadas:

```text
base: {'records': 5, 'blocks': 3, 'rejected': 1, 'max_block_size': 3}
Q1 revisar P001
Q2 enlazado P005
Q3 sin_candidato
Q4 consulta_ilegible
Q5 revisar P002
{'revisar': 2, 'enlazado': 1, 'sin_candidato': 1, 'consulta_ilegible': 1}
```

Reporte siempre `index.rejected` y el número de consultas ilegibles junto con la tasa de vinculación: ambas son direcciones que el parser no pudo leer, no direcciones sin contraparte.

## 6. Patrón de servicio

El comparador es una biblioteca; no incluye servidor. Un esbozo mínimo (ejecutado, sin dependencias de framework) que puede envolverse en cualquier framework web:

```python
import csv
import sys

sys.path.insert(0, "ruta/al/proyecto/src")
from emparejador import AddressIndex


def construir_indice(ruta_csv: str) -> AddressIndex:
    index = AddressIndex(plate_tolerance=0)
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            index.add(fila["id"], fila["direccion"])
    return index


INDEX = construir_indice("base.csv")  # una vez, al iniciar el proceso


def vincular(direccion_canonica: str, top: int = 5) -> dict:
    r = INDEX.find(direccion_canonica, threshold=0.90)
    if not r.parse_ok:
        return {"estado": "consulta_ilegible", "motivo": r.reason}
    if r.best is None:
        return {"estado": "sin_candidato"}
    return {
        "estado": "revisar" if r.near_ambiguous else "enlazado",
        "mejor": {"id": r.best.record_id, "puntaje": r.best.score},
        "alternativas": r.alternatives,
        "avisos": list(r.best.result.warnings),
        "candidatos": [{"id": c.record_id, "puntaje": c.score} for c in r.candidates[:top]],
        "truncado": r.truncated,
    }
```

Salida de `vincular("CL 5 # 10 - 20")` con la base inventada:

```json
{"estado": "revisar", "mejor": {"id": "P001", "puntaje": 1.0}, "alternativas": 2, "avisos": [], "candidatos": [{"id": "P001", "puntaje": 1.0}, {"id": "P002", "puntaje": 0.98}, {"id": "P003", "puntaje": 0.98}], "truncado": false}
```

y de `vincular("basura")`: `{"estado": "consulta_ilegible", "motivo": "consulta no canónica (falta_separador_hash)"}`.

Recomendaciones de diseño del servicio:

- Coloque el normalizador **antes** de este paso y registre qué direcciones no pudo normalizar.
- Devuelva siempre el estado y las alternativas, no solo el mejor candidato.
- Registre `reason`, `warnings` y `n_compared` para auditoría.

## 7. Notas de producción

| Tema | Detalle |
|---|---|
| Determinismo | No hay aleatoriedad en ejecución. La misma entrada produce la misma salida; los candidatos se ordenan por (-puntaje, id) con orden estable incluso con ids de tipos mixtos |
| Memoria | Crece linealmente con el número de registros; el tamaño real depende de la longitud de sus direcciones. Mídala localmente con la sección H de `tools/eval_catastro.py` |
| Tiempo de arranque | Construir el índice incluye el parseo de todas las filas (mídalo con la sección H de `tools/eval_catastro.py`). No hay serialización: se reconstruye al iniciar |
| Hilos | Según el código, `find` solo lee estructuras del índice (no modifica diccionarios ni listas del índice), por lo que varias consultas simultáneas sobre un índice ya construido no comparten estado mutable. `add` sí modifica el índice y no es seguro de ejecutar en paralelo con consultas ni con otro `add`. Esto es una lectura del código; no existe una prueba de concurrencia en la documentación original |
| Rendimiento | Un solo hilo; mida las consultas por segundo con su propia base (sección H de `tools/eval_catastro.py`). Para más caudal use varios procesos, cada uno con su índice |
| Bases con cubetas grandes | Si muchas filas comparten vía, cruce y placa, `find` las recorre todas. `max_block_scan` limita el costo, pero marca `truncated` y puede omitir el mejor candidato |
| Codificación | Los scripts reconfiguran stdout a UTF-8; el CSV se lee con `--encoding` (UTF-8 por defecto, BOM admitido) |
| Registro de actividad | El proyecto no emite registros; añádalos en su capa de integración |
| Versión de Python | El código fuente declara 3.12 como mínimo; se ejecutó aquí con 3.14.4 |
