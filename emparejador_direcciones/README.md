# emparejador_direcciones

Comparador determinista de direcciones urbanas colombianas ya normalizadas. Recibe dos cadenas en forma canónica (por ejemplo `KR 1 # 4 - 80`) y responde **MATCH / NO_MATCH** junto con un puntaje entre 0 y 1. También permite vincular una dirección contra una base completa de direcciones (por ejemplo un catastro) mediante un índice en memoria con bloqueo.

**Estado:** versión preliminar probada con datos de Cali. No ha sido validada con etiquetas humanas ni en otros municipios (ver [08](documentacion/08-metricas-y-confiabilidad.md) y [09](documentacion/09-advertencias-y-limitaciones.md)).

> **Este modelo NO es una red neuronal.** No tiene capas, pesos, entrenamiento ni aprendizaje automático de ningún tipo. Es un analizador sintáctico (parser) de gramática estricta, más un puntuador de producto de penalizaciones y un índice por bloques. Por eso esta documentación no incluye diagramas de redes neuronales: serían inventados. En su lugar hay diagramas precisos del flujo de datos, de la gramática, del árbol de decisión del puntaje, del índice y de la secuencia de vinculación (ver [01](documentacion/01-teoria.md) y [02](documentacion/02-arquitectura.md)). Ventajas: auditable, reproducible (sin semillas en ejecución), sin archivos de pesos. Limitaciones: no generaliza a escrituras que la gramática no conoce y su puntaje no es una probabilidad calibrada.

## Inicio rápido

Requiere Python 3.12 o superior y ninguna dependencia de terceros. Desde la raíz del proyecto:

```bash
# 1. Comparar dos direcciones canónicas (código de salida 0 = MATCH, 1 = NO_MATCH, 2 = error de uso)
python scripts/emparejar.py "CL 5 # 10 - 20" "CL 5 # 10 - 21" --tolerancia-placa 1

# 2. Vincular una dirección contra una base CSV (columnas id y direccion)
python scripts/vincular.py --base base.csv --id-column id --address-column direccion "CL 5 # 10 - 20"

# 3. Usar como biblioteca
python -c "import sys; sys.path.insert(0, 'src'); from emparejador import match; print(match('KR 1 # 4 - 80', 'kr 1 #4-80').score)"
```

Salida real del primer comando (direcciones inventadas):

```text
MATCH  puntaje=0.9600  umbral=0.9
A: CL 5 # 10 - 20
B: CL 5 # 10 - 21
motivo: coincide; diferencias menores: plate
```

## Versión y paquete

Versión **0.2.0** (`emparejador.__version__`). El directorio incluye un `pyproject.toml` mínimo (diseño `src/`, sin dependencias de ejecución, Python >= 3.12): `pip install -e .` instala el paquete `emparejador`; los scripts y las pruebas también funcionan sin instalar (agregan `src/` a `sys.path`). Pruebas: `pip install -e ".[dev]"` y `python -m pytest -q` desde este directorio.

## Reglas de comparación recientes

- **Cuadrante por posición:** un cuadrante en la vía y otro en el cruce nunca se igualan.
- **`N` = `NORTE` solo dentro del mismo segmento y con prefijo idéntico** (tipo, número, sufijo, letras del sufijo y `BIS`); `E` y `S` siguen siendo letras de vía. Solo afecta la comparación; los vetos de placa, vía y cruce no cambian.
- **Cadenas idénticas:** `match(x, x)` es `MATCH` con el aviso `cadena_identica` aunque `x` exceda el complemento permitido o traiga varias direcciones legibles; no aplica a cadenas sin dígitos ni a otras entradas ilegibles.
- Detalle y ejemplos en [01-teoria.md](documentacion/01-teoria.md) y [06-predicciones.md](documentacion/06-predicciones.md).

## Validación con JEV (opcional)

Juez semántico externo e independiente. **Envía cadenas de dirección a un servicio de terceros**, por lo que su uso es una decisión explícita de quien lo ejecuta; consulte [`validacion_jev/README.md`](../validacion_jev/README.md) antes de usarlo.

## Precondición crítica

La entrada debe ser una dirección **ya normalizada** por un paso previo (en este ecosistema, el proyecto hermano `normalizador_direcciones`). Este modelo no limpia texto libre: sobre texto crudo su desempeño es mucho menor que sobre texto normalizado (ver [08](documentacion/08-metricas-y-confiabilidad.md)).

## Advertencias principales de confiabilidad

- El puntaje **no es una probabilidad** y toma valores discretos (ver [01](documentacion/01-teoria.md)).
- Las métricas perfectas de las pruebas sobre el catastro provienen de pruebas parcialmente circulares y no son evidencia de precisión real; los resultados de las corridas no se publican (ver [08](documentacion/08-metricas-y-confiabilidad.md)).
- Una coincidencia con puntaje 1,0 puede tener otra parcela distinta a continuación: revise siempre `near_ambiguous` / `alternativas`.
- No usar como única base de decisiones legales o catastrales sin revisión humana (ver [09](documentacion/09-advertencias-y-limitaciones.md)).
- La gramática usa `#` y `-` como separadores, lo que difiere del estándar nacional de direcciones (que los excluye); ver [10](documentacion/10-estandar-y-generalizacion.md).

## Documentación

| Archivo | Contenido |
|---|---|
| [documentacion/README.md](documentacion/README.md) | Índice de la documentación |
| [01-teoria.md](documentacion/01-teoria.md) | Problema, gramática, puntaje por producto de penalizaciones, vetos, niveles, bloqueo |
| [02-arquitectura.md](documentacion/02-arquitectura.md) | Mapa de módulos, diagramas, API pública, complejidad |
| [03-configuracion.md](documentacion/03-configuracion.md) | Parámetros, opciones de línea de comandos, códigos de salida |
| [04-evaluacion-y-calibracion.md](documentacion/04-evaluacion-y-calibracion.md) | Herramientas de evaluación (no hay entrenamiento) y cómo repetirlas en su ciudad |
| [05-uso-e-implementacion.md](documentacion/05-uso-e-implementacion.md) | Instalación, ejemplos ejecutados, integración |
| [06-predicciones.md](documentacion/06-predicciones.md) | Formato de entrada, esquema de salida, interpretación |
| [07-requisitos.md](documentacion/07-requisitos.md) | Requisitos de software, hardware y datos |
| [08-metricas-y-confiabilidad.md](documentacion/08-metricas-y-confiabilidad.md) | Definición de métricas, cómo reproducirlas, lo no medido, escala de confiabilidad |
| [09-advertencias-y-limitaciones.md](documentacion/09-advertencias-y-limitaciones.md) | Advertencias, elementos específicos de Cali/Colombia, modos de fallo |
| [10-estandar-y-generalizacion.md](documentacion/10-estandar-y-generalizacion.md) | Conformidad con el estándar nacional y adaptación a otros territorios |

## Requisitos (resumen)

| Uso | Software | Datos |
|---|---|---|
| Predicción (runtime) | Python >= 3.12, solo biblioteca estándar | Cadenas canónicas; base CSV UTF-8 con columnas de id y dirección |
| Evaluación / calibración | Python >= 3.12, `pyarrow==25.0.1`, opcionalmente `pytest==9.0.3` | Archivos parquet con el catastro de su ciudad (esquemas en [04](documentacion/04-evaluacion-y-calibracion.md)) |

Detalle completo en [07-requisitos.md](documentacion/07-requisitos.md). No se requiere GPU ni archivos de pesos.

## Estructura del proyecto

```text
src/emparejador/   parser.py, matcher.py, index.py, __init__.py
tests/             suite de pytest (direcciones sintéticas)
pyproject.toml     paquete mínimo (versión 0.2.0)
scripts/           emparejar.py, vincular.py
tools/             eval_catastro.py, build_groundtruth.py, eval_groundtruth.py, eval_realworld.py
docs/              informes de evaluación originales (en inglés)
documentacion/     esta documentación (en español)
```

## Licencias y datos de terceros

- **Licencia del código.** El repositorio todavía no incluye un archivo `LICENSE`: la licencia está pendiente de decisión del titular. Hasta que exista, no se concede ningún permiso de uso o redistribución más allá de los que otorgue la plataforma de alojamiento.
- **Datos de terceros que no se redistribuyen.** La capa catastral de Cali, con la que se evaluó el modelo, y los mapas base municipales son datos de terceros y **no forman parte de este repositorio**; aquí solo se publican cifras agregadas, ejemplos sintéticos y un manifiesto (`data/MANIFEST.json`). Cada usuario debe verificar los términos de uso y de redistribución de las fuentes que emplee.
- **Sin pesos entrenados.** Este modelo es un algoritmo determinista y no incluye pesos ni archivos derivados de datos de terceros.

## Licencia

Licencia: por definir (ver LICENSE en la raíz cuando exista).
