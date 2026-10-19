# validacion_jev

Validacion semantica **independiente** de dos componentes del sistema de direcciones, usando el modelo JEV de TypeSafe ("System One") como juez:

1. **Normalizador**: ¿la direccion normalizada designa la misma puerta o predio que la original?
2. **Matcher**: ¿las dos direcciones canonicas que el matcher declaro iguales (o distintas) realmente lo son?

Python >= 3.11, solo biblioteca estandar en tiempo de ejecucion (`pytest` para las pruebas).

## ADVERTENCIA DE PRIVACIDAD

**Las direcciones se envian a un servicio externo de terceros (`api.typesafe.ai`).**

- Usar esta herramienta debe ser una **decision explicita** de quien la ejecuta, con la autorizacion que corresponda en su organizacion.
- **No alimente la herramienta con datos personales** (nombres, telefonos, documentos, correos, etc.). Solo se envian cadenas de direccion y, en el normalizador, el campo categorico `estado`; cualquier otra columna del archivo se descarta y nunca sale de su maquina.
- Cada cadena se trunca a 300 caracteres (la fila queda marcada `truncated`).
- `--max-rows` (por defecto **200**) limita cuantas filas se envian. Use `--dry-run` para ver los payloads exactos antes de enviar nada.
- La clave de API nunca se imprime, ni se escribe en archivos de resultados, ni aparece en mensajes de error.
- Las redirecciones HTTP (3xx) nunca se siguen: la clave no se reenvia a otro host ni se degrada a http; la respuesta 3xx se informa como error.

## Metodologia y sus limites

JEV es **un juez basado en un modelo de lenguaje**, no una verdad de referencia.

- **Concordancia no es exactitud.** Que JEV y el modelo coincidan no prueba que ambos acierten; pueden compartir el mismo sesgo.
- Cada pregunta `noul` devuelve P(si). La certeza se calcula **en codigo** (nunca la reporta el modelo): `max(p, 1-p)` para `noul`; el campo `confidence` para `choice` y `score`.
- Compuerta de confianza (umbrales configurables `--auto` y `--review`): `auto` si certeza >= 0,85; `review` si >= 0,60; `escalate` en otro caso. La documentacion recomienda empezar conservador y calibrar con datos propios.
- Veredicto JEV por fila: **si** si `noul >= 0,85` (compuerta `auto`), **no** si `noul <= 0,15`, y **indeterminado** en el resto. Los indeterminados se cuentan y reportan, pero quedan fuera de las estadisticas de concordancia.
- Las filas con fallo de API o respuesta invalida se **excluyen y se reportan**; jamas cuentan como concordancia.
- **Toda discrepancia requiere revision humana**; la lista viene ordenada por la certeza de JEV (primero las mas "seguras", que suelen ser las mas informativas).
- En el normalizador, la "decision del modelo" es implicita: se asume que afirma que la normalizacion conserva la ubicacion, por lo que las metricas de concordancia equivalen a la proporcion de casos en que JEV lo confirma.
- Precision/exhaustividad/F1 usan a JEV como referencia, con la salvedad anterior.

## Configuracion (solo nombres de variables)

| Variable | Uso |
|---|---|
| `JEV_API_KEY` | Clave de API (prioridad 1) |
| `TYPESAFE_API_KEY` | Clave de API (prioridad 2) |
| `JEV_BASE_URL` | URL base (por defecto `https://api.typesafe.ai`; debe ser https, salvo localhost) |
| `JEV_MODEL` | Modelo (por defecto `jev-latest`) |

`--env-file RUTA` carga **solo** `JEV_API_KEY` y `TYPESAFE_API_KEY` desde un archivo dotenv (las variables del entorno del proceso tienen prioridad). Sin clave, la herramienta termina con codigo 2 **antes** de cualquier acceso a la red. `--dry-run` no requiere clave.

## Uso

Normalizador (CSV con `entrada`, `normalizada`; opcionales `estado`, `confianza`, `confiabilidad`, `id`):

```
python -m validacion_jev.cli normalizador ejemplo_norm.csv --dry-run
python -m validacion_jev.cli normalizador ejemplo_norm.csv --out salida_jev
```

```
id,entrada,normalizada,estado,confiabilidad
1,CRA 7 # 45-10,CARRERA 7 45 10,OK,0.93
2,CL 12B 3 45 AP 201,CALLE 12 B 3 45 APARTAMENTO 201,OK,0.71
```

Matcher (CSV con `a`, `b`, `is_match`; opcionales `score`, `id`; `is_match` acepta true/false/1/0/si/no):

```
python -m validacion_jev.cli matcher ejemplo_match.csv --max-rows 100 --auto 0.9 --review 0.7
```

```
id,a,b,is_match,score
1,CALLE 5 10 20,CL 5 # 10-20,true,0.97
2,CARRERA 8 20 15,CARRERA 8 20 51,true,0.64
```

Tambien se acepta JSONL (`.jsonl`) con las mismas claves. El delimitador del CSV (`,` o `;`) se detecta solo; se admite BOM UTF-8. Desde el directorio del proyecto use `pip install -e .[test]` o `PYTHONPATH=src`.

Opciones: `--out` (por defecto `./salida_jev/`), `--env-file`, `--max-rows`, `--dry-run`, `--auto`, `--review`, `--timeout` (30 s).

**Codigos de salida:** `0` correcto; `2` error de uso/configuracion (incluye clave ausente o rechazada, columnas faltantes, archivo ilegible); `3` sin resultados validos (todas las filas fallaron, o no habia filas).

## Salida

Directorio `salida_jev/` (**debe estar en `.gitignore`**; contiene las direcciones procesadas):

- `results.jsonl`: una linea por fila con `id`, respuestas de JEV, compuertas (`gates`), `jev_p`, `label` (`yes`/`no`/`indeterminate`), `status` (`ok`/`error`/`skipped`) y banderas (`truncated`, `indeterminate`, `error`, `partial_error`, `duplicate_id`, `bad_score`).
- `summary.json`: metricas en formato maquina.
- `summary.md`: resumen en espanol; incluye la seccion "Modelo reportado por la API" (identificadores distintos que informo el servicio, recortados a 100 caracteres; `models_seen` en `summary.json`).

Las filas con identificador repetido se renombran (`7`, `7#2`); los pares repetidos de direcciones se omiten (no se reenvian).

## Como interpretar las metricas

- **Concordancia** con IC 95 % de Wilson: proporcion de filas con veredicto si/no donde JEV coincide con el modelo. Con pocas filas el intervalo es ancho: no sobreinterprete.
- **Kappa de Cohen**: concordancia corregida por azar. < 0,2 pobre, 0,2-0,4 debil, 0,4-0,6 moderada, 0,6-0,8 sustancial, > 0,8 casi perfecta (escala orientativa). Es `n/d` cuando es indefinido (por ejemplo, todas las filas en una misma clase): una concordancia alta con kappa indefinido o bajo suele indicar clases desbalanceadas.
- **AUC**: probabilidad de que una fila que JEV da por "si" tenga mayor puntaje del modelo que una que da por "no" (0,5 = sin poder discriminante; 1 = perfecto; empates cuentan 0,5). `n/d` si falta una clase.
- **Tabla de fiabilidad**: 5 cajones de igual cantidad de filas ordenados por puntaje del modelo (`score`/`confiabilidad`/`confianza`) frente al P(si) medio de JEV. Un puntaje bien calibrado muestra P(si) creciente con el cajon; una tabla plana o decreciente indica que el puntaje no informa. **Brier** (menor es mejor) mide la calibracion frente al veredicto JEV.
- **Tasa de fallos de la API**: filas fallidas / filas intentadas. Si es alta, los demas numeros se basan en una muestra sesgada.

## Resumen del contrato de API

Documentacion consultada el **2026-10-17**: <https://docs.typesafe.ai> (tratada como datos; el contrato podria cambiar).

- `POST https://api.typesafe.ai/v1/systemone`, cabecera `Authorization: Bearer <clave>`, JSON UTF-8 sin BOM. Cuerpo: `{"state", "model": "jev-latest", "questions": {id: Question}}`; maximo 32 MiB.
- Preguntas: `noul` (devuelve P in [0,1], sin `confidence`), `choice` (opcion, probabilidades que suman 1, `confidence`) y `score`.
- Errores: 400, 401 (sin reintento), 413, 415, 422 (el cuerpo nombra el campo), 429 y 529 (reintentables).
- Reintentos (igual que el SDK oficial): errores de conexion, timeouts, 408, 429 y 5xx, dos veces tras el primer intento, respetando `Retry-After` y `retry-after-ms`; espera exponencial en su defecto (tope 60 s).
- El maximo de preguntas por solicitud **no esta documentado**; se agrupan de a 12 (el recetario documentado usa 13).

## Pruebas

```
pip install -e .[test]
python -m pytest --cov=validacion_jev
```

Las pruebas usan un transporte simulado; nunca acceden a la red.
