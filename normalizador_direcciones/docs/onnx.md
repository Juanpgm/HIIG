# Backend ONNX Runtime (fp32 e int8) para el servicio

El normalizador puede servir las consultas con ONNX Runtime en lugar de torch. El motivo es el tamaño y el costo
de la imagen: torch CPU pesa 930 MB en la imagen actual (609 MB comprimida, 2,33 GB en disco) y es la dependencia
más grande, mientras que `onnxruntime` ocupa unas decenas de MB. **El backend por defecto sigue siendo `torch`** y
su comportamiento no cambia (la prueba de identidad byte a byte de `tests/test_output_validation_wiring.py` sigue
en verde). Los backends ONNX son opcionales y se activan con una variable de entorno.

Todos los datos de este documento son direcciones catastrales públicas o cadenas inventadas; ningún número sale de
`frozen_test` y ningún artefacto de producción se modificó (los archivos generados viven en
`artifacts/experiments/onnx_bench/`).

## Qué se exporta

El servicio necesita, por cada texto de consulta (siempre `role=0`):

1. la incrustación normalizada L2 (`AddressEncoder.embed`), y
2. la cabeza auxiliar de coordenadas (`coord_head(pool)`), que usa el reordenamiento geográfico.

Ambas salen de **un único grafo** `ids -> emb, xy` (`ids` int64 `[batch, 40]`, eje `batch` dinámico, opset 17).
La tokenización de caracteres vive ahora en `cali_address/tokenizer.py` (solo numpy); `cali_address.model` la
reexporta, así que los imports existentes siguen funcionando.

Los embeddings de documentos (`catastro_emb.pt`, un pickle de torch) se convierten una vez a `catastro_emb.npy`
(fp16, misma matriz) para que el servicio ONNX no importe torch. La búsqueda coseno top-k corre en numpy
(`NumpyDocIndex`): producto matricial por bloques sobre los documentos y `argpartition`; después se recalcula el
coseno exacto en fp32 de cada candidato, igual que `AddressNormalizer.search_exact`.

## Generar los archivos

```bash
pip install -e ".[onnx]"        # onnx>=1.16 (solo para exportar) y onnxruntime>=1.18

# Escribe model.onnx, model.int8.onnx, onnx_meta.json y catastro_emb.npy junto a model.pt
PYTHONPATH=src python scripts/export_onnx.py --artifacts-dir artifacts            # in situ
PYTHONPATH=src python scripts/export_onnx.py --artifacts-dir artifacts --out-dir <otro directorio>
```

- `export_onnx` (`cali_address.onnx_backend`) desactiva el "fast path" fusionado de `nn.TransformerEncoderLayer`
  durante la exportación (no tiene símbolo ONNX) y deja el grafo operación por operación, numéricamente equivalente.
  `onnx_meta.json` guarda `max_len`, `d_out`, el escalado de la coordenada y el sha256 de `model.pt`; el servicio
  rechaza una exportación vieja ("stale") si `model.pt` cambió después.
- `quantize_int8` usa `onnxruntime.quantization.quantize_dynamic` sobre `MatMul` y `Gemm` con pesos `qint8`
  (el valor por omisión de onnxruntime para cuantización dinámica; `--weight-type quint8` y `--per-channel`
  están disponibles). Las tablas de embeddings, LayerNorm y softmax quedan en fp32. Tamaño: 13,26 MB (fp32) a
  3,60 MB (int8).
- Un experimento que hereda `model.pt` de producción lee los archivos ONNX del directorio donde está ese `model.pt`.

## Elegir el backend

| Variable | Valores | Efecto |
|---|---|---|
| `NORMALIZER_BACKEND` | `torch` (defecto), `onnx`, `onnx-int8` | Backend del `AddressNormalizer` (también el parámetro `backend=`, que tiene prioridad). |
| `NORMALIZER_ONNX_THREADS` | entero | Hilos intra-op de la sesión ONNX (0/vacío: decide onnxruntime). |
| `NORMALIZER_ONNX_INDEX` | `fp32` (defecto), `fp16` | Precisión de la matriz de documentos en memoria (ver abajo). |

```bash
NORMALIZER_BACKEND=onnx-int8 uvicorn cali_address.api.main:app
NORMALIZER_BACKEND=onnx python scripts/eval_strict.py --device cpu ...
```

Comportamiento ante fallas:

- Archivo `.onnx` o `catastro_emb.npy` ausente: `FileNotFoundError` con el nombre del archivo y el comando para
  crearlo. No hay degradación silenciosa a torch.
- `onnxruntime` no instalado: `OnnxUnavailableError` (mensaje con `pip install 'cali-address[onnx]'`); el backend
  torch no se ve afectado.
- torch no instalado: los backends ONNX funcionan (torch solo se importa si se pide el backend `torch`, y en ese
  caso falla con el `ImportError` original). Hay una prueba con la importación de torch bloqueada y otra que
  comprueba que `torch` no entra a `sys.modules` con `onnx` aunque esté instalado.
- `device="cuda"` con un backend ONNX se ignora (proveedor CPU), y el dispositivo queda en `cpu`.
- Valor desconocido de `NORMALIZER_BACKEND`: `ValueError` antes de cargar nada.

Decisiones de implementación con efecto medido:

- **Lote del codificador ONNX = 256** (`DEFAULT_ENCODE_BATCH`). Con lotes de 1.024 filas, onnxruntime conserva su
  arena de activaciones al tamaño del mayor lote y el pico de memoria fue de 2,4 a 3,3 GB en la primera pasada del
  benchmark; con 256 filas el pico del codificador baja de 1,1-1,4 GB a 0,3-0,4 GB y además es más rápido.
- **Matriz de documentos en fp32 por defecto**. Convertir fp16 a fp32 en cada búsqueda costaba unos 0,15 s sobre
  330 mil documentos y dominaba la latencia de una dirección (400 ms por consulta). Guardar fp32 en lugar de fp16
  duplica esa matriz (169 MB a 338 MB en producción) pero el resultado es idéntico (la conversión es exacta).
  `NORMALIZER_ONNX_INDEX=fp16` reduce la memoria a cambio de latencia.
- Las dos vistas de la consulta (texto canónico y texto pre-normalizado) se buscan en una sola pasada sobre la
  matriz de documentos, porque el recorrido está limitado por ancho de banda de memoria.

## Paridad con torch

`scripts/onnx_parity.py` compara, en CPU, la incrustación de consulta, el top-1 y la coordenada predicha, sobre
10 cadenas inventadas y 2.000 cadenas públicas de `catastro_docs.parquet` (`direccion`, semilla 0).

| Conjunto | Backend | Coseno mín. | Coseno medio | Top-1 idéntico | Top-1 misma dirección | Dif. máx. coordenada |
|---|---|---|---|---|---|---|
| Producción (330.387 docs) | onnx fp32 | 1,000000 | 1,000000 | 100 % | 100 % | 0,014 m |
| Producción | onnx-int8 | 0,999618 | 0,999840 | 100 % | 100 % | 138,8 m |
| Candidato `v2d_cert_pcer_s42` (395 mil docs, medición previa) | onnx fp32 | 1,000000 | 1,000000 | 99,75 % (5 de 2.000) | 100 % | 0,008 m |
| Candidato `v2d_cert_pcer_s42` (medición previa) | onnx-int8 | 0,999588 | 0,999828 | 99,75 % | 100 % | 149,1 m |
| **Candidato final `v2g_cert_pcer_catfb_s42_r3_basecal` (401.860 docs)** | onnx fp32 | 1,000000 (0,99999988) | 1,000000 | 99,95 % (1 de 2.000) | 99,95 % (1 de 2.000) | 0,007 m |
| **Candidato final** | onnx-int8 | 0,999500 | 0,999821 | 100 % | 100 % | 231,8 m |

(Valores de las 2.000 cadenas públicas; los de las cadenas inventadas están en
`artifacts/experiments/onnx_bench/parity_*.json` y, para la pasada final,
`artifacts/experiments/onnx_bench_final/parity_{prod,cand}_serving.json`. Las filas de producción de arriba se
repitieron en la pasada final con resultados idénticos salvo el coseno mínimo fp32, 0,99999988.) En el candidato final
el único desacuerdo de fp32 (1 de 2.000) es un empate entre filas duplicadas, no un error de incrustación (coseno
mínimo 0,99999988). Criterio pedido para fp32: coseno mínimo >= 0,9999 y top-1
idéntico >= 99,9 %. Se cumple en producción. En el candidato, 5 de 2.000 consultas eligen un documento distinto, pero
siempre uno con **la misma cadena de dirección** (el índice del candidato tiene filas duplicadas con incrustación
idéntica, y el desempate entre ellas difiere: torch decide los candidatos con una multiplicación fp16, ONNX con fp32).
Es un empate y no un desacuerdo: por eso se reporta también "misma dirección".

El int8 **no es invariante al lote**: la cuantización dinámica calcula la escala de activación sobre todo el tensor
del lote, así que la incrustación de una fila depende de las otras filas (coseno mínimo 0,99975 entre lote de 1 y
de 256 sobre 1.000 direcciones; fp32: 0,9999999). La diferencia de la coordenada (decenas a cientos de metros en el
peor caso) alimenta el factor geográfico del reordenamiento.

### Evaluación estricta en dev (CPU, índice de producción)

`scripts/eval_strict.py` sobre `artifacts/splits/dev.parquet` (nunca `frozen_test`), con los archivos de producción
copiados a `artifacts/experiments/onnx_bench/prod_serving/`; resultados en `eval_<backend>/`.

| Backend | Dev todas: OK | Manzana (precisión) | Predial exacto | Predial lote | Dev gold: OK | Manzana | Predial exacto | Predial lote |
|---|---|---|---|---|---|---|---|---|
| torch | 1.084 | 0,8630 | 0,6465 | 0,6566 | 278 | 0,9964 | 0,9880 | 0,9960 |
| onnx fp32 | 1.084 | 0,8630 | 0,6465 | 0,6566 | 278 | 0,9964 | 0,9880 | 0,9960 |
| onnx-int8 | 1.084 | 0,8630 | 0,6439 | 0,6540 | 278 | 0,9964 | 0,9840 | 0,9920 |

El fp32 reproduce a torch en las 2.078 filas de dev (estado y predial idénticos, 289 de 289 en gold). El int8
cambia 1 predial de 2.078 en dev (`-0,0026` en predial exacto y en predial lote; `-0,004` en gold, un registro de
289), sin cambios en manzana.

## Benchmark

`scripts/bench_backends.py` (CPU; cada backend en un proceso nuevo; hilos fijos; calentamiento de 2.000 consultas;
5 repeticiones; consultas sintéticas = cadenas públicas de documentos corrompidas con `augment.corrupt`, semilla 0;
todo pasa por `normalize_strict` con el gazetteer real). Equipo: AMD Ryzen 9 9950X (16 núcleos físicos, 32 lógicos),
Windows 11, Python 3.14. El pico de RSS es el pico del conjunto de trabajo del proceso.

Hay dos pasadas:

- **Pasada final (en silencio)**, la que vale: la máquina estaba libre (carga de fondo de unos 2 % de CPU, solo
  aplicaciones del usuario; ningún otro agente ni prueba corriendo mientras se medía). Datos en
  `artifacts/experiments/onnx_bench_final/bench_final.json` (agregado), `bench_threads16.json`, `bench_threads4.json` y
  `bench_cand_index_fp16.json`. La dispersión entre repeticiones es baja (el p50 de latencia varía menos de 1 % entre
  las 5 repeticiones; el tiempo del lote de 10.000 filas varía hasta 7 % en torch y hasta 2 % en ONNX salvo la primera
  repetición, que es ~10 % más lenta), así que diferencias de más de unos 3 % son reales en esta máquina.
- **Pasada preliminar (ruidosa)**, descrita al final de esta sección, se hizo con otro agente ejecutando pruebas y se
  conserva solo por trazabilidad.

Artefactos medidos: producción (330.387 documentos, sin capa de decisión calibrada) y el **candidato final**
`v2g_cert_pcer_catfb_s42_r3_basecal` (401.860 documentos, con `decision.json` y `reliability_calibrated.json`, de modo
que `normalize_strict` incluye la capa de decisión calibrada). Los archivos ONNX del candidato se generaron con
`scripts/export_onnx.py` en `artifacts/experiments/onnx_bench_final/cand_serving/` (copia del índice; el directorio
original del experimento no se tocó).

Con 16 hilos (núcleos físicos); latencia = mediana de las repeticiones del p50 / p95 sobre 500 consultas sueltas;
rendimiento = filas por segundo (mediana de 5 repeticiones) por tamaño de lote:

| Artefactos | Backend | Arranque en frío (s) | p50 (ms) | p95 (ms) | 100 filas/s | 2.000 filas/s | 10.000 filas/s | RSS pico (MB) |
|---|---|---|---|---|---|---|---|---|
| Producción | torch | 0,90 | 19,9 | 22,2 | 136 | 142 | 137 | 1.568 |
| Producción | onnx | 1,08 | 17,9 | 19,9 | 209 | 216 | 215 | 1.675 |
| Producción | onnx-int8 | 1,07 | 17,4 | 19,4 | 215 | 222 | 221 | 1.779 |
| Candidato final | torch | 1,09 | 23,5 | 25,9 | 116 | 120 | 120 | 1.827 |
| Candidato final | onnx | 1,28 | 21,5 | 23,8 | 179 | 182 | 184 | 1.962 |
| Candidato final | onnx-int8 | 1,28 | 20,9 | 23,3 | 186 | 190 | 190 | 2.077 |

Con 4 hilos (más parecido a un contenedor pequeño):

| Artefactos | Backend | Arranque en frío (s) | p50 (ms) | p95 (ms) | 100 filas/s | 2.000 filas/s | 10.000 filas/s | RSS pico (MB) |
|---|---|---|---|---|---|---|---|---|
| Producción | torch | 0,88 | 29,3 | 31,9 | 46 | 49 | 49 | 1.542 |
| Producción | onnx | 1,07 | 21,3 | 23,5 | 187 | 195 | 192 | 1.660 |
| Producción | onnx-int8 | 1,05 | 19,5 | 21,7 | 208 | 211 | 209 | 1.758 |
| Candidato final | torch | 1,10 | 35,1 | 38,3 | 41 | 40 | 40 | 1.788 |
| Candidato final | onnx | 1,29 | 25,8 | 28,7 | 155 | 155 | 155 | 1.926 |
| Candidato final | onnx-int8 | 1,29 | 24,9 | 27,3 | 175 | 176 | 176 | 2.040 |

Memoria con el índice en fp16 (`NORMALIZER_ONNX_INDEX=fp16`), candidato final, onnx fp32, 16 hilos:

| Índice en memoria | RSS pico (MB) | RSS tras arranque (MB) | p50 (ms) | p95 (ms) | 100 / 2.000 / 10.000 filas/s |
|---|---|---|---|---|---|
| fp32 (defecto) | 1.962 | 1.190 | 21,5 | 23,8 | 179 / 182 / 184 |
| fp16 | 1.743 | 1.001 | 181,7 | 197,6 | 141 / 165 / 165 |

Lectura:

- **Velocidad**: ONNX fp32 rinde 1,5 veces lo de torch con 16 hilos (1,52 a 1,57 en producción, 1,52 a 1,54 en el
  candidato) y entre 3,8 y 4,1 veces con 4 hilos (torch casi no escala con pocos hilos). En una dirección suelta la
  latencia p50 baja 10 % con 16 hilos (19,9 a 17,9 ms en producción; 23,5 a 21,5 ms en el candidato) y 27 % con 4 hilos
  (29,3 a 21,3 ms; 35,1 a 25,8 ms).
- **int8 frente a fp32**: con 16 hilos aporta solo 3 a 4 % de rendimiento y 3 % menos de latencia; con 4 hilos
  aporta 8 a 14 % de rendimiento y 3 a 8 % menos de latencia. Es una ganancia pequeña, y el int8 no es
  invariante al lote (ver más arriba). Su coste: +100 MB de RSS pico sobre el fp32 en esta medición (la sesión int8
  conserva además los pesos fp32 de las tablas de embeddings y LayerNorm).
- **Arranque en frío**: ONNX tarda 0,2 s más que torch en producción (1,08 frente a 0,90 s) y el mismo orden en el
  candidato; no es una ventaja de ONNX, y la diferencia es pequeña.
- **Memoria**: el RSS pico de ONNX fp32 es mayor que el de torch (+107 MB en producción, +135 MB en el candidato; +211 y
  +250 MB con int8), no menor. El proceso incluye gazetteer, tabla de documentos y cachés de análisis comunes a los
  tres, y ONNX guarda la matriz de documentos en fp32 (el doble que fp16). Con `NORMALIZER_ONNX_INDEX=fp16` el pico
  del candidato baja 219 MB (11 %: 1.962 a 1.743 MB) y el RSS tras el arranque 189 MB, lo que lo deja 84 MB **por
  debajo** del torch del candidato (1.827 MB), pero a cambio la latencia de una dirección se multiplica por 8,5
  (21,5 a 181,7 ms, por la conversión fp16 a fp32 en cada búsqueda) y el rendimiento cae 9 a 21 %. Solo conviene
  fp16 si la memoria manda sobre la latencia. No se midió RSS de contenedor.
- **Capa de decisión calibrada**: está incluida en las cifras del candidato final (es parte de `normalize_strict`
  cuando existe `decision.json`). Las diferencias candidato frente a producción (23,5 frente a 19,9 ms en torch)
  mezclan el índice más grande (401.860 frente a 330.387 documentos) y esa capa; no se separaron.
- Los números absolutos no son comparables con las mediciones anteriores en Docker (2.000 filas en 4 s): aquí la carga
  es corrompida y pasa por el gazetteer completo, en Windows. Solo la comparación relativa entre backends es válida.

### Pasada preliminar (ruidosa) y diferencias con la final

La primera tabla del benchmark (16 hilos: torch 104 a 114 filas/s, onnx 172 a 209, onnx-int8 202 a 224 en producción
y candidato `v2d_cert_pcer_s42`) se midió en una máquina compartida con otro agente ejecutando pruebas, por lo que las
diferencias de menos de 10 % no eran concluyentes. Sus datos siguen en `artifacts/experiments/onnx_bench/bench.json`
y `bench_threads4.json`; la primera pasada con lote de 1.024 está en `bench_first_pass_batch1024.json` (RSS de
2,4 a 3,3 GB para ONNX, que motivó el lote de 256). Lo que cambió al medir en silencio (producción, 16 hilos):

- torch pasó de 104 / 113 / 114 a **136 / 142 / 137** filas/s (+25 a +30 %): el ruido lo penalizaba más que a ONNX, y el
  arranque en frío de torch bajó de 2,60 a 0,90 s (la medición vieja incluía la carga en frío del disco).
- onnx fp32 pasó de 172 / 186 / 209 a **209 / 216 / 215**, y su p50 de 17,4 a 17,9 ms (sin cambio real).
- onnx-int8 pasó de 224 / 222 / 219 a **215 / 222 / 221**. La ventaja de hasta 30 % del int8 sobre el fp32 en lotes
  pequeños era ruido: queda en 3 %. El cociente ONNX/torch baja de 1,5 a 2,2 a **1,5 a 1,6** con 16 hilos; con 4
  hilos se mantiene (3,8 a 4,6 antes, 3,8 a 4,1 ahora).
- El RSS pico cambió poco (producción onnx 1.679 a 1.675 MB; torch 1.564 a 1.568 MB).

### Tamaño instalado y estimación de la imagen

Medido en el `site-packages` de `.venv` con `importlib.metadata` (suma de los archivos que instala cada
distribución, para la clausura de requisitos del paquete, restando la clausura de las dependencias base):

| Pila | Tamaño instalado |
|---|---|
| torch (+ filelock, fsspec, jinja2, markupsafe, mpmath, networkx, setuptools, sympy) | 517 MB (rueda CPU de Windows) |
| onnxruntime (+ flatbuffers, packaging, protobuf) | 46 MB |
| onnx + ml-dtypes (solo para exportar, no hace falta en servicio) | 22 MB |

Estimación de la imagen sin torch: `2,33 GB (imagen actual) - 0,93 GB (capa de torch medida con docker history) +
0,05 GB (pila onnxruntime medida aquí)` = **aprox. 1,45 GB en disco**. Los artefactos casi no cambian de tamaño
(`model.onnx` 13 MB e `model.int8.onnx` 3,6 MB reemplazan a `model.pt` 14 MB; `catastro_emb.npy` pesa lo mismo que
`catastro_emb.pt`; si se conserva `model.pt` para reentrenar se suma). Es una estimación: el valor real debe
confirmarse con `docker build`, y el tamaño comprimido no se estimó.

## Recomendación

Con la medición en silencio y la paridad del candidato final:

- **Defecto: `onnx` fp32.** Reproduce a torch (coseno mínimo 0,99999988; top-1 idéntico en 100 % en producción y en
  99,95 % en el candidato, donde el único desacuerdo es un empate entre filas duplicadas; diferencia de coordenada
  menor a 1 cm; sin cambios en la evaluación estricta de dev), rinde 1,5 veces lo de torch con 16 hilos y 3,8 a 4,1
  veces con 4 hilos, y la imagen pierde la capa de torch (estimación aprox. 1,45 GB en disco frente a 2,33 GB).
- **`onnx-int8` es opcional.** Solo gana 3 a 4 % de rendimiento con 16 hilos y 8 a 14 % con 4 hilos, usa unos 100 MB más
  de RSS, no es invariante al lote (el resultado de una fila depende de las demás filas del lote; coseno mínimo 0,9995
  contra torch y hasta 232 m de diferencia en la coordenada predicha) y cambió 1 predial de 2.078 en dev. Con esa
  ganancia no justifica el riesgo; habría que reevaluarlo sobre dev completo antes de adoptarlo.
- **`NORMALIZER_ONNX_INDEX=fp16` solo si la memoria manda**: ahorra unos 220 MB de pico (11 %) en el candidato de
  401.860 documentos, pero multiplica por 8,5 la latencia de una dirección y baja el rendimiento 9 a 21 %.
- **torch sigue disponible** como backend de referencia y es el único que puede regenerar `catastro_emb.pt`.

Limitaciones de la medición (honestas): una sola máquina (AMD Ryzen 9 9950X, Windows 11, Python 3.14); en un contenedor
Linux con otra CPU, otro asignador de memoria y cuota de CPU los números absolutos serán distintos y solo la
comparación relativa entre backends debe tomarse como válida; los 4 hilos se fijaron con un parámetro de hilos y no con
una cuota de CPU de contenedor; no se midió RSS de contenedor ni el tamaño comprimido de la imagen, y la estimación de
imagen sin torch debe confirmarse con `docker build`; las consultas son sintéticas (cadenas públicas corrompidas), no
tráfico real; la diferencia entre candidato y producción mezcla índice y capa de decisión.

## Propuesta de Dockerfile sin torch (solo propuesta, no aplicada)

El `Dockerfile` actual **no se modificó**. Para servir sin torch:

```dockerfile
# 1. Quitar la capa de torch:
#    RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
# 2. Instalar onnxruntime (no hace falta el paquete onnx, solo se usa para exportar):
RUN pip install --no-cache-dir "onnxruntime>=1.18"

ENV NORMALIZER_BACKEND=onnx \
    NORMALIZER_ONNX_THREADS=4

# 3. Copiar los archivos ONNX junto con los demás (los genera export_onnx.py; ver abajo)
COPY deploy/artifacts/ ./artifacts/

# 4. La verificación del build pasa a exigir los archivos ONNX en lugar de torch
RUN python -c "import cali_address.service, cali_address.api.main" \
    && test -f /app/artifacts/model.onnx \
    && test -f /app/artifacts/onnx_meta.json \
    && test -f /app/artifacts/catastro_emb.npy
```

Cambios asociados fuera del Dockerfile:

- `deploy/prepare_artifacts.py`: ejecutar `scripts/export_onnx.py` y copiar `model.onnx` / `model.int8.onnx`,
  `onnx_meta.json` y `catastro_emb.npy` a `deploy/artifacts/`. Con un backend ONNX no hacen falta `model.pt`
  ni `catastro_emb.pt` en la imagen (el sha256 de `model.pt` se verifica solo si el archivo está presente).
- `requirements.txt`: añadir `onnxruntime` (o instalar el extra `onnx`); el comentario sobre torch deja de aplicar
  a la imagen de servicio.
- `cali-address setup` (regenera `catastro_emb.pt`) sigue necesitando torch y no corre en la imagen sin torch.
- Antes de adoptar `onnx-int8` en producción conviene decidir si la pérdida medida (1 predial de 2.078 en dev, y la
  dependencia del lote) es aceptable; `onnx` fp32 reproduce a torch sin diferencias en dev y es la opción conservadora.

## Reproducción

Ver la sección "Backend ONNX opcional" de [model-artifacts.md](model-artifacts.md). Resumen:

```bash
PYTHONPATH=src python scripts/export_onnx.py --artifacts-dir <dir con model.pt> --out-dir <dir de servicio>
PYTHONPATH=src python scripts/onnx_parity.py --artifacts-dir <dir de servicio> --out artifacts/experiments/onnx_bench/parity.json
PYTHONPATH=src python scripts/bench_backends.py --set prod=<dir de servicio> --set cand=<otro dir de servicio>
```

El directorio de servicio debe contener `model.pt`, `catastro_emb.pt`, los archivos ONNX y, para un índice propio,
los archivos del índice (`catastro_docs.parquet`, `index_meta.json`, ...). Pruebas: `tests/test_onnx_backend.py`.
