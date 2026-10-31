# 12. Guía para elegir métricas y métodos de validación

Guía breve para decidir **qué medir, con qué método y cómo leerlo** cuando se entrena o se adapta el normalizador, en Cali o en otra ciudad. Complementa, sin repetirlos, a [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md) (definiciones y protocolo de Cali) y a la sección 4 de [10-estandar-y-generalizacion.md](10-estandar-y-generalizacion.md) (tamaños de muestra y puertas de aceptación para un territorio nuevo). Este documento es de método: no contiene resultados medidos. Las cifras medidas de la ciudad sintética están en [11-entrenar-para-cualquier-ciudad.md](11-entrenar-para-cualquier-ciudad.md), sección 5.

> **Código del proyecto de origen, aún no publicado en este espejo.** Los elementos marcados con † en este documento (scripts y módulos como `build_index.py`†, `index_union.py`†, `calibration_report.py`†, `final_evidence.py`†, `retrain_compare.py`†, `jev_check.py`†, `fit_decision.py`†, `index_fallback.py`†, `compare_runs.py`†, `write_retrain_report.py`†, `monitor_signals.py`†, `download_nomenclatura.py`†, `clean_nomenclatura.py`†, `reproject_nomenclatura.py`†, `export_onnx.py`†, y los documentos `docs/onnx.md`† y `docs/retrain_2026-10_*.md`) pertenecen al proyecto de origen y **no están en `normalizador_direcciones/` de este repositorio**; no se pueden ejecutar desde aquí. `decision.py`†, `output_schema.py`†, `onnx_backend.py`† y `tokenizer.py`† solo se publican como copia de servicio en [`normalizador_produccion/src/cali_address/`](../../normalizador_produccion/src/cali_address/), no en `normalizador_direcciones/src/`. Las cifras y los comportamientos descritos se tomaron del proyecto de origen y no se pueden reproducir con solo el contenido de este espejo.

## Contenido

1. [Qué pregunta responde cada métrica](#1-qué-pregunta-responde-cada-métrica)
2. [Métricas: definición, cuándo aplica y cómo leerla](#2-métricas-definición-cuándo-aplica-y-cómo-leerla)
3. [Intervalos de confianza](#3-intervalos-de-confianza)
4. [Métodos de validación](#4-métodos-de-validación)
5. [Qué medir según la situación](#5-qué-medir-según-la-situación)
6. [Errores de lectura frecuentes](#6-errores-de-lectura-frecuentes)

## 1. Qué pregunta responde cada métrica

| Pregunta | Métrica | Herramienta implementada |
|----------|---------|--------------------------|
| ¿El codificador aprendió algo? | recall@k del codificador frente al azar (`k / N`) y frente al mismo modelo sin entrenar | `train.recall_at_k` (validación por época en `train_report.json`); `city_eval.encoder_recall` y `scripts/eval_city.py` |
| ¿El candidato correcto llega a la lista que se reordena? | recall@k de la ruta completa (codificador más reordenamiento) | `city_eval.pipeline_metrics`; `scripts/eval_strict.py` (`raw_top1_correct_manzana`) |
| ¿Cuántas filas recibe una respuesta? | Cobertura: filas `OK` (o `matched`) sobre filas evaluadas | `scripts/eval_strict.py`, `final_evidence.run_metrics`†, `city_eval.pipeline_metrics` |
| Cuando responde, ¿acierta? | Precisión de predio (lote) y de manzana sobre filas emitidas | `evaluate.predial_lot`, `evaluate.lots_match`, `final_evidence.run_metrics`†, `city_eval.pipeline_metrics` |
| ¿Cuánto acierto se descarta por prudencia? | Correctas perdidas: filas no emitidas cuyo primer candidato era correcto | `final_evidence`† (`lost_correct`), `city_eval.pipeline_metrics` (`lost_correct`) |
| ¿El número de confianza significa lo que dice? | Calibración: Brier, ECE y AUC | `scripts/fit_reliability.py` (`ece`, `metrics`), `scripts/fit_decision.py`†, `city_eval.ece` |
| ¿Sirven las bandas para decidir revisión humana? | Precisión por banda de `confiabilidad` o de `confianza` | `calibration_report.band_breakdown`†, `final_evidence.band_table`†, `city_eval.pipeline_metrics` (`bands`) |
| ¿La diferencia entre dos modelos es real? | Diferencia pareada con intervalo bootstrap | `final_evidence.paired_bootstrap`†, `city_eval.paired_bootstrap` |
| ¿Qué tan precisa es una proporción con este tamaño de muestra? | Intervalo de Wilson | `calibration_report.wilson_interval`†, `final_evidence.wilson`† |

## 2. Métricas: definición, cuándo aplica y cómo leerla

### 2.1 recall@k

- **Definición.** Proporción de consultas cuyo documento correcto aparece entre los `k` primeros candidatos. Se reporta para `k` = 1, 5 y 20 (20 es el tamaño de la lista que se reordena).
- **Cuándo aplica.** Para diagnosticar la recuperación (el codificador) y para comparar entrenamientos. No mide la salida que recibe el usuario: una fila puede tener el correcto en el puesto 1 y aun así no emitirse por las reglas.
- **Cómo leerla.** Compárela siempre con dos referencias: el azar (`k / N`, con `N` documentos del índice) y el mismo modelo sin entrenar. Un recall cercano al azar indica que el modelo no aprendió; un recall alto **con consultas sintéticas** indica que el flujo funciona, no que el modelo sirva con texto real. recall@20 bajo limita todo lo demás: lo que no entra en la lista no se puede recuperar después.

### 2.2 Cobertura

- **Definición.** Filas emitidas (`estado == "OK"` en la ruta de producción, `matched` en `normalize_batch`) sobre filas evaluadas. Conviene informarla también sobre filas analizables.
- **Cuándo aplica.** Siempre, junto con la precisión. Las dos se mueven en sentidos opuestos al cambiar el umbral o las reglas.
- **Cómo leerla.** Una cobertura baja con precisión alta es un sistema prudente; lo contrario, uno arriesgado. Ninguna de las dos se informa sola.

### 2.3 Precisión de predio (lote) y de manzana

- **Definición.** Entre las filas emitidas con referencia, proporción cuyo predio coincide con la referencia. En Cali la coincidencia de predio se mide por **lote** (los 21 primeros dígitos del NPN, `evaluate.predial_lot`), porque varias unidades de un mismo lote comparten dirección; la de manzana por los 17 primeros. En otra ciudad use el nivel equivalente de su identificador.
- **Cuándo aplica.** Es la métrica principal de uso: responde "cuando el sistema contesta, ¿acierta?".
- **Cómo leerla.** Infórmela con su intervalo de Wilson (sección 3), por `nivel_precision` y por estrato (fuente, zona). Con referencia derivada del mismo texto la cifra se infla; véase la sección 4.

### 2.4 Correctas perdidas

- **Definición.** Filas no emitidas cuyo primer candidato era correcto.
- **Cuándo aplica.** Al ajustar el umbral o las reglas: indica cuánto acierto se sacrifica por prudencia.
- **Cómo leerla.** Si es grande frente a los errores emitidos, el umbral o una regla es más estricto de lo necesario para ese uso.

### 2.5 Calibración: Brier, ECE y AUC

- **Definiciones.** Brier es el error cuadrático medio entre la probabilidad declarada y el acierto (0 a 1). ECE (error de calibración esperado) agrupa las filas en intervalos de igual tamaño por probabilidad y promedia la distancia entre la probabilidad media y la tasa de acierto de cada grupo. AUC mide si la probabilidad **ordena** bien aciertos y errores (0.5 = azar).
- **Cuándo aplica.** A `confiabilidad` (probabilidad calibrada) y, como diagnóstico, a `confianza` (puntaje fusionado sin calibrar). Solo con filas reales con referencia; con consultas sintéticas la calibración describe el ruido sintético, no el uso.
- **Cómo leerla.** ECE baja y AUC cercana a 0.5 significan "bien calibrada en promedio pero inútil para ordenar". Con pocos cientos de filas la ECE es inestable; valide por fuente (dejando una fuente fuera, como `fit_reliability.py`). La `confianza` no es una probabilidad aunque esté entre 0 y 1.

### 2.6 Bandas de confianza o confiabilidad

- **Definición.** Precisión de las filas emitidas dentro de cada intervalo de la probabilidad (por ejemplo `alta`, `media`, `baja`).
- **Cuándo aplica.** Para fijar una política operativa: aceptar automáticamente la banda alta y enviar las demás a revisión.
- **Cómo leerla.** Cada banda debe tener su propio intervalo; una banda con pocas filas no permite concluir nada aunque su tasa puntual sea alta. Las bandas deben ser monótonas: si una banda inferior acierta más que una superior, la calibración no sirve para decidir.

### 2.7 Errores graves

- **Definición.** Proporción de filas emitidas cuyo punto queda a más de una distancia acordada (por ejemplo 100 m) del predio de referencia.
- **Cuándo aplica.** Cuando el uso es espacial (mapas, conteos por zona) y un error lejano cuesta más que uno vecino.

## 3. Intervalos de confianza

| Situación | Método | Herramienta |
|-----------|--------|-------------|
| Una proporción (precisión, cobertura, recall) | Intervalo de Wilson al 95 %: se comporta bien con proporciones cercanas a 0 o 1 y con muestras pequeñas, a diferencia del intervalo normal | `calibration_report.wilson_interval`†, `final_evidence.wilson`† |
| Diferencia entre dos sistemas evaluados sobre **las mismas filas** | Bootstrap pareado: se remuestrean filas conjuntamente y se toma el intervalo de percentiles de la diferencia | `final_evidence.paired_bootstrap`† (10 000 remuestreos por defecto), `city_eval.paired_bootstrap` |
| Diferencia entre conjuntos de filas distintos | Dos intervalos de Wilson por separado; no se pueden restar filas que no son las mismas | `calibration_report.wilson_interval`† |
| Variabilidad del entrenamiento | Repetir con varias semillas y comparar contra ese piso de ruido | `scripts/run_experiment.py`, `scripts/compare_runs.py`† |

Reglas de lectura:

- Decida con el **límite inferior** del intervalo, no con la estimación puntual (es la regla de las puertas de [10](10-estandar-y-generalizacion.md), sección 4.4).
- Un intervalo de diferencia que contiene el 0 no demuestra mejora ni empeoramiento.
- El bootstrap pareado mide la variabilidad del muestreo de filas, no la del entrenamiento; para esta última hacen falta varias semillas.

## 4. Métodos de validación

| Método | Qué valida | Cuándo usarlo | Límite |
|--------|------------|---------------|--------|
| Consultas sintéticas retenidas (variantes con ruido de documentos del índice, semilla nunca usada en el entrenamiento) | Que el flujo completo corre y que el modelo aprende más allá del azar | Prueba de humo de un territorio nuevo, regresiones del código (`scripts/eval_city.py`, `scripts/synthetic_city_e2e.py`) | El ruido sintético es el mismo generador del entrenamiento: no mide texto real |
| Partición por documento (`q_is_val`, 4 % de los documentos) | Generalización a direcciones que el modelo nunca vio como consulta | Seguimiento por época en `train_report.json` | Sigue siendo sintética |
| Partición por clave de dirección de las filas reales (ajuste, desarrollo y prueba final disjuntos) | Desempeño con texto real sin fuga entre conjuntos | Siempre que haya filas reales con referencia (`scripts/build_splits.py`, [docs/splits.md](../docs/splits.md)) | Requiere referencia independiente del texto |
| Validación dejando una fuente fuera | Que la calibración y los umbrales se transfieren a una fuente no vista | Ajuste de `reliability.json` (`scripts/fit_reliability.py`) | Pocas fuentes dan pocas particiones |
| Prueba final congelada | Estimación sin sesgo de selección del candidato elegido | Una sola vez, al final | Si se usa para elegir, deja de ser prueba final |
| Muestreo estratificado | Que ningún estrato relevante (fuente, zona, tipo de vía) queda oculto en el promedio | Diseño de la muestra de referencia | Exige `n` suficiente por estrato ([10](10-estandar-y-generalizacion.md), sección 4.2) |

Sobre la referencia: debe ser **independiente del texto de la dirección** (verificación en campo, trámite, punto GPS dentro del polígono del predio). Una referencia derivada del mismo texto que se evalúa infla la precisión. Registre la calidad de la referencia (distancia típica entre la referencia y el predio real) junto con los resultados.

## 5. Qué medir según la situación

| Situación | Mínimo a reportar | Método |
|-----------|-------------------|--------|
| Probar que el flujo funciona en una ciudad nueva (sin datos reales) | recall@1/5/20 del codificador entrenado frente al azar y frente al modelo sin entrenar; recall de la ruta completa; cobertura y precisión sobre consultas sintéticas | Consultas sintéticas retenidas; diferencia pareada entrenado menos sin entrenar con bootstrap |
| Elegir entre dos entrenamientos | recall@k de validación; cobertura, precisión de lote y de manzana y correctas perdidas en desarrollo | Bootstrap pareado sobre las mismas filas; varias semillas |
| Fijar el umbral | Curva cobertura frente a precisión y correctas perdidas | `scripts/tune_threshold.py` (sintético más negativos de no dirección), luego verificación en desarrollo real |
| Decidir si `confiabilidad` sirve para ordenar o filtrar | Brier, ECE y AUC fuera de muestra; precisión por banda con Wilson | Validación dejando una fuente fuera |
| Aceptar el sistema para un uso | Precisión de lote y de manzana con límite inferior de Wilson, por nivel y por estrato; errores graves; cobertura | Prueba final congelada sobre referencia independiente; puertas de [10](10-estandar-y-generalizacion.md), sección 4.4 |
| Vigilar el sistema en operación | Distribución de `estado` y de `nivel_precision`, señales de deriva | `calibration_report.monitoring_signals`†, `scripts/monitor_signals.py` |

## 6. Errores de lectura frecuentes

1. **Tomar el recall sintético como precisión real.** Las consultas sintéticas salen del mismo generador de ruido del entrenamiento; prueban el flujo, no la calidad con texto de usuarios.
2. **Informar precisión sin cobertura** (o al revés). Subir el umbral mejora una y empeora la otra.
3. **Leer `confianza` como probabilidad.** Solo `confiabilidad` está calibrada, y únicamente para los datos con que se ajustó.
4. **Concluir con muestras pequeñas.** Con decenas de filas por estrato el intervalo de Wilson es amplio; informe `n` y el intervalo.
5. **Comparar sistemas sobre filas distintas.** Una diferencia solo se puede atribuir al sistema si las filas son las mismas (bootstrap pareado).
6. **Reutilizar la prueba final.** Cada vez que se mira para decidir, pierde su valor como estimación sin sesgo.
7. **Olvidar las referencias triviales.** Sin el azar, el modelo sin entrenar y la ruta solo de reglas (`use_model=False`), no se sabe cuánto aporta el aprendizaje.
