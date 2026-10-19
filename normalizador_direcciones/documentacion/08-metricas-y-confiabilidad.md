# 08. Métricas y confiabilidad

> **Los resultados de las corridas no se publican.** Este documento describe qué se mide, cómo se define cada métrica, el protocolo y los comandos exactos para reproducirlas **localmente** con datos propios. No contiene cifras medidas ni tablas de resultados, y es intencional: las salidas de evaluación y los reportes de ajuste se generan y se conservan fuera del repositorio.

## Contenido

1. [Qué se mide y cómo se define](#1-qué-se-mide-y-cómo-se-define)
2. [Protocolo](#2-protocolo)
3. [Cómo reproducirlo](#3-cómo-reproducirlo)
4. [Confiabilidad: Brier, ECE y AUC](#4-confiabilidad-brier-ece-y-auc)
5. [Escala operativa recomendada](#5-escala-operativa-recomendada)
6. [Lo que no se mide](#6-lo-que-no-se-mide)
7. [Salvedades metodológicas](#7-salvedades-metodológicas)

## 1. Qué se mide y cómo se define

| Métrica | Definición |
|---|---|
| Recall@k de validación sintética | Fracción de consultas con ruido sintético cuyo documento correcto aparece entre los `k` primeros candidatos (k = 1, 5, 20). Condición necesaria; no dice nada sobre texto real. |
| Tasa de `OK` | Fracción de filas evaluadas cuyo `estado` es `OK` en la ruta de producción (`normalize_strict`). |
| Acierto de manzana / predio | Entre las filas `OK` con referencia, fracción donde la `manzana` / el `numero_predial_nacional` devuelto coincide con el de la referencia. Se juzga por identidad catastral, no por coordenadas. |
| Subconjunto `gold` | Filas cuyo texto concuerda con la dirección catastral de la referencia. Las métricas sobre `gold` son cotas optimistas (sesgo de selección). |
| Distancia | Mediana, P90 y fracción dentro de 100 m entre la coordenada devuelta y la de referencia; solo informativa, nunca criterio de corrección. |
| Precisión de alta confianza | Acierto del predio entre las filas por encima del umbral de `confianza`. |
| Calibración de `confianza` | Acierto del predio por banda de `confianza` (por ejemplo 0.9-1.0); `confianza` no es una probabilidad. |
| Brier, ECE, AUC | Error cuadrático medio de la probabilidad, error de calibración esperado y área bajo la curva ROC de la `confiabilidad` frente al acierto, con intervalos bootstrap al 95 %. |
| Intervalos de Wilson | Intervalo binomial al 95 % para `p = k / n`; use siempre `n` junto a cada proporción. |

## 2. Protocolo

- Las particiones (ajuste, desarrollo, prueba congelada) se construyen por documento, sin direcciones repetidas entre particiones (`scripts/build_splits.py`, ver [docs/splits.md](../docs/splits.md)).
- La prueba congelada se usa una sola vez por decisión; el desarrollo sirve para ajustar pesos, umbral y confiabilidad y por tanto no es independiente.
- La referencia es la parcela que contiene el punto GPS registrado. Es **ruidosa**: la separación típica entre predios vecinos es comparable al error del punto, por lo que la referencia suele ser un vecino. El ruido afecta más a las métricas de predio que a las de manzana.
- Las cifras de una corrida solo son comparables con otra si la huella del código y los artefactos son los mismos.

## 3. Cómo reproducirlo

Las filas de evaluación son privadas y no se incluyen; use datos propios y deje las salidas fuera del control de versiones. Ver [04](04-entrenamiento.md) para el detalle de cada paso.

```bash
# Ruta de producción (normalize_strict) contra la referencia
PYTHONPATH=src python scripts/eval_strict.py --tag baseline
PYTHONPATH=src python scripts/eval_strict.py --limit 50 --tag smoke

# Pesos y umbral de la capa de reglas
PYTHONPATH=src python scripts/tune_threshold.py

# Calibrador de confiabilidad (v4 por defecto; v5 sobre etiquetas consistentes con el texto)
PYTHONPATH=src python scripts/fit_reliability.py --rows <filas_de_evaluacion.parquet>
PYTHONPATH=src python scripts/fit_reliability.py --kind logistic --gold-only --rows <filas_dev> <filas_train_pairs> --out artifacts/reliability_v2.json

# Experimentos de reentrenamiento
PYTHONPATH=src python scripts/run_experiment.py --tag baseline_prod --eval-only --from-dir artifacts
```

## 4. Confiabilidad: Brier, ECE y AUC

`artifacts/reliability.json` (v4) es el calibrador que se sirve y `artifacts/reliability_v2.json` (v5) es la alternativa ajustada solo con etiquetas consistentes con el texto (ver [03](03-configuracion.md) y [04](04-entrenamiento.md)). Los parámetros de cada calibrador se leen desde el JSON; los resultados de su ajuste (métricas de validación cruzada, tamaños de muestra, fechas) se regeneran localmente con `scripts/fit_reliability.py`.

Puntos de método que conviene conservar:

- **Procedencia.** Antes de citar una métrica de confiabilidad, verifique con qué filas y con qué versión del código se ajustó y vuelva a medirla con el código actual.
- **Ruido de etiqueta, no falta de señal.** Si la etiqueta de referencia es ruidosa (GPS), un calibrador ajustado sobre ella puede no discriminar aunque la señal exista. Compare el ajuste con etiquetas consistentes con el texto (`--gold-only`).
- **Mida el AUC sobre filas no vistas** (prueba congelada) y compárelo con el obtenido sobre las filas del ajuste.
- **No mezcle escalas.** Los cortes de `confiabilidad` del v4 y del v5 no son intercambiables.

## 5. Escala operativa recomendada

**Esto es una recomendación de diseño. No es una garantía, no está validada fuera de Cali ni sobre otras fuentes, y no sustituye la verificación humana en decisiones jurídicas, catastrales o tributarias.**

| Categoría | Criterio | Uso sugerido |
|-----------|----------|--------------|
| **Aceptable con muestreo de control** | `estado == OK`, `nivel_precision == predio`, `confiabilidad` por encima del corte configurado, sin notas en `motivo` que indiquen aproximación, y `margen` nulo o por encima del mínimo configurado | Análisis agregados, enriquecimiento de bases, priorización. Auditar una muestra aleatoria (con tamaño según la sección 4 de [10](10-estandar-y-generalizacion.md)) antes de usar el NPN individual. |
| **Requiere revisión humana** | Cualquier otra fila `OK`: nivel `manzana`, `esquina`, `via` o `direccion`; `confiabilidad` por debajo del corte; notas `aproximado`, `compartida` o `NPN`; `margen` pequeño; el `SIN_MATCH` con `regla: sufijo unilateral` | Revisar caso a caso o aceptar solo la manzana para agregación espacial gruesa. |
| **Rechazar (no usar el resultado)** | `NO_PARSEABLE`, `FUERA_DE_AREA`, `ERROR`, y todo `SIN_MATCH` salvo la excepción anterior; direcciones de otros municipios o rurales; cualquier resultado cuando la fuente no es de Cali | Corregir la fuente o buscar por otro medio. |

Los cortes de `confiabilidad` y `margen` se eligen con su propia muestra de desarrollo (por ejemplo, con la meta "precisión de predio mayor o igual a un valor objetivo") y se confirman una sola vez sobre una prueba congelada. Un corte válido para un calibrador no vale para otro.

Reglas de uso complementarias:

- **No usar `confianza` como porcentaje** ni como criterio de aceptación individual.
- **No validar con coordenadas.** Un punto cercano no confirma el predio, y el GPS de referencia tiene un error comparable a la separación entre predios vecinos.
- La primera categoría **no** equivale a "correcto": una fracción de las filas que cumplen todo puede ser un lote distinto, y más si la referencia real es más estricta que la capa SIG.
- Cualquier uso fuera de Cali exige el protocolo de validación de [10-estandar-y-generalizacion.md](10-estandar-y-generalizacion.md), sección 4.

## 6. Lo que no se mide

| No medido | Comentario |
|-----------|------------|
| Otras ciudades o municipios | El sistema está ajustado a Cali. |
| Otras fuentes de texto | No hay evaluación en texto de formularios web, dictado, reconocimiento óptico de caracteres, etc. |
| Zona rural | El índice es urbano (capa catastral urbana). |
| Direcciones muy largas o con caracteres no ASCII | El efecto no se mide por separado. |
| Concurrencia y carga de la API | Solo se mide un trabajador en CPU. |
| Estabilidad temporal | No se mide el desempeño al actualizar el catastro. |
| Sensibilidad al orden de los campos de unidades y a entradas adversariales | No se mide. |
| Generalización del modelo de confiabilidad | Sin medición en filas con texto inconsistente ni con referencia SIG. |

## 7. Salvedades metodológicas

1. **Tamaños pequeños.** Con subconjuntos pequeños una fila equivale a varios puntos de cobertura; trate como ruido diferencias de una o dos filas y reporte siempre intervalos.
2. **Sesgo del subconjunto `gold`.** Se selecciona por concordancia textual con la dirección catastral de la referencia; sus precisiones son cotas superiores para filas fáciles.
3. **Ruido de etiqueta por GPS.** Ver la sección 2.
4. **Constantes de la capa de reglas ajustadas con los mismos datos que se usan para informar.** `tuning.json` declara `evaluation_rows_used: none` para pesos y umbral, y es cierto para ellos. El resto de las constantes (holgura de barrio con escalamiento, `min_struct`, `ambiguity_delta`, reglas duras y blandas, corte de `confiabilidad`, ventana de placa) se fijó mirando referencias que luego se usan para informar. Los números finales no son una estimación fuera de muestra de esas decisiones.
5. **Dependencia de la fuente.** Una prueba congelada de una sola fuente no representa otras fuentes.
6. **El desarrollo no es independiente**: se usa para ajustar la confiabilidad y las reglas.
7. **Integración continua.** El flujo de CI ejecuta las pruebas con modelos de prueba; no ejecuta ninguna prueba de calidad del modelo real (las que necesitan los artefactos se omiten).
8. **Reproducibilidad.** Las filas de evaluación son privadas; este repositorio no permite reproducir una corrida sin datos propios.
