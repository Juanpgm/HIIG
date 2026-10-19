# 08 - Métricas y confiabilidad

> **Los resultados de las corridas no se publican.** Este documento describe qué se mide, cómo se define cada métrica y los comandos exactos para reproducirlas **localmente** con sus propios datos. No contiene cifras medidas ni tablas de resultados, y es intencional: las salidas de evaluación se generan y se conservan fuera del repositorio.

**Lectura honesta en una frase:** las pruebas sobre el catastro propio son parcialmente circulares y sirven como invariantes y regresión; la confianza real exige muestras con etiquetas humanas y un intervalo de confianza que usted mismo calcule.

## Contenido

1. [Qué se mide y cómo se define](#1-qué-se-mide-y-cómo-se-define)
2. [Cómo reproducirlo](#2-cómo-reproducirlo)
3. [Lo que el arnés NO mide](#3-lo-que-el-arnés-no-mide)
4. [Evaluación crítica (limitaciones metodológicas)](#4-evaluación-crítica-limitaciones-metodológicas)
5. [Escala de nivel de confiabilidad (recomendación)](#5-escala-de-nivel-de-confiabilidad-recomendación)

Relacionados: [04 evaluación](04-evaluacion-y-calibracion.md), [06 interpretación](06-predicciones.md), [09 advertencias](09-advertencias-y-limitaciones.md).

## 1. Qué se mide y cómo se define

| Métrica | Definición | Herramienta |
|---|---|---|
| Cobertura de lectura | Fracción de filas cuya dirección el parser lee (`parse_ok`); se clasifican las causas de fallo | `tools/eval_catastro.py`, sección A |
| Autocoincidencia y simetría | `match(a, a)` debe dar 1,0 y `match(a, b) == match(b, a)`; se cuentan las violaciones | `eval_catastro.py`, sección B |
| Recall de variantes de formato | Fracción de variantes de formato (positivos por construcción) con puntaje mayor o igual al umbral | `eval_catastro.py`, sección C |
| Rechazo de diferencias reales | Fracción de mutaciones de un campo (negativos por construcción) con puntaje menor que el umbral | `eval_catastro.py`, sección D |
| Riesgo de falsos vínculos | Pares del propio catastro con parcelas distintas que alcanzan el umbral (misma cara de manzana, misma base con otro complemento, misma vía y placa con otro cruce, etc.) | `eval_catastro.py`, sección E |
| Precisión, recall y F1 | Por umbral y tolerancia de placa, positivos de C contra negativos de D (circular) | `eval_catastro.py`, sección F |
| Rendimiento | Pares por segundo, tiempo de parseo, construcción del índice, latencia de consulta, memoria; un solo hilo | `eval_catastro.py`, secciones G y H |
| Vinculación contra el catastro completo | La propia parcela es recuperada, queda en primer lugar, hay empate o el mejor candidato es falso | `eval_catastro.py`, sección H |
| Métricas `same_door` / `same_unit` | Precisión y recall por estrato con etiquetas humanas (tasas condicionales, no poblacionales) | `tools/eval_groundtruth.py` |
| Recall en texto real | Recall al umbral sobre filas `gold`, con intervalos de Wilson al 95 %, tasa de lectura, tasa de falso vínculo en negativos difíciles y recuperación sobre el catastro | `tools/eval_realworld.py` |

Una fila es `gold` cuando la vía, el cruce y la placa del texto coinciden con la parcela etiquetada por GPS. Las etiquetas GPS son ruidosas (el acierto suele ser la parcela vecina), por lo que `gold` no equivale a verdad de campo.

## 2. Cómo reproducirlo

Requiere un entorno con `pyarrow` (`pip install -r requirements-dev.txt`) y datos propios; ver los esquemas en [04](04-evaluacion-y-calibracion.md), sección 5. Las salidas deben quedar fuera del control de versiones.

```bash
# Invariantes, cobertura, rendimiento y vinculación (muestra determinista)
python tools/eval_catastro.py --catastro <ruta>/catastro.parquet --sample 20000 --seed 42 --link-sample 5000 --json-out resultado.json
# Corrida completa (lenta)
python tools/eval_catastro.py --catastro <ruta>/catastro.parquet --full --seed 42 --link-sample 5000

# Pares de referencia y evaluación con etiquetas humanas
python tools/build_groundtruth.py --catastro <ruta>/catastro.parquet --out-dir data/groundtruth --private-dir data/groundtruth_private
python tools/eval_groundtruth.py --labels <hoja_diligenciada>.csv --pairs-dir data/groundtruth --threshold 0.90

# Texto real con etiquetas GPS (ver 04, sección 4, para los argumentos de datos)
python tools/eval_realworld.py --help
```

Reporte siempre con intervalos de confianza, el tamaño de muestra y las limitaciones de la sección 4.

## 3. Lo que el arnés NO mide

| No medido | Consecuencia |
|---|---|
| Calibración del puntaje (probabilidad frente a frecuencia observada) | El puntaje no puede leerse como probabilidad |
| Matriz de confusión sobre una muestra representativa con etiquetas humanas | Se desconocen los falsos positivos y negativos reales |
| AUC o curva ROC | No hay medida de ordenamiento del puntaje |
| Precisión del nivel `structure` con datos reales | `eval_realworld.py` no evalúa `structure` |
| Validación en municipios distintos de Cali o con texto rural | Sin evidencia fuera del catastro de Cali |
| Intervalos de confianza en las pruebas sobre el catastro | Solo `eval_realworld.py` los calcula |
| Rendimiento con carga concurrente | Solo se mide un hilo |
| Otros estándares de dirección (por ejemplo sin `#`) | Ver [10](10-estandar-y-generalizacion.md) |

## 4. Evaluación crítica (limitaciones metodológicas)

1. **Circularidad.** En las secciones C, D y F de `eval_catastro.py` los positivos se reescriben con la gramática del propio proyecto y los negativos son mutaciones de un campo vetadas por construcción. Un resultado perfecto ahí no es evidencia de precisión real (ver `../docs/groundtruth.md`, "Why the previous eval is circular").
2. **Ajuste sobre los mismos datos.** Las reglas de gramática se amplían observando los fallos del catastro, y no hay partición de datos reservada. Las métricas sobre el catastro describen un sistema ajustado a esos datos.
3. **Dependencia de la normalización.** Con texto crudo el rendimiento práctico es bajo; la normalización previa es la condición para un rendimiento útil. Mida ambos escenarios con `--input raw` y `--input normalized`.
4. **Ruido de etiquetas GPS.** Incluso las filas `gold` no equivalen a verdad de campo y las muestras pequeñas dan intervalos amplios.
5. **Falsos vínculos por diseño.** El nivel de cruce (0,97) y el complemento conocido en un lado (0,98) vinculan parcelas distintas a propósito, y hay cadenas canónicas compartidas por varias parcelas. Quien consuma el resultado debe revisar `near_ambiguous`.
6. **Veto por letra de placa incierto.** No se confirmó con el propietario de los datos si la letra es un código de unidad o un sufijo real de placa.
7. **Dependencia del normalizador.** La calidad real depende de un normalizador externo que no forma parte de este proyecto.
8. **Una fuente.** El diseño se ajustó con el catastro de Cali; el rendimiento en otra ciudad o país es desconocido.

## 5. Escala de nivel de confiabilidad (recomendación)

> Esta escala es una **recomendación** de diseño. No fue validada con etiquetas humanas ni calibrada. Ajústela con su propia muestra etiquetada (ver [04](04-evaluacion-y-calibracion.md)) antes de usarla en producción.

| Nivel | Condición (todas deben cumplirse) | Acción recomendada |
|---|---|---|
| A. Aceptable automáticamente, con muestreo de auditoría | Ambas direcciones legibles; puntaje 1,0; `near_ambiguous = False`; `ambiguous = False`; `truncated = False`; sin avisos; nivel `unit`; tolerancia de placa 0 | Aceptar el vínculo para usos de bajo costo de error; auditar una muestra periódicamente |
| B. Revisar | Cualquiera de: puntaje entre 0,9126 y 0,98; `near_ambiguous` o `ambiguous` verdaderos; cualquier aviso (`sin_placa`, `sin_cruce`, `complemento_no_estructurado`); tolerancia de placa mayor que 0; niveles `building` o `structure`; `truncated` | Revisión humana, o presentar todas las alternativas |
| C. Rechazar el vínculo automático | Puntaje menor que 0,90 (incluye 0,85, 0,833, 0,70 y 0,0 por veto) | No vincular; si el contexto lo permite, revisar manualmente los puntajes de 0,85 (una diferencia de un lado) |
| D. No evaluable | Alguna dirección no legible (`parse_ok = False`) | Devolver a normalización; **no** interpretarlo como "no coincide" |

Reglas generales:

- Para decisiones legales, de tributación o de modificación catastral no hay nivel automático: siempre revisión humana (ver [09](09-advertencias-y-limitaciones.md)).
- Si necesita distinguir unidades interiores, use el nivel `unit` y no interprete un 0,98 como "misma unidad".
- Si cambia el umbral, la tolerancia o el nivel, vuelva a medir con su muestra.
