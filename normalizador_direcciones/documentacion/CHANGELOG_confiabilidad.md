# Registro de cambios: `confiabilidad` del normalizador de direcciones

Alcance: la puntuación `confiabilidad` (predio) y `confiabilidad_manzana` (manzana) de `normalizador_direcciones`. No se reentrenó el modelo neuronal ni se cambió el índice ni las reglas de decisión.

> Los resultados de las corridas (métricas de ajuste, validación cruzada, tamaños de muestra, comparaciones) no se publican. Este registro describe solo los cambios de comportamiento y las decisiones. Para medirlos localmente, ver [08](08-metricas-y-confiabilidad.md).

## 1. Procedencia del calibrador servido (v4)

- `artifacts/reliability.json` (`version = reliability-v4-20260925`) es un artefacto guardado, no una evaluación reciente: se ajustó una sola vez con filas de una versión anterior del código. El artefacto no guarda huella del código.
- La etiqueta de predio es un cruce de punto GPS en polígono, sin filtrar por calidad (no es la capa SIG). Esa etiqueta es mayoritariamente ruido del GPS en filas cuyo texto no coincide con el predio.
- Un calibrador ajustado sobre esa etiqueta no discrimina bien entre filas correctas e incorrectas, aunque el cálculo del script sea correcto: lo que falla es el dato.

## 2. Causa raíz de la baja discriminación

1. **El ruido de etiqueta domina.** En filas `gold` (el texto coincide con la dirección catastral del predio GPS) la etiqueta es casi siempre correcta; en filas `noisy` mide sobre todo la distancia del GPS al predio.
2. **Proximidad.** Muchos predios marcados como errados son vecinos del correcto, separados unos pocos metros.
3. **La señal real queda oculta.** Sobre etiquetas `gold`, `margen_predio` (la distancia al segundo predio candidato) ordena bien entre las filas `exacto`; mezclado con filas ruidosas, la señal desaparece.
4. **Heterogeneidad entre fuentes.** La capacidad de ordenar y las tasas base difieren por fuente.
5. **El diseño del híbrido aplana.** El predio se calcula como manzana por `q` por tipo de fila, con `q` casi constante; así el predio solo ordena por `confianza` y descarta `margen_predio`.
6. **El artefacto publicado ordena algo fuera de muestra, pero está mal calibrado**: subestima la probabilidad en filas cuyo texto coincide con el catastro.

## 3. Mejora implementada: artefacto v5 (`reliability-v5-gold`)

Es una regresión logística L2 sobre las 13 variables actuales, ajustada **solo con filas `gold`** de `dev` y `train_pairs`, validada dejando una fuente fuera y con selección anidada de C. `frozen_test` no participó en el ajuste ni en la elección del corte.

Se compararon varios calibradores (logístico sobre `gold`, GBM monótono, híbrido v4 reajustado, v4 sin reajustar y una tasa base constante) con el mismo protocolo; se eligió el logístico. El GBM se descartó porque no mejoró sobre el logístico y su formato no tiene cargador.

**Regla de abstención.** El corte se eligió en desarrollo con la meta "precisión de predio mayor o igual a un valor objetivo", y resultó v5 >= 0.81. Se confirmó una sola vez sobre `frozen_test` `gold`.

**Cambios de código** (probados localmente; las pruebas no se publican en este repositorio):

- `scripts/fit_reliability.py`: opción `--gold-only`; `--rows` acepta varios archivos; el artefacto registra `label_quality`, `code_fingerprint`, `cv_protocol` y `source_rows` con todos los archivos de origen.
- `scripts/eval_strict.py`: copia `gt_quality` a las filas de salida; `ensure_label_columns` completa `has_gt`, `gt_lat` y `gt_lon` cuando faltan; nueva opción `--label-quality gold` para `train_pairs`.
- Nuevo `artifacts/reliability_v2.json` (v5). `artifacts/reliability.json` (v4) **no cambia y sigue siendo el que se sirve**.
- Sin cambios en `src/`. El v5 es de tipo `logistic`, así que el cargador actual lo sirve tal cual. Las pruebas locales cubren variables vacías, NaN, infinitas o no numéricas, variables o categorías no vistas y folds de una sola clase.
- `CHANGELOG.md` del proyecto: nueva entrada.

## 4. Lo que no mejoró

- **Manzana.** Los conjuntos `gold` casi no tienen manzanas erradas, así que no se puede medir; sobre filas ruidosas el v5 ordena peor la manzana que el v4.
- **Filas `noisy`.** El v5 no ordena mejor y su Brier es peor, porque estima la probabilidad de que el predio sea consistente con el texto y no la de coincidir con un GPS ruidoso.
- El GBM monótono y el híbrido reajustado no superaron al logístico.

## 5. Limitaciones que siguen

- El v5 está calibrado para la población "texto consistente con el catastro". En direcciones mal escritas o inexistentes puede ser optimista.
- Una fuente domina las filas de ajuste y la prueba congelada procede de una sola fuente.
- Usar `train_pairs` para el calibrador es válido solo mientras el modelo servido no se entrene con esos pares (hoy `real_frac = 0`).
- El v5 puede dar valores extremos y en el extremo bajo hay poco apoyo.
- **Promoverlo cambia la escala:** el corte del v4 deja de tener sentido y debe usarse el equivalente elegido en desarrollo (v5 >= 0.81).

## 6. Datos que hacen falta

- Una muestra verificada contra la capa SIG o en campo, estratificada por `margen_predio` y por fuente, que incluya filas `noisy` y manzanas erradas.
- Guardar la huella del código en cada artefacto de evaluación. El v5 ya la guarda.

## 7. Decisión pendiente

Si se promueve `reliability_v2.json` a `reliability.json`, siguiendo el procedimiento de `docs/experiments.md`, paso 4, hay que actualizar la regla de uso a `>= 0.81` y aceptar la pérdida en el orden de la manzana.
