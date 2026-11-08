# Informe de reentrenamiento: normalizador basado en nomenclatura (2026-10)

Público: los mantenedores y el equipo de datos que deciden si el normalizador reentrenado es al menos tan confiable como el de producción. Solo agregados (conteos, tasas, valores de métricas, nombres de reglas); sin direcciones, identificadores ni nombres.

## 1. Resumen ejecutivo

Veredicto: NO APRUEBA

Estado de promoción: promoción pendiente de aprobación explícita del usuario

- Criterios aprobados: A1, A3, A5, A7, A8, A11.
- Criterios no aprobados: A2, A4, A6, A9, A10, A12.
- Criterios no evaluados: ninguno.
- Corrida evaluada: `v2_final_catfb` frente a la línea base medida `baseline_prod_2026_10`.
  - dev gold OK: 283 (línea base 278, diferencia +5).
  - dev gold precisión predial de lote: 0.9881 (línea base 0.9960, diferencia -0.0079).
  - dev lost_correct: 121 (línea base 108, diferencia +13).
  - congelada gold OK: 399 (línea base 400, diferencia -1).
- Documentos del índice: `index_v2a_cert` = 343136; `index_v2b_cert_pcer` = 384157; `index_v2c_cert_r2` = 243468; `index_v2d_cert_pcer_r2` = 394919; `index_v2e_cert_pcer_r3` = 394919; `index_v2f_cert_r3` = 243468; `index_v2g_cert_pcer_catfb_r3` = 401860.
- Auditoría JEV `jev_audit`: concordancia 1.0000 sobre 296 filas etiquetadas (la concordancia no es exactitud).
- Auditoría JEV `jev_audit_synthetic`: concordancia 1.0000 sobre 184 filas etiquetadas (la concordancia no es exactitud).
- Las secciones 13 a 21 resumen la historia en tres etapas, la banda de ruido, la calibración, JEV, ONNX, la lectura de A1-A12 y las decisiones abiertas. La prueba congelada de la ronda final es una SEGUNDA MIRADA, no una prueba ciega (sección 13).
- La promoción de cualquier artefacto a `artifacts/` es un paso aparte que requiere la aprobación explícita de los mantenedores; este informe no la autoriza.

## 2. Alcance y decisiones

Decisiones que determinan el significado de las cifras (las fechas son las de cada decisión):

- 2026-10-30: la fuente principal es la capa WFS de IDESC (registros certificados `cert` más los aún no certificados `pcer`); la copia del shapefile se conserva solo como referencia de validación.
- 2026-10-30: todas las coordenadas del índice provienen de reproyectar `ndcoordx` / `ndcoordy` (EPSG:6249 a EPSG:4326) localmente con el `.prj` del shapefile; las lon/lat del WFS se conservan solo para auditoría. Las compuertas de comparación se ajustaron a nivel de lote para el NPN (>= 98 %) y a mediana <= 1 m / p99 <= 2 m para coordenadas.
- 2026-10-31: solo fuentes nuevas. El índice se construye únicamente con los registros de nomenclatura; el catastro es de solo lectura (auditoría y compuerta de techo del oráculo) y nunca es fuente de documentos, NPN, `n_predios`, manzana o centroides. La cobertura de direcciones catastrales sin registro de nomenclatura se pierde por construcción y se cuantifica en las secciones 4 y 6.
- 2026-10-31: la semántica predial es a nivel de lote. `gt_predial` es catastral y a nivel de lote, mientras que un NPN de nomenclatura puede llevar un sufijo de unidad; por eso A2/A3 y la compuerta del oráculo comparan `NPN[:21]`, y la precisión exacta por NPN se reporta junto a ella. La línea base medida se recalcula con la misma métrica de lote.
- 2026-10-31: la pre-normalización que separa letras (`pre_normalize_separates_letters`) está desactivada por defecto para que el comportamiento de producción y la línea base no cambien; se activa por artefacto mediante `index_meta.json`. Los índices nuevos la llevan activada.
- La validación de salidas (`validate_outputs`: off / report / correct) está desactivada por defecto; este informe muestra el efecto de `report` y `correct` por separado (sección 8).

Indicadores registrados por artefacto:

| Índice | pre_normalize_separates_letters | desplazamiento objectid | esquema de index_meta |
|---|---|---|---|
| `index_v2a_cert` | true | n/d (pendiente) | 1 |
| `index_v2b_cert_pcer` | true | n/d (pendiente) | 1 |
| `index_v2c_cert_r2` | true | 10000000 | 2 |
| `index_v2d_cert_pcer_r2` | true | 10000000 | 2 |
| `index_v2e_cert_pcer_r3` | true | 10000000 | 2 |
| `index_v2f_cert_r3` | true | 10000000 | 2 |
| `index_v2g_cert_pcer_catfb_r3` | true | 10000000 | 2 |

## 3. Perfil de datos

Tres fuentes, una primaria:

| Fuente | Filas | Notas |
|---|---|---|
| Copia del shapefile (solo certificados) | 437825 | EPSG:6249, referencia de validación |
| Capa WFS (primaria) | 692389 | cert=437945; lote=808; pcer=253627; prov=9 |
| Catastro de producción (solo lectura) | 330387 | referencia de auditoría, nunca fuente de documentos |

Descarga WFS: páginas 139, coincidencias 692389, descargadas 692389, estado ok, corte de la capa 2026-09-03Z. El shapefile (exportado el 2026-10-30) puede ser más reciente que el corte del WFS; la compuerta a nivel de lote absorbe las diferencias por sufijo de unidad.

Limpieza (antes de cualquier ingesta):

| Magnitud | Valor |
|---|---|
| filas de entrada | 692389 |
| filas limpias | 691263 |
| filas en cuarentena | 1126 |
| proporción en cuarentena | 0.0016 |
| límite de cuarentena | 0.0050 |
| compuerta | aprueba |

| Regla de limpieza | Filas |
|---|---|
| control_chars_removed | 0 |
| coords_fallback_to_wfs | 7541 |
| coords_nan | 0 |
| coords_outside_bbox | 0 |
| coords_projected_missing | 0 |
| coords_projected_zero | 7541 |
| coords_shift_vs_shp | 0 |
| duplicate_gid | 0 |
| duplicate_ndid_conflicting | 4 |
| duplicate_ndid_identical | 0 |
| empty_address | 305 |
| marker_removed | 254528 |
| npn_invalid | 2 |
| npn_len_28 | 1 |
| npn_len_29 | 1 |
| number_sign_normalized | 1 |
| quotes_removed | 1 |
| rebuilt_from_parts | 101 |
| status_invalid | 817 |

Estados excluidos del índice: lote=808; prov=9.
Fuente de coordenadas: local_6249_to_4326=683722; wfs=7541.

Reproyección del shapefile (EPSG:6249 a EPSG:4326):

| Magnitud | Valor |
|---|---|
| SRC | EPSG:6249 a EPSG:4326 |
| puntos | 437825 |
| geometrías NaN | 0 |
| fuera del bbox de Cali | 0 |
| NPN compartidos verificados contra el catastro | 192180 |
| desplazamiento mediano (m) | 0.054 |
| desplazamiento p90 (m) | 2.804 |
| desplazamiento máximo (m) | 4872.6 |
| compuerta bbox | aprueba |
| compuerta de cordura (mediana < 30 m advertencia, > 100 m aborta) | aprueba |

WFS frente a shapefile (compuertas ajustadas; cada resultado se recalcula a partir del valor y el umbral):

| Control | Valor | Umbral | Resultado |
|---|---|---|---|
| proporción de diferencia de conteo certificado | 0.00027 | <= 0.005 | aprueba |
| concordancia de direcciones (ids comunes) | 0.99998 | >= 0.995 | aprueba |
| concordancia NPN a nivel de lote (NPN[:21]) | 0.9896 | >= 0.98 (98 %) | aprueba |
| concordancia NPN, 30 dígitos completos | 0.7003 | informativo | informativo |
| desplazamiento de coordenadas, mediana (m) | 0.721 | <= 1.0 | aprueba |
| desplazamiento de coordenadas, p99 (m) | 1.376 | <= 2.0 | aprueba |
| ids comunes comparados | 319690 | - | informativo |

La discrepancia del NPN completo es el sufijo de unidad (últimos dos dígitos); el lote coincide. El desplazamiento de coordenadas es una diferencia de transformación de datum entre GeoServer y pyproj, no un error de datos.

## 4. Composición del índice y contabilidad de registros

Composición por índice (los documentos son direcciones normalizadas únicas; las unidades viven en un archivo auxiliar):

| Índice | Niveles | Documentos | Documentos por fuente | Tasa de fallo del parser | Proporción de documentos de respaldo | Proporción de consultas oficiales | Consultas sintéticas | Consultas oficiales | Filas de unidades |
|---|---|---|---|---|---|---|---|---|---|
| `index_v2a_cert` | cert,catastro | 343136 | catastro=99664; nomenclatura_cert=243472 | 0.0000 | 0.0000 | n/d (pendiente) | 1029408 | 1465536 | 193535 |
| `index_v2b_cert_pcer` | cert,pcer,catastro | 384157 | catastro=31092; nomenclatura_cert=243472; nomenclatura_pcer=109593 | 0.0016 | 0.0011 | n/d (pendiente) | 1152471 | 2112066 | 344195 |
| `index_v2c_cert_r2` | cert | 243468 | nomenclatura_cert=243468 | 0.0000 | 0.0000 | 0.4917 | 730404 | 706520 | 193416 |
| `index_v2d_cert_pcer_r2` | cert,pcer | 394919 | nomenclatura_cert=243468; nomenclatura_pcer=151451 | 0.0016 | 0.0022 | 0.4861 | 1184757 | 1120682 | 282831 |
| `index_v2e_cert_pcer_r3` | cert,pcer | 394919 | nomenclatura_cert=243468; nomenclatura_pcer=151451 | 0.0016 | 0.0022 | 0.4861 | 1184757 | 1120682 | 282831 |
| `index_v2f_cert_r3` | cert | 243468 | nomenclatura_cert=243468 | 0.0000 | 0.0000 | 0.4917 | 730404 | 706520 | 193416 |
| `index_v2g_cert_pcer_catfb_r3` | cert,pcer | 401860 | catastro=6941; nomenclatura_cert=243468; nomenclatura_pcer=151451 | 0.0016 | 0.0022 | 0.4861 | 1184757 | 1120682 | 282831 |

Contabilidad de registros: cada registro de entrada limpio cae en exactamente un grupo.

| Grupo | `index_v2a_cert` | `index_v2b_cert_pcer` | `index_v2c_cert_r2` | `index_v2d_cert_pcer_r2` | `index_v2e_cert_pcer_r3` | `index_v2f_cert_r3` | `index_v2g_cert_pcer_catfb_r3` |
|---|---|---|---|---|---|---|---|
| base_doc_new | 236416 | 313005 | 236416 | 365567 | 365567 | 236416 | 365567 |
| base_doc_merged | 7989 | 32956 | 7988 | 31695 | 31695 | 7988 | 31695 |
| unit | 193535 | 344195 | 193416 | 282831 | 282831 | 193416 | 282831 |
| fallback_doc | 1 | 1107 | 1 | 1107 | 1107 | 1 | 1107 |
| dropped | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| excluded_tier | 253322 | 0 | 253322 | 0 | 0 | 253322 | 0 |
| incomplete_address | n/d (pendiente) | n/d (pendiente) | 81 | 10024 | 10024 | 81 | 10024 |
| spread_outlier | n/d (pendiente) | n/d (pendiente) | 39 | 39 | 39 | 39 | 39 |
| total (grupos) | 691263 | 691263 | 691263 | 691263 | 691263 | 691263 | 691263 |
| entrada | 691263 | 691263 | 691263 | 691263 | 691263 | 691263 | 691263 |
| en cuarentena antes de indexar | n/d (pendiente) | n/d (pendiente) | 1126 | 1126 | 1126 | 1126 | 1126 |
| contabilidad correcta | aprueba | aprueba | aprueba | aprueba | aprueba | aprueba | aprueba |

Techo del oráculo (mejor coincidencia de lote alcanzable en filas gold) y pérdida de cobertura:

| Índice | Compuerta | Caída máxima (pp) | Producción coincidente, lotes en la fuente (%) | Nuevo coincidente, lotes en la fuente (%) | Pérdida de cobertura, titular (pp) | Pérdida inherente a la fuente (pp) | Particiones fallidas |
|---|---|---|---|---|---|---|---|
| `index_v2a_cert` | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) |
| `index_v2b_cert_pcer` | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) |
| `index_v2c_cert_r2` | aprueba | 2.0 | 100.00 | 98.63 | 27.48 | 27.48 | ninguna |
| `index_v2d_cert_pcer_r2` | aprueba | 2.0 | 99.71 | 97.81 | 3.97 | 2.83 | ninguna |
| `index_v2e_cert_pcer_r3` | aprueba | 2.0 | 99.71 | 97.81 | 3.97 | 2.83 | ninguna |
| `index_v2f_cert_r3` | aprueba | 2.0 | 100.00 | 98.63 | 27.48 | 27.48 | ninguna |
| `index_v2g_cert_pcer_catfb_r3` | aprueba | 2.0 | 99.71 | 100.44 | 1.42 | 2.83 | ninguna |

La pérdida titular cuenta cada lote gold que cubre el índice de producción; la parte inherente es el lote ausente de la propia fuente de nomenclatura, que ningún modelo puede recuperar. La compuerta de construcción compara solo los lotes presentes en la fuente.

Comparación de índices contra el índice de producción (auditoría de solo lectura):

| Índice | NPN compartidos | Desplazamiento p50 (m) | p90 (m) | p99 (m) | máx (m) | Cobertura de lotes gold, producción | Cobertura de lotes gold, nuevo |
|---|---|---|---|---|---|---|---|
| `index_v2a_cert` | 270939 | 0.034 | 0.985 | 9.76 | 4872.6 | 0.9759 | 0.9759 |
| `index_v2b_cert_pcer` | 254769 | 0.046 | 1.550 | 13.00 | 9328.2 | 0.9759 | 0.9745 |

## 5. Entrenamiento

Receta: receta de producción desde cero (`mode=mixed`, `AddressEncoder` nuevo, con semilla primero), épocas configuradas 8, batch 2048, lr 0.0030, minutos máximos 120; las cadenas oficiales son consultas de entrenamiento (`q_source=official`).

Entorno: GPU NVIDIA GeForce RTX 5070 Ti (capacidad de cómputo 12.0), torch n/d (pendiente), CUDA n/d (pendiente).

| Corrida | Semilla | Épocas completadas/configuradas | Detenida antes | Recall@1 de validación, consultas oficiales | Recall@1 de validación, consultas sintéticas | Segundos de entrenamiento | Estado |
|---|---|---|---|---|---|---|---|
| v2e_cert_pcer_s42_r3_cal | 42 | 8/8 | no | 0.9967 | 0.8806 | 2298.4 | heredado de v2d_cert_pcer_s42 |
| v2g_cert_pcer_catfb_s42_r3_basecal | 42 | 8/8 | no | 0.9967 | 0.8806 | 2298.4 | heredado de v2d_cert_pcer_s42 |
| v2e_cert_pcer_s43_cal | 43 | 8/8 | no | 0.9964 | 0.8849 | 2355.3 | correcto |
| v2_final_catfb | 42 | 8/8 | no | 0.9967 | 0.8806 | 2298.4 | heredado de v2d_cert_pcer_s42 |
| v2_final | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) |

El recall de validación sintético no es comparable entre índices (documentos distintos, mezcla de corrupción distinta); el recall de consultas oficiales es más cercano pero sigue ligado al índice. Ninguno sirve como selector: la selección usa solo las métricas dev gold.

## 6. Resultados antes y después (dev y congelada)

Las tasas son fracciones. Las diferencias comparan la corrida evaluada con la línea base MEDIDA (mismo código, mismas filas). Dev gold es pequeño (289 filas): una variación de 1-2 filas está dentro de la banda de ruido.

| Métrica | baseline_prod_2026_10 (línea base) | v2e_cert_pcer_s42_r3_cal | v2g_cert_pcer_catfb_s42_r3_basecal | v2e_cert_pcer_s43_cal | v2_final_catfb | v2_final | Diferencia (v2_final_catfb - línea base) |
|---|---|---|---|---|---|---|---|
| **dev (todas las filas)** |  |  |  |  |  |  |  |
| dev filas | 2078 | n/d (pendiente) | n/d (pendiente) | 2078 | 2078 | 2078 | 0 |
| dev OK | 1084 | n/d (pendiente) | n/d (pendiente) | 1167 | 1231 | 1217 | +147 |
| dev precisión manzana | 0.8630 | n/d (pendiente) | n/d (pendiente) | 0.8555 | 0.8609 | 0.8614 | -0.0021 |
| dev precisión predial exacta | 0.6465 | n/d (pendiente) | n/d (pendiente) | 0.1102 | 0.1073 | 0.1061 | -0.5392 |
| dev precisión predial de lote | 0.6566 | n/d (pendiente) | n/d (pendiente) | 0.6551 | 0.6631 | 0.6667 | +0.0065 |
| dev lost_correct | 108 | n/d (pendiente) | n/d (pendiente) | 140 | 121 | 115 | +13 |
| **dev (filas gold)** |  |  |  |  |  |  |  |
| dev gold filas | 289 | n/d (pendiente) | n/d (pendiente) | 289 | 289 | 289 | 0 |
| dev gold OK | 278 | n/d (pendiente) | n/d (pendiente) | 270 | 283 | 281 | +5 |
| dev gold proporción de OK respondidas | 0.9619 | n/d (pendiente) | n/d (pendiente) | 0.9343 | 0.9792 | 0.9723 | +0.0173 |
| dev gold precisión manzana | 0.9964 | n/d (pendiente) | n/d (pendiente) | 0.9925 | 0.9929 | 0.9928 | -0.0035 |
| dev gold precisión predial exacta | 0.9880 | n/d (pendiente) | n/d (pendiente) | 0.1962 | 0.1865 | 0.1833 | -0.8015 |
| dev gold precisión predial de lote | 0.9960 | n/d (pendiente) | n/d (pendiente) | 0.9846 | 0.9881 | 0.9880 | -0.0079 |
| dev gold lost_correct | 9 | n/d (pendiente) | n/d (pendiente) | 15 | 6 | 5 | -3 |
| **congelada (prueba evaluada una sola vez, --final)** |  |  |  |  |  |  |  |
| congelada todas OK | 1002 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 1090 | 1062 | +88 |
| congelada todas precisión manzana | 0.7314 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.7121 | 0.7112 | -0.0192 |
| congelada todas precisión predial exacta | 0.4518 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.1361 | 0.1240 | -0.3158 |
| congelada todas precisión predial de lote | 0.4912 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.5044 | 0.5026 | +0.0132 |
| congelada todas lost_correct | 223 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 221 | 225 | -2 |
| congelada gold OK | 400 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 399 | 381 | -1 |
| congelada gold precisión manzana | 0.9949 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.9847 | 0.9840 | -0.0101 |
| congelada gold precisión predial exacta | 0.9231 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.3096 | 0.2853 | -0.6135 |
| congelada gold precisión predial de lote | 0.9615 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.9938 | 0.9968 | +0.0323 |
| congelada gold lost_correct | 12 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 9 | 14 | -3 |

Confianza en 0.70 (dev gold, definición relativa de A10: la línea base dentro de los márgenes; la precisión se calcula sobre filas OK que emiten predial):

| Magnitud | Línea base | v2e_cert_pcer_s42_r3_cal | v2g_cert_pcer_catfb_s42_r3_basecal | v2e_cert_pcer_s43_cal | v2_final_catfb | v2_final | Diferencia |
|---|---|---|---|---|---|---|---|
| proporción de filas gold con confianza >= 0.70 | 0.9619 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.9792 | 0.9723 | +0.0173 |
| precisión predial de lote (filas que emiten predial) | 0.9960 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.9881 | 0.9880 | -0.0079 |
| tasa de error manzana | 0.0036 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 0.0071 | 0.0071 | +0.0035 |
| filas OK que emiten predial | 250 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 252 | 251 | +2 |

Precisión frente al umbral de confianza (línea base y corrida evaluada):

| Umbral | Proporción de filas (línea base) | Precisión de lote (línea base) | Proporción de filas (corrida) | Precisión de lote (corrida) |
|---|---|---|---|---|
| 0.50 | 0.9619 | 0.9960 | 0.9792 | 0.9881 |
| 0.55 | 0.9619 | 0.9960 | 0.9792 | 0.9881 |
| 0.60 | 0.9619 | 0.9960 | 0.9792 | 0.9881 |
| 0.65 | 0.9619 | 0.9960 | 0.9792 | 0.9881 |
| 0.70 | 0.9619 | 0.9960 | 0.9792 | 0.9881 |
| 0.75 | 0.9619 | 0.9960 | 0.9723 | 0.9880 |
| 0.80 | 0.9550 | 0.9960 | 0.9619 | 0.9880 |
| 0.85 | 0.9343 | 0.9960 | 0.9412 | 0.9879 |
| 0.90 | 0.8374 | 0.9956 | 0.8512 | 0.9913 |
| 0.95 | 0.4498 | 0.9921 | 0.1903 | 1.0000 |

## 7. Desglose por fuente (cert vs pcer)

Aciertos separados por nivel de fuente del documento seleccionado (`nomenclatura_cert` certificado, `nomenclatura_pcer` pendiente de certificación; `(none)` = filas sin documento seleccionado):

| Corrida | Conjunto | Alcance | Fuente | Filas | OK | Precisión manzana | Precisión predial exacta | Precisión predial de lote | lost_correct |
|---|---|---|---|---|---|---|---|---|---|
| v2e_cert_pcer_s42_r3_cal | dev_all | all | (none) | 861 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 115 |
| v2e_cert_pcer_s42_r3_cal | dev_all | all | nomenclatura_cert | 797 | 797 | 0.8678 | 0.1088 | 0.6618 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_all | all | nomenclatura_pcer | 420 | 420 | 0.8480 | 0.0984 | 0.6803 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_all | gold | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s42_r3_cal | dev_all | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_all | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_all_validate | all | (none) | 861 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 115 |
| v2e_cert_pcer_s42_r3_cal | dev_all_validate | all | nomenclatura_cert | 797 | 797 | 0.8678 | 0.1088 | 0.6618 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_all_validate | all | nomenclatura_pcer | 420 | 420 | 0.8480 | 0.0984 | 0.6803 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_all_validate | gold | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s42_r3_cal | dev_all_validate | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_all_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold | all | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s42_r3_cal | dev_gold | all | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold | gold | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s42_r3_cal | dev_gold | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold_validate | all | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s42_r3_cal | dev_gold_validate | all | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold_validate | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold_validate | gold | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s42_r3_cal | dev_gold_validate | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2e_cert_pcer_s42_r3_cal | dev_gold_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | all | (none) | 847 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 121 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | all | catastro | 19 | 19 | 0.8000 | 0.2000 | 0.4000 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | all | nomenclatura_cert | 793 | 793 | 0.8678 | 0.1088 | 0.6618 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | all | nomenclatura_pcer | 419 | 419 | 0.8480 | 0.0992 | 0.6777 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | gold | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | gold | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | all | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | all | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | all | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | gold | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | gold | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2e_cert_pcer_s43_cal | dev_all | all | (none) | 862 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 126 |
| v2e_cert_pcer_s43_cal | dev_all | all | nomenclatura_cert | 797 | 797 | 0.8643 | 0.1088 | 0.6588 | 0 |
| v2e_cert_pcer_s43_cal | dev_all | all | nomenclatura_pcer | 419 | 419 | 0.8529 | 0.0949 | 0.6861 | 0 |
| v2e_cert_pcer_s43_cal | dev_all | gold | (none) | 9 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s43_cal | dev_all | gold | nomenclatura_cert | 204 | 204 | 0.9901 | 0.1735 | 0.9949 | 0 |
| v2e_cert_pcer_s43_cal | dev_all | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2203 | 0.9661 | 0 |
| v2e_cert_pcer_s43_cal | dev_all_validate | all | (none) | 862 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 126 |
| v2e_cert_pcer_s43_cal | dev_all_validate | all | nomenclatura_cert | 797 | 797 | 0.8643 | 0.1088 | 0.6588 | 0 |
| v2e_cert_pcer_s43_cal | dev_all_validate | all | nomenclatura_pcer | 419 | 419 | 0.8529 | 0.0949 | 0.6861 | 0 |
| v2e_cert_pcer_s43_cal | dev_all_validate | gold | (none) | 9 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s43_cal | dev_all_validate | gold | nomenclatura_cert | 204 | 204 | 0.9901 | 0.1735 | 0.9949 | 0 |
| v2e_cert_pcer_s43_cal | dev_all_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2203 | 0.9661 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold | all | (none) | 9 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s43_cal | dev_gold | all | nomenclatura_cert | 204 | 204 | 0.9901 | 0.1735 | 0.9949 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2203 | 0.9661 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold | gold | (none) | 9 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s43_cal | dev_gold | gold | nomenclatura_cert | 204 | 204 | 0.9901 | 0.1735 | 0.9949 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2203 | 0.9661 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold_validate | all | (none) | 9 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s43_cal | dev_gold_validate | all | nomenclatura_cert | 204 | 204 | 0.9901 | 0.1735 | 0.9949 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold_validate | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2203 | 0.9661 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold_validate | gold | (none) | 9 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2e_cert_pcer_s43_cal | dev_gold_validate | gold | nomenclatura_cert | 204 | 204 | 0.9901 | 0.1735 | 0.9949 | 0 |
| v2e_cert_pcer_s43_cal | dev_gold_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2203 | 0.9661 | 0 |
| v2_final_catfb | dev_all | all | (none) | 847 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 121 |
| v2_final_catfb | dev_all | all | catastro | 19 | 19 | 0.8000 | 0.2000 | 0.4000 | 0 |
| v2_final_catfb | dev_all | all | nomenclatura_cert | 793 | 793 | 0.8678 | 0.1088 | 0.6618 | 0 |
| v2_final_catfb | dev_all | all | nomenclatura_pcer | 419 | 419 | 0.8480 | 0.0992 | 0.6777 | 0 |
| v2_final_catfb | dev_all | gold | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2_final_catfb | dev_all | gold | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2_final_catfb | dev_all | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final_catfb | dev_all | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2_final_catfb | dev_all_validate | all | (none) | 847 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 121 |
| v2_final_catfb | dev_all_validate | all | catastro | 19 | 19 | 0.8000 | 0.2000 | 0.4000 | 0 |
| v2_final_catfb | dev_all_validate | all | nomenclatura_cert | 793 | 793 | 0.8678 | 0.1088 | 0.6618 | 0 |
| v2_final_catfb | dev_all_validate | all | nomenclatura_pcer | 419 | 419 | 0.8480 | 0.0992 | 0.6777 | 0 |
| v2_final_catfb | dev_all_validate | gold | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2_final_catfb | dev_all_validate | gold | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2_final_catfb | dev_all_validate | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final_catfb | dev_all_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2_final_catfb | dev_gold | all | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2_final_catfb | dev_gold | all | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2_final_catfb | dev_gold | all | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final_catfb | dev_gold | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2_final_catfb | dev_gold | gold | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2_final_catfb | dev_gold | gold | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2_final_catfb | dev_gold | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final_catfb | dev_gold | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2_final_catfb | dev_gold_validate | all | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2_final_catfb | dev_gold_validate | all | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2_final_catfb | dev_gold_validate | all | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final_catfb | dev_gold_validate | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2_final_catfb | dev_gold_validate | gold | (none) | 6 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 6 |
| v2_final_catfb | dev_gold_validate | gold | catastro | 2 | 2 | 1.0000 | 0.5000 | 1.0000 | 0 |
| v2_final_catfb | dev_gold_validate | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final_catfb | dev_gold_validate | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2264 | 0.9623 | 0 |
| v2_final_catfb | final_all | all | (none) | 910 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 221 |
| v2_final_catfb | final_all | all | catastro | 38 | 38 | 0.7143 | 0.4800 | 0.6000 | 0 |
| v2_final_catfb | final_all | all | nomenclatura_cert | 748 | 748 | 0.6821 | 0.1376 | 0.4631 | 0 |
| v2_final_catfb | final_all | all | nomenclatura_pcer | 304 | 304 | 0.7875 | 0.0833 | 0.6278 | 0 |
| v2_final_catfb | final_all | gold | (none) | 18 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 9 |
| v2_final_catfb | final_all | gold | catastro | 18 | 18 | 1.0000 | 0.8462 | 0.9231 | 0 |
| v2_final_catfb | final_all | gold | nomenclatura_cert | 269 | 269 | 0.9813 | 0.3275 | 0.9956 | 0 |
| v2_final_catfb | final_all | gold | nomenclatura_pcer | 112 | 112 | 0.9907 | 0.1728 | 1.0000 | 0 |
| v2_final_catfb | final_gold | all | (none) | 18 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 9 |
| v2_final_catfb | final_gold | all | catastro | 18 | 18 | 1.0000 | 0.8462 | 0.9231 | 0 |
| v2_final_catfb | final_gold | all | nomenclatura_cert | 269 | 269 | 0.9813 | 0.3275 | 0.9956 | 0 |
| v2_final_catfb | final_gold | all | nomenclatura_pcer | 112 | 112 | 0.9907 | 0.1728 | 1.0000 | 0 |
| v2_final_catfb | final_gold | gold | (none) | 18 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 9 |
| v2_final_catfb | final_gold | gold | catastro | 18 | 18 | 1.0000 | 0.8462 | 0.9231 | 0 |
| v2_final_catfb | final_gold | gold | nomenclatura_cert | 269 | 269 | 0.9813 | 0.3275 | 0.9956 | 0 |
| v2_final_catfb | final_gold | gold | nomenclatura_pcer | 112 | 112 | 0.9907 | 0.1728 | 1.0000 | 0 |
| v2_final | dev_all | all | (none) | 861 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 115 |
| v2_final | dev_all | all | nomenclatura_cert | 797 | 797 | 0.8678 | 0.1088 | 0.6618 | 0 |
| v2_final | dev_all | all | nomenclatura_pcer | 420 | 420 | 0.8480 | 0.0984 | 0.6803 | 0 |
| v2_final | dev_all | gold | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2_final | dev_all | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final | dev_all | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2_final | dev_gold | all | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2_final | dev_gold | all | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final | dev_gold | all | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2_final | dev_gold | gold | (none) | 8 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 5 |
| v2_final | dev_gold | gold | nomenclatura_cert | 205 | 205 | 0.9902 | 0.1726 | 0.9949 | 0 |
| v2_final | dev_gold | gold | nomenclatura_pcer | 76 | 76 | 1.0000 | 0.2222 | 0.9630 | 0 |
| v2_final | final_all | all | (none) | 938 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 225 |
| v2_final | final_all | all | nomenclatura_cert | 755 | 755 | 0.6816 | 0.1374 | 0.4640 | 0 |
| v2_final | final_all | all | nomenclatura_pcer | 307 | 307 | 0.7855 | 0.0811 | 0.6270 | 0 |
| v2_final | final_all | gold | (none) | 36 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 14 |
| v2_final | final_all | gold | nomenclatura_cert | 269 | 269 | 0.9813 | 0.3275 | 0.9956 | 0 |
| v2_final | final_all | gold | nomenclatura_pcer | 112 | 112 | 0.9907 | 0.1687 | 1.0000 | 0 |
| v2_final | final_gold | all | (none) | 36 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 14 |
| v2_final | final_gold | all | nomenclatura_cert | 269 | 269 | 0.9813 | 0.3275 | 0.9956 | 0 |
| v2_final | final_gold | all | nomenclatura_pcer | 112 | 112 | 0.9907 | 0.1687 | 1.0000 | 0 |
| v2_final | final_gold | gold | (none) | 36 | 0 | n/d (pendiente) | n/d (pendiente) | n/d (pendiente) | 14 |
| v2_final | final_gold | gold | nomenclatura_cert | 269 | 269 | 0.9813 | 0.3275 | 0.9956 | 0 |
| v2_final | final_gold | gold | nomenclatura_pcer | 112 | 112 | 0.9907 | 0.1687 | 1.0000 | 0 |

Concordancia de filas entre la línea base (antigua) y la corrida (nueva) sobre filas gold de dev + congelada, unidas por posición (solo conteos):

| Corrida | Corte | Filas | both_ok_predial_match | only_old | only_new | both_wrong |
|---|---|---|---|---|---|---|
| v2_final_catfb | todas | 706 | 138 | 421 | 9 | 138 |
| v2_final_catfb | catastro | n/d (pendiente) | 12 | 1 | 0 | 7 |
| v2_final_catfb | nomenclatura_cert | n/d (pendiente) | 101 | 297 | 8 | 68 |
| v2_final_catfb | nomenclatura_pcer | n/d (pendiente) | 25 | 121 | 1 | 41 |
| v2_final_catfb | unknown | n/d (pendiente) | 0 | 2 | 0 | 22 |
| v2_final_catfb | nivel de lote | 706 | 519 | 55 | 51 | 81 |
| v2_final | todas | 706 | 126 | 433 | 9 | 138 |
| v2_final | nomenclatura_cert | n/d (pendiente) | 101 | 297 | 8 | 68 |
| v2_final | nomenclatura_pcer | n/d (pendiente) | 25 | 121 | 1 | 41 |
| v2_final | unknown | n/d (pendiente) | 0 | 15 | 0 | 29 |
| v2_final | nivel de lote | 706 | 508 | 66 | 51 | 81 |

## 8. Validación de salidas (pydantic)

Validación de salidas con pydantic (`validate_outputs`). En modo `report` las violaciones se cuentan y la salida no se modifica; en modo `correct` solo se aplican correcciones conservadoras (nunca más optimistas) y el conteo de violaciones restantes debería llegar a 0:

| Corrida | Conjunto de evaluación | Modo | Regla | Violaciones | Corregidas | Filas revisadas |
|---|---|---|---|---|---|---|
| baseline_prod_2026_10 | dev_all_validate | report | ok_without_support | 1 | 0 | 2078 |
| baseline_prod_2026_10 | dev_gold_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2e_cert_pcer_s42_r3_cal | dev_all_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2e_cert_pcer_s42_r3_cal | dev_gold_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_all_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2g_cert_pcer_catfb_s42_r3_basecal | dev_gold_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2e_cert_pcer_s43_cal | dev_all | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2e_cert_pcer_s43_cal | dev_all_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2e_cert_pcer_s43_cal | dev_gold | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2e_cert_pcer_s43_cal | dev_gold_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2_final_catfb | dev_all_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |
| v2_final_catfb | dev_gold_validate | n/d (pendiente) | (sin violaciones) | 0 | 0 | n/d (pendiente) |

## 9. Auditoría JEV

JEV es un juez basado en LLM, no una verdad de referencia: la concordancia no es exactitud, y cada discrepancia requiere revisión humana. Solo se auditan filas públicas de nomenclatura y consultas sintéticas derivadas del índice público; las direcciones ciudadanas (dev, congelada, pares de entrenamiento) nunca se envían.

Auditoría `jev_audit` (fuente nomenclatura; orígenes nomenclatura_public=300; filas privadas permitidas: no):

| Magnitud | Valor |
|---|---|
| filas auditadas | 300 |
| respondidas | 300 |
| errores | 0 |
| omitidas | 0 |
| veredictos indeterminados | 4 |
| veredictos decisivos (etiquetados) | 296 |
| tasa de concordancia | 1.0000 |
| IC 95 % de concordancia (Wilson) | 0.9872 a 1.0000 |
| concordancia / n | 296 / 296 |
| kappa | indefinido |
| AUC | indefinido |
| Brier | indefinido |

No se devolvió ningún veredicto decisivo "no" (o ningún "sí" decisivo), así que los veredictos etiquetados contienen una sola clase: kappa y AUC son indefinidos (n/d), no cero, y la tasa de concordancia no puede separar a un buen juez de uno permisivo.

Conteos de compuerta: auto=296; escalate=1; review=3.

| Grupo | Etiqueta | Filas | Etiquetadas | Tasa de concordancia |
|---|---|---|---|---|
| estado_nomenclatura | cert | 202 | 202 | 1.0000 |
| estado_nomenclatura | pcer | 98 | 94 | 1.0000 |
| marker_removed | false | 202 | 202 | 1.0000 |
| marker_removed | true | 98 | 94 | 1.0000 |
| rebuilt_from_parts | false | 300 | 296 | 1.0000 |
| zona | EXP | 10 | 10 | 1.0000 |
| zona | URB | 290 | 286 | 1.0000 |

Auditoría `jev_audit_synthetic` (fuente synthetic; orígenes synthetic=300; filas privadas permitidas: no):

| Magnitud | Valor |
|---|---|
| filas auditadas | 300 |
| respondidas | 275 |
| errores | 0 |
| omitidas | 25 |
| veredictos indeterminados | 91 |
| veredictos decisivos (etiquetados) | 184 |
| tasa de concordancia | 1.0000 |
| IC 95 % de concordancia (Wilson) | 0.9795 a 1.0000 |
| concordancia / n | 184 / 184 |
| kappa | indefinido |
| AUC | indefinido |
| Brier | 0.1122 |

No se devolvió ningún veredicto decisivo "no" (o ningún "sí" decisivo), así que los veredictos etiquetados contienen una sola clase: kappa y AUC son indefinidos (n/d), no cero, y la tasa de concordancia no puede separar a un buen juez de uno permisivo.

Conteos de compuerta: auto=184; escalate=12; review=79.

| Grupo | Etiqueta | Filas | Etiquetadas | Tasa de concordancia |
|---|---|---|---|---|
| estado | NO_PARSEABLE | 9 | 0 | n/d (pendiente) |
| estado | OK | 245 | 165 | 1.0000 |
| estado | SIN_MATCH | 46 | 19 | 1.0000 |
| fuente_normalizacion | catastro | 231 | 163 | 1.0000 |
| fuente_normalizacion | reglas | 44 | 21 | 1.0000 |
| fuente_normalizacion | unknown | 25 | 0 | n/d (pendiente) |

## 10. Lista de criterios de aceptación (A1-A12)

Los umbrales están escritos en `docs/retrain_2026-10_acceptance.md` antes de cualquier corrida. Un criterio que no pudo evaluarse bloquea un veredicto APRUEBA; nunca cuenta como aprobado.

| ID | Estado | Criterio | Evidencia |
|---|---|---|---|
| A1 | APRUEBA | dev_gold_ok >= baseline - 1 | dev_gold_ok=283 >= 277 |
| A2 | NO APRUEBA | dev_gold_predial_lot_precision >= baseline (lot level, NPN[:21]; the exact precision is reported) | dev_gold_predial_lot_precision=0.9881 >= 0.9960 -- dev_gold_predial_lot_precision=0.9880952380952381 violates >= 0.996 |
| A3 | APRUEBA | frozen gold OK >= baseline - 2 rows; frozen gold lot-level predial precision >= baseline - 2 / baseline frozen gold OK | frozen_gold_ok=399 >= 398; frozen_gold_predial_lot_precision=0.9938 >= 0.9565 |
| A4 | NO APRUEBA | dev_lost_correct <= max(measured baseline, s2b_retrain_s42, s2b_retrain_s43 dev_lost_correct) (112 when the noise band is not recorded) | dev_lost_correct=121 <= 112 -- dev_lost_correct=121 violates <= 112 |
| A5 | APRUEBA | dev_manzana_precision >= baseline - 0.005 | dev_manzana_precision=0.8609 >= 0.8580 |
| A6 | NO APRUEBA | refitted reliability (same kind as production), BOTH heads: pooled CV calibration error (ECE) <= the refitted baseline's + 0.01, and the ship rule against the refit on the same rows passes (the rule against the v2 reference is not required) | ship_rule_vs_refit=False i... True; calib_error_cv_predio=0.0044 <= 0.1075; calib_error_cv_manzana=0.0022 <= 0.0438 -- ship_rule_vs_refit is False, expected true |
| A7 | APRUEBA | cert parser failure rate < 1 %; CRS gates pass; (c1) construction: over gold rows whose lot exists in the cleaned source records, the new index's matched share >= the production index's - 2.0 pp; (c2) coverage loss: (production lot share - new lot share) on gold <= the inherent loss (lot absent from the source / rows) + 1.0 pp; the headline coverage numbers and the missing-lot counts are reported; accounting.dropped == 0 or every drop explained | parser_failure_rate_cert=0.0016 < 0.0100; crs_gates_passed=True i... True; gold_lot_cond_new=1.0044 >= 0.9771; gold_coverage_loss=0.0142 <= 0.0383; accounting_dropped=0 == 0 |
| A8 | APRUEBA | production_untouched is true in every run | production_untouched=True i... True; baseline_production_untouched=True i... True |
| A9 | NO APRUEBA | full pytest green in .venv; check_no_real_addresses.py exits 0 | pytest_green=False i... True; no_real_addresses_exit=0 == 0 -- pytest_green is False, expected true |
| A10 | NO APRUEBA | at confianza >= 0.7 on dev gold rows, relative to the measured baseline: (a) share of gold rows with estado OK >= baseline - 1.0 pp; (b) lot-level predial precision among OK rows that emit a predial >= baseline - 0.5 pp; (c) manzana error rate among OK rows <= baseline + 0.5 pp | (otra etiqueta)=0.9792 >= 0.9519; (otra etiqueta)=0.9881 >= 0.9910; (otra etiqueta)=0.0071 <= 0.0086 -- predial_lot_precision_emitted@0.7=0.9880952380952381 violates >= 0.991 |
| A11 | APRUEBA | cleaning quarantine < 0.5 %; build_index.py and download_nomenclatura.py exit 0 on real data and on the anomaly fixtures | quarantine_share=0.0016 < 0.0050; build_index_exit=0 == 0; download_exit=0 == 0; anomaly_fixtures_ok=True i... True |
| A12 | NO APRUEBA | with validate_outputs=correct: A1-A5 and A10 do not get worse and 0 violations remain on dev | A1_with_correct=283 c... 277; A2_with_correct=0.9881 c... 0.9960; A3_with_correct=n/d (pendiente) c... n/d (pendiente); A4_with_correct=121 c... 112; A5_with_correct=0.8609 c... 0.8580; A10_with_correct=n/d (pendiente) c... n/d (pendiente) -- A2 with validate_outputs=correct: dev_gold_predial_lot_precision=0.988095238095238 violates >= 0.996; A4 with validate_outputs=correct: dev_lost_correct=121 ... |

## 11. Riesgos y limitaciones

- Costo de cobertura: el índice de solo fuentes nuevas pierde direcciones catastrales sin registro de nomenclatura. Medido en filas gold: pérdida titular 1.42 pp, de la cual 2.83 pp es inherente a la fuente (sección 4).
- Sesgo de la verdad de referencia hacia el catastro: `gt_predial` es catastral. Un NPN certificado legítimamente distinto cuenta como error, así que la precisión predial es una cota inferior de la calidad real; la evaluación a nivel de lote y el diagnóstico "correcto frente a cualquier fuente" reducen, pero no eliminan, ese sesgo.
- Banda de ruido: dev gold tiene 289 filas, así que 1-2 filas (aproximadamente 0.3-0.7 puntos) son ruido. Los reentrenamientos de referencia de la receta de producción dieron dev lost_correct s2b_retrain_s42=107, s2b_retrain_s43=112.
- El recall de validación sintético y el oficial no son comparables entre índices; son diagnósticos, nunca selectores.
- La prueba congelada se evalúa una sola vez, solo sobre la corrida seleccionada (`--final`); no se debe ajustar contra ella.
- Los registros `pcer` están pendientes de certificación: se etiquetan `nomenclatura_pcer` y pueden eliminarse del índice sin tocar el resto si la sección 7 muestra que restan.
- La concordancia JEV no es exactitud y cubre solo filas públicas o sintéticas (sección 9).
- La promoción no forma parte de este informe: requiere la aprobación explícita de los mantenedores.

## 12. Comandos de reproducción

Ejecutar desde `models/normalizador_direcciones` con el `.venv` (CPU); el entrenamiento usa `.venv-train` (CUDA). Pasos en orden:

```bash
.venv/Scripts/python scripts/reproject_nomenclatura.py --shp train_data/nomenclatura_certificada_20261030/nomenclatura_certificada_20261030.shp
.venv/Scripts/python scripts/download_nomenclatura.py --out artifacts/nomenclatura_wfs.parquet
.venv/Scripts/python scripts/clean_nomenclatura.py --in artifacts/nomenclatura_wfs.parquet --out artifacts/nomenclatura_clean.parquet
# Índices (r3): base cert+pcer, solo cert y con respaldo de catastro (el respaldo se agrega sobre el base, que no se modifica)
.venv/Scripts/python scripts/build_index.py --tiers cert,pcer --out-dir artifacts/index_v2e_cert_pcer_r3   # ver docs/model-artifacts.md
.venv/Scripts/python scripts/build_index.py --tiers cert --out-dir artifacts/index_v2f_cert_r3
.venv/Scripts/python scripts/index_fallback.py add --base-index artifacts/index_v2e_cert_pcer_r3 --splits-dir artifacts/splits --out-dir artifacts/index_v2g_cert_pcer_catfb_r3
.venv/Scripts/python scripts/compare_indexes.py --old artifacts --new artifacts/index_v2g_cert_pcer_catfb_r3
# Modelo semilla 42: ya entrenado sobre el índice r2 anterior (mismo texto y orden de documentos que el r3); se reutiliza sin reentrenar:
# .venv-train/Scripts/python scripts/run_experiment.py --tag v2d_cert_pcer_s42 --index-dir artifacts/index_v2d_cert_pcer_r2 --set max_minutes=120
.venv/Scripts/python scripts/make_inherit_experiment.py --index-dir artifacts/index_v2e_cert_pcer_r3 --model-dir artifacts/experiments/v2d_cert_pcer_s42 --out artifacts/experiments/v2e_cert_pcer_s42_r3
# Semilla 43 (banda de ruido): entrenamiento completo sobre el índice r3
.venv-train/Scripts/python scripts/run_experiment.py --tag v2e_cert_pcer_s43 --index-dir artifacts/index_v2e_cert_pcer_r3 --set seed=43 --set max_minutes=120
# Calibración de la capa de decisión (CPU): se ajusta con los pares de train_pairs y se instala en un directorio nuevo
.venv/Scripts/python scripts/fit_decision.py --artifacts-dir artifacts/experiments/v2e_cert_pcer_s42_r3 --pairs artifacts/splits/train_pairs.parquet --out-dir artifacts/experiments/calib_design/params --tag cand --device cpu
.venv/Scripts/python scripts/make_decision_experiment.py --source artifacts/experiments/v2e_cert_pcer_s42_r3 --out artifacts/experiments/v2e_cert_pcer_s42_r3_cal --decision artifacts/experiments/calib_design/params/decision_cand.json --reliability artifacts/experiments/calib_design/params/reliability_calibrated_cand.json
.venv/Scripts/python scripts/fit_decision.py --artifacts-dir artifacts/experiments/v2e_cert_pcer_s43 --pairs artifacts/splits/train_pairs.parquet --out-dir artifacts/experiments/calib_s43/params --tag cand43 --device cpu
.venv/Scripts/python scripts/make_decision_experiment.py --source artifacts/experiments/v2e_cert_pcer_s43 --out artifacts/experiments/v2e_cert_pcer_s43_cal --decision artifacts/experiments/calib_s43/params/decision_cand43.json --reliability artifacts/experiments/calib_s43/params/reliability_calibrated_cand43.json
# Candidato final: modelo del r3 con los documentos de respaldo embebidos y la calibración del candidato (sin reentrenar)
.venv/Scripts/python scripts/index_fallback.py experiment --index-dir artifacts/index_v2g_cert_pcer_catfb_r3 --model-dir artifacts/experiments/v2e_cert_pcer_s42_r3 --out-dir artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 --device cpu
.venv/Scripts/python scripts/make_decision_experiment.py --source artifacts/experiments/v2g_cert_pcer_catfb_s42_r3 --out artifacts/experiments/v2g_cert_pcer_catfb_s42_r3_basecal --decision artifacts/experiments/calib_design/params/decision_cand.json --reliability artifacts/experiments/calib_design/params/reliability_calibrated_cand.json
.venv/Scripts/python scripts/run_experiment.py --tag v2_final_catfb --eval-only --from-dir artifacts/experiments/v2g_cert_pcer_catfb_s42_r3_basecal --eval-set both --final
.venv/Scripts/python scripts/run_experiment.py --tag baseline_prod_2026_10 --eval-only --from-dir artifacts --eval-set both --final
.venv/Scripts/python scripts/compare_runs.py --old artifacts/experiments/baseline_prod_2026_10 --new artifacts/experiments/v2_final_catfb
.venv/Scripts/python scripts/audit_with_jev.py --source nomenclatura --sample 300 --env-file <ruta al .env>   # requiere una clave JEV; agregar --dry-run primero
python scripts/write_retrain_report.py --out docs/retrain_2026-10_nomenclatura.md \
  --baseline-tag baseline_prod_2026_10 \
  --candidates v2e_cert_pcer_s42_r3_cal v2g_cert_pcer_catfb_s42_r3_basecal v2e_cert_pcer_s43_cal v2_final_catfb \
  --control v2_final \
  --final-tag v2_final_catfb \
  --jev-dir artifacts/jev_audit artifacts/jev_audit_synthetic
```

## 13. Historia en tres etapas y segunda mirada a la prueba congelada

ADVERTENCIA: la evaluación sobre la prueba congelada de la ronda final es una SEGUNDA MIRADA y no es ciega. La decisión de agregar el respaldo de catastro se tomó después de ver el resultado congelado de la ronda 1 (la brecha de cobertura). No se ajustó ningún parámetro con la congelada y se puntuó una sola vez, pero esa puntuación no es una prueba independiente.

**Etapa 1: ronda 1, sin respaldo de catastro (corrida `v2_final`).** El índice se construyó solo con registros de nomenclatura (cert y pcer). En la prueba congelada gold el resultado fue 381 filas OK de 417, con 30 filas cuyo lote no está en el índice: esa es la brecha de cobertura que motivó la etapa siguiente.

**Etapa 2: decisión de agregar el respaldo de catastro.** Se agrega, como último recurso y de forma reversible (sección 21), 6941 documentos de catastro a partir de 330387 candidatos (colisiones omitidas: 2245); el índice resultante tiene 401860 documentos. El modelo no se reentrena: se reutiliza el de la ronda 1 y los documentos agregados se embeben con él.

**Etapa 3: ronda 2, con respaldo.** Se evaluaron dos variantes de calibración sobre el índice con respaldo (`basecal` conserva la calibración de la ronda 1; `cal` la reajusta) y se eligió con las métricas dev gold, sin usar la prueba congelada.

| Variante | dev gold OK | lotes correctos / emitidos | precisión de lote dev gold | precisión manzana dev (todas) | dev lost_correct | piso de A5 (0.8580) |
|---|---|---|---|---|---|---|
| basecal (SELECCIONADA) | 283 | 249 / 252 | 0.9881 | 0.8609 | 121 | cumple |
| cal | 283 | 258 / 261 | 0.9885 | 0.8485 | 90 | no cumple |

Selección: `basecal`. Empate en precisión de lote dentro de una fila gold entre `basecal` y `cal`; se desempató por la precisión manzana sobre todas las filas dev, y `cal` además queda bajo el piso de A5.

Prueba congelada, filas gold, ronda 2 menos ronda 1 sobre las MISMAS filas (SEGUNDA MIRADA):

| Métrica | Ronda 1 | Ronda 2 | Diferencia |
|---|---|---|---|
| filas | 417 | 417 | 0 |
| filas OK | 381 | 399 | +18 |
| filas que emiten predial | 312 | 323 | +11 |
| lotes correctos | 311 | 321 | +10 |
| precisión de lote (filas que emiten predial) | 0.9968 | 0.9938 | -0.0030 |
| precisión manzana | 0.9840 | 0.9847 | +0.0007 |
| lost_correct | 14 | 9 | -5 |
| lote ausente del índice | 30 | 15 | -15 |

Transiciones: filas OK ganadas 18, perdidas 0; lotes correctos ganados 12, perdidos 2, conservados 309.

Prueba congelada, todas las filas, ronda 2 menos ronda 1 sobre las MISMAS filas (SEGUNDA MIRADA):

| Métrica | Ronda 1 | Ronda 2 | Diferencia |
|---|---|---|---|
| filas | 2000 | 2000 | 0 |
| filas OK | 1062 | 1090 | +28 |
| filas que emiten predial | 782 | 801 | +19 |
| lotes correctos | 393 | 404 | +11 |
| precisión de lote (filas que emiten predial) | 0.5026 | 0.5044 | +0.0018 |
| precisión manzana | 0.7112 | 0.7121 | +0.0010 |
| lost_correct | 225 | 221 | -4 |
| lote ausente del índice | 305 | 87 | -218 |

Transiciones: filas OK ganadas 30, perdidas 2; lotes correctos ganados 15, perdidos 4, conservados 389.

## 14. Banda de ruido entre semillas y ruido de muestreo

La banda de ruido se mide con dos semillas del mismo procedimiento sobre el índice de la ronda 1: la diferencia entre ellas es lo que se mueve por azar de entrenamiento y no por un cambio real. Una fila gold de dev equivale a 0.346 puntos porcentuales (289 filas).

| Métrica | Semilla s42 | Semilla s43 | Diferencia absoluta |
|---|---|---|---|
| dev gold OK | 281 | 280 | 1 |
| dev gold proporción de OK respondidas | 0.9723 | 0.9689 | 0.0035 |
| dev gold filas que emiten predial | 251 | 255 | 4 |
| dev gold precisión de lote | 0.9880 | 0.9882 | 0.0002 |
| dev gold precisión manzana | 0.9928 | 0.9928 | 0.0000 |
| dev OK (todas) | 1217 | 1216 | 1 |
| dev precisión de lote (todas) | 0.6667 | 0.6667 | 0.0000 |
| dev precisión manzana (todas) | 0.8614 | 0.8606 | 0.0008 |
| dev lost_correct | 115 | 126 | 11 |
| umbral tau_p | 0.9400 | 0.9450 | 0.0050 |
| ECE fuera de muestra (predio) | 0.0044 | 0.0043 | 0.0001 |
| Brier fuera de muestra (predio) | 0.0271 | 0.0276 | 0.0005 |
| AUC fuera de muestra (predio) | 0.8951 | 0.9008 | 0.0056 |
| ECE fuera de muestra (manzana) | 0.0022 | 0.0017 | 0.0005 |
| Brier fuera de muestra (manzana) | 0.0040 | 0.0054 | 0.0014 |
| AUC fuera de muestra (manzana) | 0.9146 | 0.8995 | 0.0151 |

Lectura contra la corrida seleccionada (las semillas se midieron sobre el índice sin respaldo, así que la comparación sirve como orden de magnitud y no como igualdad):

- dev lost_correct: seleccionada 121; semillas 115 y 126: dentro del rango entre semillas.
- dev gold precisión de lote: seleccionada 0.9881; semillas 0.9880 y 0.9882: dentro del rango entre semillas.
- dev gold OK: seleccionada 283; semillas 281 y 280: fuera del rango entre semillas.

Ruido de muestreo: diferencias pareadas contra la línea base con bootstrap sobre las filas (10000 remuestreos, semilla 20261031; diferencia = candidato - línea base; intervalo percentil de 95 %). Si el intervalo contiene el cero, la diferencia no se distingue del ruido de muestreo:

| Conjunto | Métrica | Línea base | Candidato | Diferencia | IC 95 % | P(diferencia < 0) |
|---|---|---|---|---|---|---|
| dev | precisión manzana (todas las filas) | 0.8630 | 0.8609 | -0.0021 | [-0.0211 a 0.0168] | 0.5841 |
| dev | precisión de lote gold | 0.9960 | 0.9881 | -0.0079 | [-0.0202 a 0.0001] | 0.9010 |
| dev | filas gold OK | 278 | 283 | +5 | [1 a 10] | 0.0000 |
| dev | lost_correct | 108 | 121 | +13 | [-5 a 31] | 0.0731 |
| congelada (segunda mirada) | precisión manzana (todas las filas) | 0.7314 | 0.7121 | -0.0192 | [-0.0366 a -0.0027] | 0.9885 |
| congelada (segunda mirada) | precisión de lote gold | 0.9615 | 0.9938 | +0.0323 | [0.0121 a 0.0537] | 0.0002 |
| congelada (segunda mirada) | filas gold OK | 400 | 399 | -1 | [-9 a 7] | 0.5476 |
| congelada (segunda mirada) | lost_correct | 223 | 221 | -2 | [-27 a 23] | 0.5428 |

## 15. Precisión por banda de confianza y por fuente (intervalos de Wilson)

Los intervalos son de Wilson al 95 % (un intervalo ancho indica pocas filas; una banda sin filas evaluadas no tiene precisión definida). Las filas congeladas son una SEGUNDA MIRADA, no una prueba ciega (ver la advertencia de la sección 13).

Por banda de confianza (`alta` >= 0.97, `media` >= 0.94, `baja` por debajo; `sin_dato` sin banda):

| Conjunto | Banda | Filas | Lotes correctos / evaluados | Precisión de lote [IC 95 % Wilson] | Manzanas correctas / evaluadas | Precisión manzana [IC 95 % Wilson] |
|---|---|---|---|---|---|---|
| dev_gold | alta | 224 | 223 / 224 | 0.9955 [0.9751 a 0.9992] | 222 / 224 | 0.9911 [0.9680 a 0.9975] |
| dev_gold | media | 28 | 26 / 28 | 0.9286 [0.7735 a 0.9802] | 28 / 28 | 1.0000 [0.8794 a 1.0000] |
| dev_gold | baja | 31 | 0 / 0 | n/d (sin filas evaluadas) | 28 / 28 | 1.0000 [0.8794 a 1.0000] |
| dev_all | alta | 823 | 272 / 407 | 0.6683 [0.6212 a 0.7123] | 350 / 407 | 0.8600 [0.8229 a 0.8903] |
| dev_all | media | 133 | 37 / 59 | 0.6271 [0.4995 a 0.7392] | 48 / 59 | 0.8136 [0.6962 a 0.8926] |
| dev_all | baja | 275 | 0 / 0 | n/d (sin filas evaluadas) | 66 / 73 | 0.9041 [0.8150 a 0.9528] |
| frozen_gold (segunda mirada) | alta | 288 | 287 / 288 | 0.9965 [0.9806 a 0.9994] | 284 / 288 | 0.9861 [0.9648 a 0.9946] |
| frozen_gold (segunda mirada) | media | 35 | 34 / 35 | 0.9714 [0.8547 a 0.9949] | 35 / 35 | 1.0000 [0.9011 a 1.0000] |
| frozen_gold (segunda mirada) | baja | 76 | 0 / 0 | n/d (sin filas evaluadas) | 68 / 70 | 0.9714 [0.9017 a 0.9921] |
| frozen_all (segunda mirada) | alta | 691 | 352 / 691 | 0.5094 [0.4722 a 0.5465] | 492 / 691 | 0.7120 [0.6771 a 0.7445] |
| frozen_all (segunda mirada) | media | 110 | 52 / 110 | 0.4727 [0.3819 a 0.5654] | 83 / 110 | 0.7545 [0.6664 a 0.8255] |
| frozen_all (segunda mirada) | baja | 289 | 0 / 0 | n/d (sin filas evaluadas) | 135 / 196 | 0.6888 [0.6209 a 0.7494] |

Por nivel de fuente del documento seleccionado (candidato; la línea base no tiene niveles de fuente):

| Conjunto | Fuente | Filas | Lotes correctos / evaluados | Precisión de lote [IC 95 % Wilson] | Manzanas correctas / evaluadas | Precisión manzana [IC 95 % Wilson] |
|---|---|---|---|---|---|---|
| dev_gold | catastro | 2 | 2 / 2 | 1.0000 [0.3424 a 1.0000] | 2 / 2 | 1.0000 [0.3424 a 1.0000] |
| dev_gold | nomenclatura_cert | 205 | 196 / 197 | 0.9949 [0.9718 a 0.9991] | 202 / 204 | 0.9902 [0.9650 a 0.9973] |
| dev_gold | nomenclatura_pcer | 76 | 51 / 53 | 0.9623 [0.8725 a 0.9896] | 74 / 74 | 1.0000 [0.9506 a 1.0000] |
| dev_all | catastro | 19 | 2 / 5 | 0.4000 [0.1176 a 0.7693] | 4 / 5 | 0.8000 [0.3755 a 0.9638] |
| dev_all | nomenclatura_cert | 793 | 225 / 340 | 0.6618 [0.6099 a 0.7100] | 315 / 363 | 0.8678 [0.8290 a 0.8988] |
| dev_all | nomenclatura_pcer | 419 | 82 / 121 | 0.6777 [0.5901 a 0.7544] | 145 / 171 | 0.8480 [0.7865 a 0.8941] |
| frozen_gold (segunda mirada) | catastro | 18 | 12 / 13 | 0.9231 [0.6669 a 0.9863] | 18 / 18 | 1.0000 [0.8241 a 1.0000] |
| frozen_gold (segunda mirada) | nomenclatura_cert | 269 | 228 / 229 | 0.9956 [0.9757 a 0.9992] | 262 / 267 | 0.9813 [0.9569 a 0.9920] |
| frozen_gold (segunda mirada) | nomenclatura_pcer | 112 | 81 / 81 | 1.0000 [0.9547 a 1.0000] | 107 / 108 | 0.9907 [0.9494 a 0.9984] |
| frozen_all (segunda mirada) | catastro | 38 | 15 / 25 | 0.6000 [0.4074 a 0.7660] | 25 / 35 | 0.7143 [0.5494 a 0.8367] |
| frozen_all (segunda mirada) | nomenclatura_cert | 748 | 276 / 596 | 0.4631 [0.4234 a 0.5032] | 470 / 689 | 0.6821 [0.6465 a 0.7158] |
| frozen_all (segunda mirada) | nomenclatura_pcer | 304 | 113 / 180 | 0.6278 [0.5552 a 0.6950] | 215 / 273 | 0.7875 [0.7352 a 0.8319] |

## 16. Calidad de calibración

La calibración se ajusta sobre pares reales de entrenamiento nunca usados para entrenar el modelo y se mide fuera de muestra (validación cruzada de 5 grupos por dirección). El candidato seleccionado conserva la calibración de la ronda 1.

Parámetros de decisión: tau_p 0.94, tau_m 0.70, piso de confianza 0.70, bandas: alta >= 0.97, media >= 0.94.

Calidad fuera de muestra (OOF) de las dos cabezas:

| Cabeza | Pares | ECE (10 bins) | Brier | AUC | Brier del refit | AUC del refit | Límite de ECE de A6 (refit de la línea base) |
|---|---|---|---|---|---|---|---|
| predio | 5087 | 0.0044 | 0.0271 | 0.8951 | 0.0270 | 0.8985 | 0.1075 |
| manzana | 5014 | 0.0022 | 0.0040 | 0.9146 | 0.0039 | 0.9176 | 0.0438 |

Diagramas de confiabilidad como tabla (probabilidad predicha frente a frecuencia observada por bin de igual tamaño; sin imágenes):

dev_gold: predio n 252, predicho medio 0.9867, observado 0.9881; manzana n 280, predicho medio 0.9975, observado 0.9929.

| Bin | Predio n | Predio predicho | Predio observado | Manzana n | Manzana predicha | Manzana observada |
|---|---|---|---|---|---|---|
| 0 | 26 | 0.9560 | 0.9615 | 28 | 0.9885 | 0.9643 |
| 1 | 26 | 0.9772 | 0.9615 | 28 | 0.9962 | 1.0000 |
| 2 | 25 | 0.9855 | 1.0000 | 28 | 0.9974 | 1.0000 |
| 3 | 25 | 0.9878 | 0.9600 | 28 | 0.9985 | 1.0000 |
| 4 | 25 | 0.9902 | 1.0000 | 28 | 0.9990 | 0.9643 |
| 5 | 25 | 0.9919 | 1.0000 | 28 | 0.9990 | 1.0000 |
| 6 | 25 | 0.9932 | 1.0000 | 28 | 0.9990 | 1.0000 |
| 7 | 25 | 0.9944 | 1.0000 | 28 | 0.9990 | 1.0000 |
| 8 | 25 | 0.9954 | 1.0000 | 28 | 0.9990 | 1.0000 |
| 9 | 25 | 0.9969 | 1.0000 | 28 | 0.9990 | 1.0000 |

dev_all: predio n 466, predicho medio 0.9861, observado 0.6631; manzana n 539, predicho medio 0.9962, observado 0.8609.

| Bin | Predio n | Predio predicho | Predio observado | Manzana n | Manzana predicha | Manzana observada |
|---|---|---|---|---|---|---|
| 0 | 47 | 0.9545 | 0.6596 | 54 | 0.9758 | 0.8519 |
| 1 | 47 | 0.9738 | 0.6596 | 54 | 0.9962 | 0.8333 |
| 2 | 47 | 0.9839 | 0.7234 | 54 | 0.9976 | 0.9444 |
| 3 | 47 | 0.9874 | 0.7234 | 54 | 0.9988 | 0.8148 |
| 4 | 47 | 0.9901 | 0.5745 | 54 | 0.9990 | 0.8148 |
| 5 | 47 | 0.9917 | 0.7234 | 54 | 0.9990 | 0.8519 |
| 6 | 46 | 0.9931 | 0.6087 | 54 | 0.9990 | 0.8889 |
| 7 | 46 | 0.9943 | 0.6304 | 54 | 0.9990 | 0.8889 |
| 8 | 46 | 0.9955 | 0.6522 | 54 | 0.9990 | 0.8519 |
| 9 | 46 | 0.9973 | 0.6739 | 53 | 0.9990 | 0.8679 |

frozen_gold (segunda mirada): predio n 323, predicho medio 0.9868, observado 0.9938; manzana n 393, predicho medio 0.9956, observado 0.9847.

| Bin | Predio n | Predio predicho | Predio observado | Manzana n | Manzana predicha | Manzana observada |
|---|---|---|---|---|---|---|
| 0 | 33 | 0.9558 | 0.9697 | 40 | 0.9744 | 0.9500 |
| 1 | 33 | 0.9789 | 1.0000 | 40 | 0.9944 | 1.0000 |
| 2 | 33 | 0.9854 | 0.9697 | 40 | 0.9964 | 0.9750 |
| 3 | 32 | 0.9879 | 1.0000 | 39 | 0.9977 | 1.0000 |
| 4 | 32 | 0.9895 | 1.0000 | 39 | 0.9990 | 1.0000 |
| 5 | 32 | 0.9913 | 1.0000 | 39 | 0.9990 | 0.9744 |
| 6 | 32 | 0.9929 | 1.0000 | 39 | 0.9990 | 0.9744 |
| 7 | 32 | 0.9942 | 1.0000 | 39 | 0.9990 | 1.0000 |
| 8 | 32 | 0.9956 | 1.0000 | 39 | 0.9990 | 1.0000 |
| 9 | 32 | 0.9975 | 1.0000 | 39 | 0.9990 | 0.9744 |

## 17. Auditoría JEV: límites y contraste por banda

Límites: JEV es un juez basado en LLM de equivalencia textual entre la consulta y la dirección normalizada; no es una verdad de referencia ni un oráculo de corrección de lote. Casi no devuelve veredictos decisivos "no", así que la concordancia es trivial (una sola clase: kappa 0 o indefinido, AUC sin significado con muy pocos negativos) y no distingue a un buen juez de uno permisivo; los veredictos indeterminados son una fracción grande. Solo se auditan filas públicas de nomenclatura/catastro y consultas sintéticas; las direcciones ciudadanas nunca se envían. Cada discrepancia requiere revisión humana.

| Auditoría | Filas | Respondidas | Omitidas | Indeterminadas | Etiquetadas | Concordancia | IC 95 % (Wilson) | kappa | AUC | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_candidate | 300 | 274 | 26 | 82 | 192 | 0.9948 | 0.9711 a 0.9991 | 0.0000 | 0.7833 | 0.0080 |
| synthetic_baseline | 300 | 278 | 22 | 86 | 192 | 1.0000 | 0.9804 a 1.0000 | indefinido | indefinido | 0.1167 |
| catfbdocs_candidate | 300 | 210 | 90 | 87 | 123 | 0.9837 | 0.9426 a 0.9955 | 0.0000 | 0.6302 | 0.0249 |
| catfbdocs_baseline | 300 | 210 | 90 | 89 | 121 | 0.9752 | 0.9296 a 0.9915 | 0.0000 | 0.5918 | 0.1310 |

Contraste por banda de confianza y por fuente (qué tan seguido JEV dice "no" donde el modelo es menos seguro):

| Auditoría | Corte | Grupo | Juzgadas | Sí | No decisivos | Indeterminadas | p medio de JEV | Tasa de no decisivo |
|---|---|---|---|---|---|---|---|---|
| synthetic_candidate | banda | alta | 180 | 135 | 1 | 44 | 0.8771 | 0.0056 |
| synthetic_candidate | banda | media | 36 | 28 | 0 | 8 | 0.8678 | 0.0000 |
| synthetic_candidate | banda | baja | 34 | 16 | 0 | 18 | 0.6962 | 0.0000 |
| synthetic_candidate | banda | sin_dato | 24 | 12 | 0 | 12 | 0.8125 | 0.0000 |
| synthetic_candidate | fuente | catastro | 3 | 0 | 0 | 3 | 0.4567 | 0.0000 |
| synthetic_candidate | fuente | nomenclatura_cert | 160 | 122 | 0 | 38 | 0.8801 | 0.0000 |
| synthetic_candidate | fuente | nomenclatura_pcer | 87 | 57 | 1 | 29 | 0.8115 | 0.0115 |
| synthetic_candidate | fuente | sin_dato | 24 | 12 | 0 | 12 | 0.8125 | 0.0000 |
| catfbdocs_candidate | banda | alta | 143 | 107 | 0 | 36 | 0.8727 | 0.0000 |
| catfbdocs_candidate | banda | media | 11 | 5 | 0 | 6 | 0.6545 | 0.0000 |
| catfbdocs_candidate | banda | baja | 35 | 9 | 2 | 24 | 0.5471 | 0.0571 |
| catfbdocs_candidate | banda | sin_dato | 21 | 0 | 0 | 21 | 0.5624 | 0.0000 |
| catfbdocs_candidate | fuente | catastro | 185 | 120 | 2 | 63 | 0.8050 | 0.0108 |
| catfbdocs_candidate | fuente | nomenclatura_cert | 2 | 0 | 0 | 2 | 0.4000 | 0.0000 |
| catfbdocs_candidate | fuente | nomenclatura_pcer | 2 | 1 | 0 | 1 | 0.7150 | 0.0000 |
| catfbdocs_candidate | fuente | sin_dato | 21 | 0 | 0 | 21 | 0.5624 | 0.0000 |

Los "no" decisivos por banda aparecen en: `synthetic_candidate`/alta (1), `catfbdocs_candidate`/baja (2).

La sección 9 detalla cada auditoría por directorio cuando se indicaron con `--jev-dir`.

## 18. Rendimiento: ONNX y benchmark

Fuente: `onnx_bench_final` (corrida final del benchmark).

Benchmark con 16 hilos (consultas públicas corrompidas; mediana de 5 repeticiones):

| Conjunto | Backend | Documentos | Arranque en frío (s) | Latencia p50 (ms) | Latencia p95 (ms) | Filas/s (lote 100) | Filas/s (lote 2000) | Filas/s (lote 10000) | RSS pico (MB) |
|---|---|---|---|---|---|---|---|---|---|
| producción | torch | 330387 | 0.90 | 19.9 | 22.2 | 136 | 142 | 137 | 1568 |
| producción | onnx | 330387 | 1.08 | 17.9 | 19.9 | 209 | 216 | 215 | 1675 |
| producción | onnx-int8 | 330387 | 1.07 | 17.4 | 19.4 | 215 | 222 | 221 | 1779 |
| candidato | torch | 401860 | 1.09 | 23.5 | 25.9 | 116 | 120 | 120 | 1827 |
| candidato | onnx | 401860 | 1.28 | 21.5 | 23.8 | 179 | 182 | 184 | 1962 |
| candidato | onnx-int8 | 401860 | 1.28 | 20.9 | 23.3 | 186 | 190 | 190 | 2077 |

Benchmark con 4 hilos:

| Conjunto | Backend | Documentos | Arranque en frío (s) | Latencia p50 (ms) | Latencia p95 (ms) | Filas/s (lote 100) | Filas/s (lote 2000) | Filas/s (lote 10000) | RSS pico (MB) |
|---|---|---|---|---|---|---|---|---|---|
| producción | torch | 330387 | 0.88 | 29.3 | 31.9 | 46 | 49 | 49 | 1542 |
| producción | onnx | 330387 | 1.07 | 21.3 | 23.5 | 187 | 195 | 192 | 1660 |
| producción | onnx-int8 | 330387 | 1.05 | 19.5 | 21.7 | 208 | 211 | 209 | 1758 |
| candidato | torch | 401860 | 1.10 | 35.1 | 38.3 | 41 | 40 | 40 | 1788 |
| candidato | onnx | 401860 | 1.29 | 25.8 | 28.7 | 155 | 155 | 155 | 1926 |
| candidato | onnx-int8 | 401860 | 1.29 | 24.9 | 27.3 | 175 | 176 | 176 | 2040 |

Benchmark del candidato con índice fp16 (16 hilos):

| Conjunto | Backend | Documentos | Arranque en frío (s) | Latencia p50 (ms) | Latencia p95 (ms) | Filas/s (lote 100) | Filas/s (lote 2000) | Filas/s (lote 10000) | RSS pico (MB) |
|---|---|---|---|---|---|---|---|---|---|
| cand_fp16 | onnx | 401860 | 1.29 | 181.7 | 197.6 | 141 | 165 | 165 | 1743 |

Paridad con torch (documentos públicos): el top-1 que difiere y la deriva máxima de coordenadas en metros:

| Conjunto | Backend | Consultas | Coseno mínimo | Top-1 idéntico | Top-1 distinto | Deriva máxima de coordenadas (m) |
|---|---|---|---|---|---|---|
| producción | onnx | 2000 | 1.000000 | 1.0000 | 0 | 0.0 |
| producción | onnx-int8 | 2000 | 0.999618 | 1.0000 | 0 | 138.8 |
| candidato | onnx | 2000 | 1.000000 | 0.9995 | 1 | 0.0 |
| candidato | onnx-int8 | 2000 | 0.999500 | 1.0000 | 0 | 231.8 |

Efecto sobre las métricas dev gold (fuente: `onnx_bench`) (filas con el mismo estado y el mismo NPN que torch):

| Backend | Filas | OK | Mismo estado que torch | Mismo NPN que torch |
|---|---|---|---|---|
| onnx-int8 | 289 | 278 | 289 | 288 |
| onnx | 289 | 278 | 289 | 289 |
| torch | 289 | 278 | (referencia) | (referencia) |

## 19. Lectura de A1-A12 frente a la banda de ruido

| ID | Estado | Valor | Umbral | Motivo |
|---|---|---|---|---|
| A1 | APRUEBA | 283 | 277 |  |
| A2 | NO APRUEBA | 0.9881 | 0.9960 | dev_gold_predial_lot_precision=0.9881 no cumple >= 0.9960 |
| A3 | APRUEBA | frozen_gold_ok=399; frozen_gold_predial_lot_precision=0.9938 | frozen_gold_ok=398; frozen_gold_predial_lot_precision=0.9565 |  |
| A4 | NO APRUEBA | 121 | 112 | dev_lost_correct=121 no cumple <= 112 |
| A5 | APRUEBA | 0.8609 | 0.8580 |  |
| A6 | NO APRUEBA | ship_rule_vs_refit=no; calib_error_cv_predio=0.0044; calib_error_cv_manzana=0.0022 | ship_rule_vs_refit=sí; calib_error_cv_predio=0.1075; calib_error_cv_manzana=0.0438 | ship_rule_vs_refit debe ser sí |
| A7 | APRUEBA | parser_failure_rate_cert=0.0016; crs_gates_passed=sí; gold_lot_cond_new=1.0044; gold_coverage_loss=0.0142; accounting_dropped=0; gold_lot_coverage_new=0.9759; gold_lot_coverage_old=0.9901; gold_lot_missing_new=17; gold_lot_absent_from_so... | parser_failure_rate_cert=0.0100; crs_gates_passed=sí; gold_lot_cond_new=0.9771; gold_coverage_loss=0.0383; accounting_dropped=0; gold_lot_coverage_new=reported only; gold_lot_coverage_old=reported only; gold_lot_missing_new=reported only... |  |
| A8 | APRUEBA | production_untouched=sí; baseline_production_untouched=sí | production_untouched=sí; baseline_production_untouched=sí |  |
| A9 | NO APRUEBA | pytest_green=no; no_real_addresses_exit=0 | pytest_green=sí; no_real_addresses_exit=0 | pytest_green debe ser sí |
| A10 | NO APRUEBA | share_of_rows@0.7=0.9792; predial_lot_precision_emitted@0.7=0.9881; n_emitting_predial@0.7=252; manzana_error_rate@0.7=0.0071 | share_of_rows@0.7=0.9519; predial_lot_precision_emitted@0.7=0.9910; n_emitting_predial@0.7=reported only; manzana_error_rate@0.7=0.0086 | predial_lot_precision_emitted@0.7=0.9881 no cumple >= 0.9910 |
| A11 | APRUEBA | quarantine_share=0.0016; build_index_exit=0; download_exit=0; anomaly_fixtures_ok=sí | quarantine_share=0.0050; build_index_exit=0; download_exit=0; anomaly_fixtures_ok=sí |  |
| A12 | NO APRUEBA | A1_with_correct=283; A2_with_correct=0.9881; A3_with_correct=frozen_gold_ok=no evaluado; frozen_gold_predial_lot_precision=no evaluado; A4_with_correct=121; A5_with_correct=0.8609; A10_with_correct=share_of_rows@0.7=0.9792; predial_lot_p... | A1_with_correct=277; A2_with_correct=0.9960; A3_with_correct=frozen_gold_ok=398; frozen_gold_predial_lot_precision=0.9565; A4_with_correct=112; A5_with_correct=0.8580; A10_with_correct=share_of_rows@0.7=0.9519; predial_lot_precision_emit... | A2_with_correct no cumple su umbral; A4_with_correct no cumple su umbral; A10_with_correct no cumple su umbral |

Lectura en lenguaje llano de lo que no aprueba (qué cae dentro del ruido y qué no):

- **A2**: El candidato tiene 3 lotes erróneos de 252 filas gold que emiten predial (precisión 0.9881) frente a la línea base (0.9960). Bootstrap pareado: diferencia -0.0079, IC 95 % [-0.0202 a 0.0001], P(diferencia < 0) = 0.9010: dentro del ruido de muestreo (el intervalo contiene el cero). No está dentro de la dispersión entre semillas: ambas quedan por debajo de la línea base (0.9880 y 0.9882), una brecha estable de pocas filas, no un efecto de semilla.
- **A4**: dev lost_correct 121 frente al límite 112. Bootstrap pareado: diferencia +13, IC 95 % [-5 a 31]: dentro del ruido de muestreo. Las dos semillas del candidato dan 115 y 126 (difieren en 11), así que el valor queda dentro de la dispersión entre semillas.
- **A6**: Esto no es una cuestión de ruido: la regla de envío frente al refit sin restricción da falso, mientras que el ECE fuera de muestra de ambas cabezas (predio 0.0044, manzana 0.0022) está muy por debajo de los límites (0.1075 y 0.0438). Es una decisión de criterio (sección 20).
- **A9**: Esto no es una cuestión de ruido: el único test que falla preexiste y pertenece a otro proyecto de la misma carpeta; pytest: 6533 pruebas pasan y 1 falla. `check_no_real_addresses.py` sale con código 0. Es una decisión de criterio (sección 20).
- **A10**: Es la misma magnitud que A2 evaluada con confianza >= 0.70. El candidato tiene 3 lotes erróneos de 252 filas gold que emiten predial (precisión 0.9881) frente a la línea base (0.9960). Bootstrap pareado: diferencia -0.0079, IC 95 % [-0.0202 a 0.0001], P(diferencia < 0) = 0.9010: dentro del ruido de muestreo (el intervalo contiene el cero). No está dentro de la dispersión entre semillas: ambas quedan por debajo de la línea base (0.9880 y 0.9882), una brecha estable de pocas filas, no un efecto de semilla.
- **A12**: La corrección de salidas en dev no cambia nada (0 violaciones restantes, métricas idénticas); el fallo se hereda de A2, A4 y A10. La parte congelada no se reevalúa por diseño (la congelada no se vuelve a puntuar con `validate_outputs=correct`).

## 20. Decisiones abiertas para el usuario

Estas decisiones son del usuario; el informe solo expone los hechos. Ninguna se tomó automáticamente.

1. **A2, tolerancia cero.** A2 exige que la precisión de lote en dev gold sea >= la de la línea base (0.9960) sin ninguna tolerancia. El candidato da 0.9881 (diferencia -0.0079; una fila gold son 0.346 puntos). El IC del bootstrap es [-0.0202 a 0.0001]. ¿Se mantiene la tolerancia cero (no aprueba) o se acepta una tolerancia de una o dos filas gold, coherente con la banda de ruido?
2. **A6, regla de envío frente al refit sin restricción.** El ECE fuera de muestra del candidato es muy bajo (0.0044 y 0.0022), pero la regla de envío contra el refit sin restricción de la línea base da falso. ¿Se sigue exigiendo esa regla o basta el límite de ECE?
3. **A9, causada por un proyecto ajeno.** A9 falla solo por una prueba de otro proyecto de la misma carpeta; pytest: 6533 pruebas pasan y 1 falla. ¿Se cuenta como aprobado el criterio con esa excepción documentada, o se corrige primero el proyecto ajeno?
4. **Pérdida de precisión manzana en filas congeladas ruidosas.** En la prueba congelada con todas las filas (SEGUNDA MIRADA) la precisión manzana baja de 0.7314 a 0.7121 (diferencia -0.0192, IC 95 % [-0.0366 a -0.0027]), una caída que se distingue del ruido de muestreo, mientras que en dev A5 sí se cumple y la cobertura de filas OK sube. A5 se evalúa solo en dev y no cubre esas filas congeladas. ¿Es aceptable esa pérdida a cambio de la cobertura, o debe fijarse un criterio para las filas congeladas ruidosas?

## 21. Promoción y reversibilidad del respaldo de catastro

Veredicto calculado: NO APRUEBA.

Estado de promoción: promoción pendiente de aprobación explícita del usuario. Este informe no promueve nada a `artifacts/`.

Reversibilidad del respaldo de catastro (tres mecanismos independientes, con pruebas automáticas; ver `docs/respaldo-catastro.md`):

1. **El índice base y los experimentos base nunca se modifican.** El respaldo vive en directorios nuevos y el modelo se reutiliza sin cambios; servir el índice base es seguir usando el índice base (`index_v2e_cert_pcer_r3`).
2. **Retirar las filas agregadas** (índice y directorio de experimento): `.venv/Scripts/python.exe scripts/index_fallback.py remove --index-dir <directorio con respaldo> --out-dir <directorio nuevo>`. Verifica cada archivo contra el manifiesto del índice base; si algo no es idéntico byte a byte, falla y borra la salida.
3. **Interruptor en tiempo de servicio, sin reconstruir.** `"exclude_fuentes": ["catastro"]` en `decision.json` y/o la variable de entorno `NORMALIZER_EXCLUDE_FUENTES=catastro` (se combinan). El servicio atiende exactamente el índice sin esos documentos; se aplica una vez por objeto normalizador y, para cambiarlo, se crea uno nuevo.
