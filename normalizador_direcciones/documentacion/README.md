# Documentación del normalizador de direcciones de Cali

Índice de la documentación en español del modelo `normalizador_direcciones`. El estado del proyecto es **versión preliminar probada con datos de Cali**. Empiece por el [README principal](../README.md) para el alcance, la política de artefactos y el inicio rápido.

> Aviso de publicación: los resultados de ejecución citados como fuentes en este documento (`strict_eval_*`, `eval_calibration.csv`, `threshold_*.csv`, `splits_report.json`, `train_report.json`, `artifacts/experiments/`) ya no se versionan en el repositorio. Las cifras no se publican; para obtenerlas se regeneran localmente con los scripts de `scripts/` (ver [04-entrenamiento.md](04-entrenamiento.md)), y no hay garantía de que coincidan bit a bit.

## Cómo leer esta documentación

| Si usted quiere... | Lea |
|--------------------|-----|
| Entender cómo funciona y por qué | [01-teoria.md](01-teoria.md), luego [02-arquitectura.md](02-arquitectura.md) |
| Usar el sistema sobre sus datos | [05-uso-e-implementacion.md](05-uso-e-implementacion.md), luego [06-predicciones.md](06-predicciones.md) |
| Saber cuánto confiar en los resultados | [08-metricas-y-confiabilidad.md](08-metricas-y-confiabilidad.md) y [09-advertencias-y-limitaciones.md](09-advertencias-y-limitaciones.md) |
| Reentrenar el modelo | [04-entrenamiento.md](04-entrenamiento.md) y [07-requisitos.md](07-requisitos.md) |
| Configurar la CLI, la API o el despliegue | [03-configuracion.md](03-configuracion.md) |
| Adaptarlo a otra ciudad o país | [10-estandar-y-generalizacion.md](10-estandar-y-generalizacion.md) |

## Índice

| # | Documento | Contenido |
|--:|-----------|-----------|
| 01 | [Fundamentos teóricos](01-teoria.md) | Codificador dual, aprendizaje contrastivo (InfoNCE), modelo a nivel de carácter, corrupción sintética, negativos duros espaciales, cabezas multitarea, puntaje de fusión y decaimiento geográfico, reglas estrictas, `confianza` frente a `confiabilidad` |
| 02 | [Arquitectura](02-arquitectura.md) | Mapa de módulos; diagrama de la red, formas de tensor y conteo de parámetros; canalización de datos de entrenamiento; flujo de inferencia; diagrama de despliegue; superficies de uso |
| 03 | [Configuración](03-configuracion.md) | Variables de entorno, opciones de la CLI, TOML, parámetros de decisión, campos de `tuning.json` y `reliability.json`, hiperparámetros, Docker y Railway |
| 04 | [Entrenamiento](04-entrenamiento.md) | Esquema de datos, etiquetas del NPN, receta paso a paso, hiperparámetros, hardware y tiempos, semillas, ajuste de umbral y de confiabilidad, salidas |
| 05 | [Uso e implementación](05-uso-e-implementacion.md) | Instalación, CLI, formatos de entrada y salida, API con ejemplos, Esri, Docker y Railway, rendimiento, seguridad |
| 06 | [Predicciones](06-predicciones.md) | Formatos de entrada, esquema de salida campo por campo, valores de `estado` y `motivo`, `nivel_precision`, guía de interpretación con ejemplos |
| 07 | [Requisitos](07-requisitos.md) | Requisitos para entrenamiento y para predicción, esquemas de datos, artefactos al cargar, política de publicación, elementos específicos de Cali |
| 08 | [Métricas y confiabilidad](08-metricas-y-confiabilidad.md) | Definición de métricas, protocolo, cómo reproducirlas, escala operativa, lo no medido |
| 09 | [Advertencias y limitaciones](09-advertencias-y-limitaciones.md) | 23 advertencias numeradas, inventario de elementos específicos de Cali con archivo y línea, inconsistencias de la documentación original |
| 10 | [Estándar y generalización](10-estandar-y-generalizacion.md) | Tabla de conformidad con el estándar nacional, lista de adaptación a otros territorios, protocolo de validación, reutilización de `model.pt` |

Anexo: [Registro de cambios de confiabilidad](CHANGELOG_confiabilidad.md) (procedencia del calibrador servido, causa de su baja discriminación y calibrador v5).

## Documentos originales del proyecto

Se conservan en [../docs/](../docs): `API.md` (contrato de la API), `ESRI_INTEGRATION.md`, `DEPLOY.md` (despliegue en Railway), `model-artifacts.md`, `experiments.md`, `splits.md`, `any-dataset.md` y `contributing.md`. Donde esas páginas y esta serie discrepan, prevalece esta serie, que verificó cada afirmación contra el código y los archivos (las discrepancias conocidas están en [09](09-advertencias-y-limitaciones.md), sección 3).

## Convenciones

- Los ejemplos de direcciones, números prediales y respuestas son inventados.
- Los identificadores de código, nombres de columnas y valores de `estado` se muestran como en el código fuente.
- Los resultados de las corridas (cifras de rendimiento e intervalos de confianza) no se publican; la documentación describe cómo obtenerlos localmente.
- Las recomendaciones se rotulan como tales y se distinguen de los hechos verificados.

Licencia: por definir (ver LICENSE en la raíz cuando exista).
