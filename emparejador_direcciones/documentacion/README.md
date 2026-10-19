# Documentación de emparejador_direcciones

Esta carpeta documenta en español el comparador determinista de direcciones `emparejador_direcciones`. Los resultados de las corridas de evaluación no se publican; la documentación describe protocolos, definiciones y comandos para reproducirlos localmente (ver [08](08-metricas-y-confiabilidad.md)).

**Aviso:** el modelo no es una red neuronal. No hay entrenamiento ni pesos. Se explica en [01-teoria.md](01-teoria.md).

## Índice

| # | Documento | Para quién | Pregunta que responde |
|---|---|---|---|
| 01 | [Teoría](01-teoria.md) | Quien quiera entender el método | ¿Cómo decide si dos direcciones son la misma? |
| 02 | [Arquitectura](02-arquitectura.md) | Desarrolladores | ¿Qué módulos hay, cómo fluyen los datos y cuál es la API? |
| 03 | [Configuración](03-configuracion.md) | Operadores | ¿Qué parámetros existen y qué efecto tienen? |
| 04 | [Evaluación y calibración](04-evaluacion-y-calibracion.md) | Quien adapte el modelo a otra ciudad | ¿Cómo se mide la calidad y cómo se repite con mi catastro? |
| 05 | [Uso e implementación](05-uso-e-implementacion.md) | Integradores | ¿Cómo lo instalo, lo uso y lo pongo en un servicio? |
| 06 | [Predicciones](06-predicciones.md) | Usuarios del resultado | ¿Qué entrada exige, qué devuelve y cómo interpretarlo? |
| 07 | [Requisitos](07-requisitos.md) | Infraestructura | ¿Qué software, hardware y datos necesito? |
| 08 | [Métricas y confiabilidad](08-metricas-y-confiabilidad.md) | Quien decida si confiar en el resultado | ¿Qué se midió, qué no, y cuánto confiar? |
| 09 | [Advertencias y limitaciones](09-advertencias-y-limitaciones.md) | Todos | ¿Qué puede salir mal? |
| 10 | [Estándar y generalización](10-estandar-y-generalizacion.md) | Quien lo use fuera de Cali | ¿Cumple el estándar nacional y cómo se adapta? |

## Lectura sugerida

- Para usarlo ya: [README principal](../README.md), luego 05, 06 y 09.
- Para decidir si confiar en él: 08, luego 09.
- Para llevarlo a otra ciudad o país: 10, luego 04.

## Convenciones

- Los identificadores de código, nombres de campos, motivos (`reason`) y avisos (`warnings`) se muestran tal como están en el código fuente.
- Todas las direcciones de los ejemplos son inventadas (por ejemplo `CL 5 # 10 - 20`) y no corresponden a predios reales.
- Las salidas pegadas en los ejemplos se obtuvieron ejecutando los scripts del proyecto sobre esas direcciones inventadas.
