# Cómo citar y atribuir HIIG

HIIG es software libre bajo Apache License 2.0. Puede usarse, modificarse y
redistribuirse (incluso con fines comerciales) siempre que se conserve el
crédito a su autor, Juanpgm.

## Qué deben conservar quienes redistribuyan

- El archivo `LICENSE` (Apache-2.0).
- El archivo `NOTICE` o su contenido (sección 4(d)): crédito a Juanpgm, fuentes
  de datos y advertencia sobre datos de terceros y pesos del modelo.
- Los avisos de derechos de autor y atribución presentes en los archivos
  fuente.
- Una indicación visible de los archivos que usted haya modificado.

No se permite usar el nombre del autor para avalar productos derivados sin
permiso (sección 6 de la licencia).

## Cómo citar

Formato legible por máquina: ver `CITATION.cff` (GitHub muestra el botón
"Cite this repository").

Texto sugerido:

> Juanpgm (2026). HIIG: Herramientas de Inferencia e Integración Geográfica
> (versión 0.2.0) [Software]. https://github.com/Juanpgm/HIIG. Apache-2.0.

BibTeX:

```bibtex
@software{juanpgm_hiig_2026,
  author  = {Juanpgm},
  title   = {{HIIG}: Herramientas de Inferencia e Integración Geográfica},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/Juanpgm/HIIG},
  license = {Apache-2.0}
}
```

## Texto breve para aplicaciones y sitios web

> Construido con HIIG, de Juanpgm (https://github.com/Juanpgm/HIIG), licencia
> Apache-2.0.

Insignia en Markdown:

```markdown
[![Hecho con HIIG](https://img.shields.io/badge/hecho%20con-HIIG%20%C2%B7%20Juanpgm-blue)](https://github.com/Juanpgm/HIIG)
```

## Convención de encabezados SPDX (opcional)

En archivos fuente nuevos se recomienda esta cabecera de dos líneas, en
inglés:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juanpgm
```

No es obligatoria: la licencia aplica a todo el repositorio por `LICENSE` y
`NOTICE`. Los archivos de terceros conservan sus propias cabeceras.

## Datos y pesos de modelo

La licencia no cubre datos de terceros ni garantiza derechos sobre ellos; vea
`NOTICE`.
