# Ejercicio 2 — Comparación de arquitecturas (concent_trn.csv / concent_tst.csv)

Parámetros fijos: `seed=42`, `b=1` (ganancia sigmoide), `max_epocas=3000`,
`err_umbral=0.05` (criterio de corte, además del 95% de aciertos en train).
Para cada arquitectura se muestra la mejor tasa de aprendizaje encontrada.

## Resultados

| Arquitectura | vel. aprendizaje | ¿Convergió? | Época | Aciertos train | Aciertos test |
|---|---|---|---|---|---|
| [2,4,1] | 1.0 | No | 3000 (agotadas) | 89.1% | 88.8% |
| [2,8,1] | 0.1 | No | 3000 (agotadas) | 63.1% | 63.1% |
| [2,4,4,4,1] | 0.1 | No | 3000 (agotadas) | 63.1% | 63.1% |
| [2,6,1] | 1.0 | Sí | 922 | 95.1% | 92.2% |
| [2,10,1] | 0.01 | Sí | 223 | 95.1% | 95.6% |
| [2,8,8,1] | 0.1 | Sí | 265 | 95.1% | 94.5% |
| [2,4,3,1] | 0.1 | Sí | 587 | 95.1% | 95.2% |
| [2,4,4,1] | 0.1 | Sí | 1094 | 95.1% | 95.4% |
| [2,6,6,6,1] | 0.1 | Sí | 1420 | 95.1% | 96.1% |
| [2,5,5,1] | 0.1 | Sí | 227 | 95.2% | 96.4% |
| [2,6,6,1] | 0.5 | Sí | 112 | 95.3% | 97.3% |
| **[2,3,3,1]** | **0.01** | **Sí** | 862 | 95.7% | **97.6%** |

## Explicación

Con **una sola capa oculta** el problema es teóricamente representable (el
disco central es una región convexa, y una capa oculta alcanza para eso según
la teoría de regiones de decisión), pero la convergencia depende mucho de la
tasa de aprendizaje: con `lr=0.1` casi todas quedaron estancadas en 63.1%

Con **dos capas ocultas** la convergencia fue mucho más robusta: todas
convergieron ya con `lr=0.1`, sin necesitar ajuste, y dieron los mejores
resultados en test. Con **tres capas ocultas** el resultado fue mixto.

**Arquitectura recomendada: [2,3,3,1]** con `lr=0.01` — es la más chica de las
que funcionan bien (6 neuronas ocultas en total) y logró en test (97.6%).
