# Ejercicio 3 — Comparación de arquitecturas y tasas de aprendizaje (iris81_trn.csv / iris81_tst.csv)

Parámetros fijos: `seed=42`, `b=1` (ganancia sigmoide), `max_epocas=3000`,
criterio de corte por `tasa_aciertos_train >= 1.0` (sin corte anticipado en
95%, para no limitar artificialmente el resultado en train). Clasificación
decidida por `argmax` sobre las 3 salidas. Dataset: 111 patrones de
entrenamiento, 37 de prueba, 4 entradas, 3 salidas (código binario de
especie: setosa=[-1,-1,1], versicolor=[-1,1,-1], virginica=[1,-1,-1]).

## Resultados

| Arquitectura | vel. aprendizaje | Aciertos train | Aciertos test |
|---|---|---|---|
| [4,3,3] | 0.001 | 99.1% | **100.0% (37/37)** |
| [4,3,3] | 0.01 | 99.1% | 94.6% (35/37) |
| [4,3,3] | 0.1 | 99.1% | **100.0% (37/37)** |
| [4,3,3] | 0.5 | 69.4% | 70.3% (26/37) |
| [4,5,3] | 0.001 | 99.1% | 91.9% (34/37) |
| [4,5,3] | 0.01 | 99.1% | 94.6% (35/37) |
| [4,5,3] | 0.1 | 99.1% | **100.0% (37/37)** |
| [4,5,3] | 0.5 | 69.4% | 70.3% (26/37) |
| [4,8,3] | 0.001 | 99.1% | 91.9% (34/37) |
| [4,8,3] | 0.01 | 99.1% | 94.6% (35/37) |
| [4,8,3] | 0.1 | 99.1% | **100.0% (37/37)** |
| [4,8,3] | 0.5 | 69.4% | 70.3% (26/37) |
| [4,5,5,3] | 0.001 | 99.1% | 91.9% (34/37) |
| [4,5,5,3] | 0.01 | 99.1% | 94.6% (35/37) |
| [4,5,5,3] | 0.1 | 99.1% | **100.0% (37/37)** |
| [4,5,5,3] | 0.5 | 37.8% | 29.7% (11/37) |

Ninguna corrida convergió formalmente al 100% en train (99.1% fue el techo
real, salvo con `lr=0.5`), así que las 16 corridas usaron las 3000 épocas
completas.

## Explicación

**`lr=0.1` da 100% de aciertos en test en las 4 arquitecturas, sin
excepción** — es el resultado más robusto del barrido y no depende de cuántas
neuronas o capas ocultas se elijan. Con `lr=0.001` o `lr=0.01` el desempeño
es más inconsistente (91.9%-94.6%) y depende de la arquitectura. Con `lr=0.5`
el resultado empeora sistemáticamente, y cuanto más profunda es la red peor
es la caída (29.7% en `[4,5,5,3]`) — un learning rate demasiado alto
desestabiliza más a las redes con más capas.

**Arquitectura y configuración recomendada: `[4,3,3]` con `vel_aprendizaje=0.1`**
— es la más simple de las probadas, llega al 100% en test, y es la única
tasa de aprendizaje que da el mismo resultado óptimo en las 4 arquitecturas
(no hace falta "afinar" el `lr` según el tamaño de la red).
