# Ejercicio 1 — Backpropagación genérico probado con XOR (XOR_trn.csv / XOR_tst.csv)


Parámetros fijos: `seed=42`, `b=1` (ganancia sigmoide), `vel_aprendizaje=0.1`,
`max_epocas=500`, `err_umbral=0.05` (único criterio de corte, sin OR por
accuracy). Dataset: 2000 patrones de entrenamiento, 200 de prueba (XOR con
ruido gaussiano alrededor de los 4 puntos ±1). Semilla fija para que los
pesos iniciales (y por lo tanto los resultados) sean reproducibles entre
corridas.

## Resultados

| Arquitectura | ¿Convergió? | Época | Aciertos test |
|---|---|---|---|
| [2,2,1] | Sí | 160 | 100.0% (200/200) |
| [2,4,1] | Sí | 141 | 100.0% (200/200) |
| [2,3,3,1] | Sí | 36 | 100.0% (200/200) |

## Explicación

Las 3 arquitecturas resolvieron XOR sin problema, a diferencia del perceptrón
simple de la Guía 1 (que no puede porque XOR no es linealmente separable). Con
más neuronas/capas la convergencia es más rápida en épocas (36 para
`[2,3,3,1]` vs. 160 para `[2,2,1]`), lo cual tiene sentido: más parámetros
significa más grados de libertad para bajar el error cuadrático total más
rápido por época, aunque acá no importa mucho porque las tres son igual de
rápidas en tiempo real (XOR es un problema chico).

**Arquitectura elegida: `[2,2,1]`** — es la mínima que resuelve el problema
(2 neuronas ocultas, el mínimo teórico para separar los 4 puntos de XOR con
una sola capa oculta), y es la que quedó en el script.
