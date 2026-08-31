# Ejercicio 1 — Backpropagación genérico probado con XOR (XOR_trn.csv / XOR_tst.csv)


Parámetros fijos: `b=1` (ganancia sigmoide), `vel_aprendizaje=0.1`,
`max_epocas=500`, `err_umbral=0.05` (único criterio de corte, sin OR por
accuracy). Dataset: 2000 patrones de entrenamiento, 200 de prueba (XOR con
ruido gaussiano alrededor de los 4 puntos ±1). Sin semilla fija, así que los
pesos iniciales cambian en cada corrida.

## Resultados

| Arquitectura | ¿Convergió? | Época | Aciertos test |
|---|---|---|---|
| [2,2,1] | Sí | 159 | 100.0% (200/200) |
| [2,4,1] | Sí | 79 | 100.0% (200/200) |
| [2,3,3,1] | Sí | 46 | 100.0% (200/200) |

## Explicación

Las 3 arquitecturas resolvieron XOR sin problema, a diferencia del perceptrón
simple de la Guía 1 (que no puede porque XOR no es linealmente separable). Con
más neuronas/capas la convergencia es más rápida en épocas (46 para
`[2,3,3,1]` vs. 159 para `[2,2,1]`), lo cual tiene sentido: más parámetros
significa más grados de libertad para bajar el error cuadrático total más
rápido por época, aunque acá no importa mucho porque las tres son igual de
rápidas en tiempo real (XOR es un problema chico).

**Arquitectura elegida: `[2,2,1]`** — es la mínima que resuelve el problema
(2 neuronas ocultas, el mínimo teórico para separar los 4 puntos de XOR con
una sola capa oculta), y es la que quedó en el script.
