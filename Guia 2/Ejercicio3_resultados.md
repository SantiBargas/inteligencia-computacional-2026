# Ejercicio 3 — Comparación de arquitecturas y tasas de aprendizaje (iris81_trn.csv / iris81_tst.csv)

Parámetros fijos: `seed=42` (fijada una vez antes de cada corrida individual,
así cada fila de la tabla es reproducible por separado), `b=1` (ganancia
sigmoide), `max_epocas=3000`. 


## Resultados

| Arquitectura | vel. aprendizaje | Aciertos train | Aciertos test | Convergencia |
|---|---|---|---|---|
| [4,3,3] | 0.001 | 71.2% (79/111) | 70.3% (26/37) | no convergió (3000 épocas) |
| [4,3,3] | 0.01 | 99.1% (110/111) | 94.6% (35/37) | época 493 |
| [4,3,3] | 0.1 | 99.1% (110/111) | **100.0% (37/37)** | época 63 |
| [4,3,3] | 0.5 | 69.4% (77/111) | 70.3% (26/37) | no convergió (3000 épocas) |
| [4,5,3] | 0.001 | 99.1% (110/111) | 91.9% (34/37) | época 772 |
| [4,5,3] | 0.01 | 99.1% (110/111) | 91.9% (34/37) | época 80 |
| [4,5,3] | 0.1 | 99.1% (110/111) | **100.0% (37/37)** | época 63 |
| [4,5,3] | 0.5 | 69.4% (77/111) | 70.3% (26/37) | no convergió (3000 épocas) |
| [4,8,3] | 0.001 | 99.1% (110/111) | 91.9% (34/37) | época 399 |
| [4,8,3] | 0.01 | 99.1% (110/111) | 91.9% (34/37) | época 70 |
| [4,8,3] | 0.1 | 99.1% (110/111) | **100.0% (37/37)** | época 59 |
| [4,8,3] | 0.5 | 69.4% (77/111) | 70.3% (26/37) | no convergió (3000 épocas) |
| [4,5,5,3] | 0.001 | 99.1% (110/111) | 91.9% (34/37) | época 581 |
| [4,5,5,3] | 0.01 | 99.1% (110/111) | 91.9% (34/37) | época 80 |
| [4,5,5,3] | 0.1 | 99.1% (110/111) | 97.3% (36/37) | época 86 |
| [4,5,5,3] | 0.5 | 0.0% (0/111) | 0.0% (0/37) | no convergió (3000 épocas) |


## Curvas de entrenamiento (ξ y error de clasificación)

Con `lr=0.1` se regeneró, para cada una de las 4 arquitecturas, el gráfico de
evolución por época de dos curvas distintas (archivos
`Ejercicio3_historial_<arquitectura>.png`), ahora truncadas en la época de
convergencia real (59-86 según la arquitectura) en vez de extenderse a las
3000 épocas completas:

- **ξ = ½·Σe²** (panel izquierdo): mide qué tan cerca está cada salida
  continua de la red del valor objetivo exacto (±1). Es la función de costo
  que minimiza el gradiente descendente.
- **Error de clasificación** (panel derecho): mide si `signo(salida)`
  coincide elemento a elemento con el código de la clase objetivo (las 3
  posiciones tienen que tener el signo correcto) — un criterio binario de
  acierto/error que no le importa la magnitud, solo el signo de cada salida.

Ambas curvas se calculan sobre los **mismos** patrones de train en la misma
época — la diferencia no es el dato, es qué miden sobre las salidas de la
red. El error de clasificación cae a casi 0 y se aplana rápido (primeras
~30-50 épocas), mientras que ξ todavía sigue bajando cuando se corta el
entrenamiento — el gradiente sigue empujando las salidas a acercarse más a
±1 exactos, no solo a que tengan el signo correcto. 

