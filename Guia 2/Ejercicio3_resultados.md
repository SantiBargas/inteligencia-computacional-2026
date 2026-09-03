# Ejercicio 3 — Comparación de arquitecturas y tasas de aprendizaje (iris81_trn.csv / iris81_tst.csv)

Parámetros fijos: `seed=42` (fijada una vez antes de cada corrida individual,
así cada fila de la tabla es reproducible por separado), `b=1` (ganancia
sigmoide), `max_epocas=3000`, criterio de corte por `tasa_aciertos_train >=
1.0` (sin corte anticipado en 95%, para no limitar artificialmente el
resultado en train). Clasificación decidida por `signo()` elemento a elemento
sobre las 3 salidas. Dataset: 111 patrones de entrenamiento, 37 de prueba, 4
entradas, 3 salidas (código binario de especie: setosa=[-1,-1,1],
versicolor=[-1,1,-1], virginica=[1,-1,-1]).

## Resultados

| Arquitectura | vel. aprendizaje | Aciertos train | Aciertos test |
|---|---|---|---|
| [4,3,3] | 0.001 | 71.2% | 70.3% (26/37) |
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
| [4,5,5,3] | 0.5 | 2.7% | 0.0% (0/37) |

Ninguna de las 16 corridas convergió formalmente (`tasa_aciertos_train >=
1.0`), así que todas usaron las 3000 épocas completas. En 14 de las 16, el
techo real de train quedó en 99.1% (1 solo patrón mal clasificado de 111);
las dos excepciones son `[4,3,3]` con `lr=0.001` (71.2%, quedó estancada en
una inicialización desfavorable) y `[4,5,5,3]` con `lr=0.5` (2.7%, colapsó
por un learning rate demasiado alto para una red de 3 capas ocultas).

## Curvas de entrenamiento (ξ y error de clasificación)

Con `lr=0.1` se generó, para cada una de las 4 arquitecturas, el gráfico de
evolución por época de dos curvas distintas (archivos
`Ejercicio3_historial_<arquitectura>.png`):

- **ξ = ½·Σe²** (panel izquierdo): mide qué tan cerca está cada salida
  continua de la red del valor objetivo exacto (±1). Es la función de costo
  que minimiza el gradiente descendente.
- **Error de clasificación** (panel derecho): mide si `signo(salida)`
  coincide elemento a elemento con el código de la clase objetivo (las 3
  posiciones tienen que tener el signo correcto) — un criterio binario de
  acierto/error que no le importa la magnitud, solo el signo de cada salida.

Ambas curvas se calculan sobre los **mismos** patrones de train en la misma
época — la diferencia no es el dato, es qué miden sobre las salidas de la
red. Por eso en los gráficos el error de clasificación cae a casi 0 y se
aplana rápido (~época 100-150), mientras que ξ sigue bajando mucho más
tiempo: el gradiente sigue empujando las salidas a acercarse más a ±1
exactos, no solo a que tengan el signo correcto. Es el mismo fenómeno que se
ve en el Ejercicio 1, donde XOR llegaba a 100% de aciertos mucho antes de
que `xi` bajara del umbral de corte.

Estas 4 curvas corresponden exactamente a las filas con `lr=0.1` de la tabla
de "Resultados" (mismo `seed=42`, mismo código): las 4 arquitecturas llegan
al mismo techo de 99.1% de aciertos train (error de clasificación 0.009, 1
patrón mal clasificado de 111) sin convergencia formal en 3000 épocas.

## Explicación

**`lr=0.1` da 100% de aciertos en test en las 4 arquitecturas, sin
excepción** — es el resultado más robusto del barrido y no depende de cuántas
neuronas o capas ocultas se elijan. Con `lr=0.01` el desempeño es más
inconsistente (94.6% siempre) y con `lr=0.001` depende mucho de la
arquitectura: 3 de las 4 dan 91.9%-99.1% en train, pero `[4,3,3]` quedó
estancada en una inicialización desfavorable (71.2% train, 70.3% test) — con
un `lr` tan bajo la red es más sensible a los pesos iniciales porque avanza
poco por época. Con `lr=0.5` el resultado empeora sistemáticamente, y cuanto
más profunda es la red peor es la caída: `[4,5,5,3]` colapsa por completo
(2.7% train, 0.0% test) — un learning rate demasiado alto desestabiliza mucho
más a las redes con más capas ocultas.

**Arquitectura y configuración recomendada: `[4,3,3]` con `vel_aprendizaje=0.1`**
— es la más simple de las probadas, llega al 100% en test, y es la única
tasa de aprendizaje que da el mismo resultado óptimo en las 4 arquitecturas
(no hace falta "afinar" el `lr` según el tamaño de la red).
