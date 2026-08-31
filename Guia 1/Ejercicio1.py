import numpy as np

#Paso 1: Cargamos el csv y separamos las entradas (x) de las salidas deseadas (y)
datos = np.loadtxt("OR_trn.csv", delimiter=",")
x=datos[:, :-1] #todas las columnas menos la última
y=datos[:, -1] #la ultima columna   

print(x.shape,y.shape)

#Paso 2: Armamos el vector de entradas extendidos con x0=-1 (bias)

#bias: sirve para que la red no pase siempre necesariamente por el origen, sino que pueda desplazarse a lo largo del espacio de entrada.
bias = -np.ones((x.shape[0], 1)) # vector columna de -1 con la misma cant de filas que x
x_extendido = np.hstack((bias, x)) # concatenamos el bias con las entradas originales

print(x_extendido[:3])   # chequeo: cada fila debe arrancar con -1

#Paso 3: Empezamos a iterar para entrenar

def entrenar(x_extendido, y, tasa_aprendizaje, max_epocas):
    pesos = np.random.uniform(-0.5,0.5, x_extendido.shape[1]) #inicializamos pesos aleatorios entre -0.5 y 0.5 w0 w1 w2 uno para cada entrada (incluido el bias)
    print("pesos iniciales:", pesos)

    n_patrones = x_extendido.shape[0] #cantidad de patrones de entrenamiento

    for epoca in range(1,max_epocas +1):
        errores_epoca = 0 #contador de errores por época
        for i in range(n_patrones):
            v = np.dot(x_extendido[i], pesos) #producto punto entre el patrón de entrada y los pesos
            y_obtenida = 1 if v >= 0 else -1 #función de activacion signo
            error = y[i] - y_obtenida 

            if error != 0:
                pesos = pesos + (tasa_aprendizaje/2) * error * x_extendido[i] 
                errores_epoca += 1

        print(f"Época {epoca}: errores = {errores_epoca}")

        if errores_epoca == 0:
            print("Entrenamiento completo, no hay errores en la época", epoca)
            break

    return pesos, epoca, errores_epoca


pesos_entrenados, epoca_final, errores_finales = entrenar(x_extendido, y, tasa_aprendizaje=0.01, max_epocas=100)


#Paso 4 utilizamos los pesos entrenados para clasificar y ver si la red aprendio correctamente

datos_test = np.loadtxt("OR_tst.csv", delimiter=",")
x_test = datos_test[:, :-1]
y_test = datos_test[:, -1]

bias_test = -np.ones((x_test.shape[0], 1))
x_test_extendido = np.hstack((bias_test, x_test))

predicciones_test = []
for i in range(x_test_extendido.shape[0]):
    v = np.dot(x_test_extendido[i], pesos_entrenados)
    if v >= 0:
        predicciones_test.append(1)
    else:
        predicciones_test.append(-1)
predicciones_test = np.array(predicciones_test)


comparacion = predicciones_test == y_test
aciertos_test = np.sum(comparacion)
total_test = len(y_test)

print(f"Aciertos en test: {aciertos_test}/{total_test} ({100*aciertos_test/total_test:.2f}%)")


"""
Problema del OR
(-1, -1) → -1     (falso OR falso = falso)
(-1, +1) → +1     (falso OR verdadero = verdadero)
(+1, -1) → +1     (verdadero OR falso = verdadero)
(+1, +1) → +1     (verdadero OR verdadero = verdadero)
"""

"""
Patrones y épocas en el entrenamiento
Cada fila del CSV extendido es un patrón: -1 (bias) + entradas reales.
1 época = una pasada completa por los 2000 patrones de entrenamiento.
"""

#Utilizamos el metodo de entrenamiento por correccion de error ya que utilizamos una funcion de activacion signo (para el metodo de gradiente necesitamos una funcion de activacion continua y derivable). Si la salida es correcta no hacemos cambios. Si la salida es incorrecta actualizamos los pesos en el sentido opuesto que contribuyeron

#El metodo de entrenamiento converge muy facil porque tenemos un problema sencillo (OR) y los datos solo tienen pequeñas desviaciones aleatorias (<5%)