import numpy as np
import matplotlib.pyplot as plt

def funcActivacion(result):
    return 1 if result >= 0 else -1

def CargarDatosCSV(nombre_archivo):
    #Cargamos el csv y separamos las entradas (x) de las salidas deseadas (y)
    datos = np.loadtxt(nombre_archivo, delimiter=",")
    x = datos[:, :-1] #todas las columnas menos la última
    y = datos[:, -1] #la ultima columna

    #bias: sirve para que la red no pase siempre necesariamente por el origen, sino que pueda desplazarse a lo largo del espacio de entrada.
    bias = -np.ones((x.shape[0], 1)) # vector columna de -1 con la misma cant de filas que x
    x_extendido = np.hstack((bias, x)) # concatenamos el bias con las entradas originales
    return x_extendido, y


def GraficarPatrones(x, y, titulo):
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 1], x[:, 2], c=-y, cmap='bwr', edgecolor='k', label='Patrones')
    plt.ylim(x[:, 2].min() - 1, x[:, 2].max() + 1)   # limite para que las rectas no rompan la escala
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(titulo)
    plt.grid(True)


def dibujar_recta(pesos, color, alpha, label=None):
    x_vals = np.linspace(-2, 2, 100)
    y_vals = (pesos[0] - pesos[1] * x_vals) / pesos[2]
    plt.plot(x_vals, y_vals, color=color, alpha=alpha, label=label)


def Entrenamiento(x_extendido, y, tasa_aprendizaje, max_epocas):
    pesos = np.random.uniform(-0.5,0.5, x_extendido.shape[1]) #inicializamos pesos aleatorios entre -0.5 y 0.5 w0 w1 w2 uno para cada entrada (incluido el bias)
    print("pesos iniciales:", pesos)

    dibujar_recta(pesos, color='red', alpha=1.0, label='Recta inicial (al azar)')

    n_patrones = x_extendido.shape[0] #cantidad de patrones de entrenamiento

    for epoca in range(1,max_epocas +1):
        errores_epoca = 0 #contador de errores por época
        for i in range(n_patrones):
            v = np.dot(x_extendido[i], pesos) #producto punto entre el patrón de entrada y los pesos
            y_obtenida = funcActivacion(v) # Función de activación signo
            error = y[i] - y_obtenida 

            if error != 0:
                pesos = pesos + (tasa_aprendizaje/2) * error * x_extendido[i] 
                errores_epoca += 1

        print(f"Época {epoca}: errores = {errores_epoca}")
        dibujar_recta(pesos, color="black", alpha=0.1)

        if errores_epoca == 0:
            print("Entrenamiento completo, no hay errores en la época", epoca)
            break
    
    dibujar_recta(pesos, color="green", alpha=1.0, label="Recta final")  #ultima recta de decision en verde
    return pesos, epoca, errores_epoca


def Test(x_extendido, y, pesos):
    #Paso 4: utilizamos los pesos entrenados para clasificar y ver si la red aprendio correctamente
    predicciones = []
    for i in range(x_extendido.shape[0]):
        v = np.dot(x_extendido[i], pesos)
        if v >= 0:
            predicciones.append(1)
        else:
            predicciones.append(-1)
    predicciones = np.array(predicciones)

    comparacion = predicciones == y
    aciertos = np.sum(comparacion)
    total = len(y)
    print(f"Aciertos en test: {aciertos}/{total} ({100*aciertos/total:.2f}%)")
    return aciertos, total


##LLAMADAS - OR
print("\n===== OR =====")
x_train, y_train = CargarDatosCSV("OR_trn.csv")
GraficarPatrones(x_train, y_train, "OR")
pesos_entrenados, epoca_final, errores_finales = Entrenamiento(x_train, y_train, tasa_aprendizaje=0.01, max_epocas=100)
x_test, y_test = CargarDatosCSV("OR_tst.csv")
Test(x_test, y_test, pesos_entrenados)
plt.legend()
plt.show()

##LLAMADAS - OR_50
print("\n===== OR_50 =====")
x_train, y_train = CargarDatosCSV("OR_50_trn.csv")
GraficarPatrones(x_train, y_train, "OR_50")
pesos_entrenados, epoca_final, errores_finales = Entrenamiento(x_train, y_train, tasa_aprendizaje=0.01, max_epocas=100)
x_test, y_test = CargarDatosCSV("OR_50_tst.csv")
Test(x_test, y_test, pesos_entrenados)
plt.legend()
plt.show()

##LLAMADAS - OR_90
print("\n===== OR_90 =====")
x_train, y_train = CargarDatosCSV("OR_90_trn.csv")
GraficarPatrones(x_train, y_train, "OR_90")
pesos_entrenados, epoca_final, errores_finales = Entrenamiento(x_train, y_train, tasa_aprendizaje=0.01, max_epocas=500)
x_test, y_test = CargarDatosCSV("OR_90_tst.csv")
Test(x_test, y_test, pesos_entrenados)
plt.legend()
plt.show()

#Utilizando el metodo de graficar como en el ejercicio 2 podemos ver que para el OR_50 a pesar de mayor dispersión con respecto al OR original, los patrones siguen siendo linealmente separables, y el entrenamiento converge y su test es  muy bueno

#En cambio para el OR_90 la red en el entrenamiento no converge ya que no podemos encontrar una recta que resuelva el problema, aunque da una tasa alta de acierto entre un 90% y 94% aproximadamente.