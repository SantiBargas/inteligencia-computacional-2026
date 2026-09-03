import numpy as np
import matplotlib.pyplot as plt

def sigmoide(v, b=1):
    """Sigmoide bipolar/simétrica: rango (-1, 1)"""
    return 2 / (1 + np.exp(-b * v)) - 1

def signo(v):
    """Función signo"""
    return np.where(v >= 0, 1, -1)

def graficar_activaciones(v_min=-5, v_max=5, n=500):
    v = np.linspace(v_min, v_max, n)

    plt.figure()
    plt.plot(v, sigmoide(v), label="Sigmoide bipolar")
    plt.plot(v, signo(v), label="Signo")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("v (potencial de activación)")
    plt.ylabel("φ(v)")
    plt.title("Funciones de activación")
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_datos(x, d):
    """Grafica patrones 2D coloreados por clase (d en {-1,1})"""
    plt.figure()
    clase_pos = d.flatten() == 1
    clase_neg = d.flatten() == -1

    plt.scatter(x[clase_pos, 0], x[clase_pos, 1], marker='x', color='black', label='+1')
    plt.scatter(x[clase_neg, 0], x[clase_neg, 1], marker='s', facecolors='none', edgecolors='red', label='-1')

    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Distribución de clases')



def leer_datos(ruta_archivo, n_salidas):
    """Lee los datos de un archivo y devuelve las entradas y salidas"""
    matriz = np.loadtxt(ruta_archivo, delimiter=',')  
    n_columnas = matriz.shape[1]
    n_entradas = n_columnas - n_salidas

    x = matriz[:, 0:n_entradas]      # todas las filas, columnas de entrada
    y = matriz[:, n_entradas:n_columnas]   # todas las filas, columnas de salida
    return x, y



def inicializar_pesos(capas):
    """
    capas: lista de tamaños, ej: [2,4,1] (entradas,ocultas,salidas)
    Devuelve una lista de matrices de pesos inicializadas aleatoriamente
    """
    W=[]
    for p in range(len(capas)-1):
       filas = capas[p+1]
       columnas = capas[p] + 1 # +1 for bias
       W.append(np.random.uniform(-0.5, 0.5, (filas, columnas)))
    return W

def PropagacionAdelante(x, w, activacion):
    """
    x: entrada a esta capa, con el sesgo agregado
    w: matriz de pesos de esta capa
    activacion: función de activación
    Devuelve la salida de esta capa (sin sesgo)
    """
    v = np.dot(w, x)
    y = activacion(v)
    return y

def derivada_sigmoide(y,b=1):
    """Derivada de la sigmoide bipolar en función de y=φ(v): 1/2*(1+y)(1-y)"""
    return b * 0.5 * (1 + y) * (1 - y)

def DeltaSalida(e, y_salida):
    """delta de la capa de salida: e * derivada_sigmoide(y_salida)"""
    return e * derivada_sigmoide(y_salida)

def DeltaOculta(delta_siguiente, w_siguiente, y_capa):
    w_sin_bias = w_siguiente[:, 1:]       # todas las filas, todas las columnas menos la 0 (bias) sacamos el bias porque no es una neurona real de la capa oculta, es un sesgo. La neurona de sesgo no tiene delta, no tiene error, no tiene salida, no tiene activación, no tiene nada. Es solo un valor fijo que se agrega a la entrada de la capa siguiente.
    delta = w_sin_bias.T @ delta_siguiente #trasponemos la matriz de pesos para que las dimensiones coincidan y multiplicamos por el delta de la capa siguiente
    return delta * derivada_sigmoide(y_capa)

def Retropropagacion(e,y,w):
    """
    e: error de la capa de salida (d - y_final)
    y: lista de activaciones del forward (y[0]=entrada, ..., y[-1]=salida)
    w: lista de matrices de pesos
    Devuelve la lista de deltas, uno por cada matriz de pesos
    """
    L=len(w) #cantidad de capas con pesos
    deltas=[None]*L #inicializamos la lista de deltas
    deltas[-1] = DeltaSalida(e, y[-1]) #delta de la capa de salida

    for p in range(L-2, -1, -1): #recorremos las capas de atrás hacia adelante 
        deltas[p] = DeltaOculta(deltas[p+1], w[p+1], y[p+1])
    #arrancamos desde L-2 porque L-1 es la capa de salida, L-2 es la última capa oculta, y vamos hasta la primera capa oculta (p=0)

    return deltas

def ActualizarPesos(w_capa, delta_capa, entrada_con_bias, vel_aprendizaje):
    """Devuelve la matriz de pesos actualizada de una capa"""
    return w_capa + vel_aprendizaje * np.outer(delta_capa, entrada_con_bias)

def Probar(x_test, y_test, w, activacion):
    n_patrones = x_test.shape[0]
    aciertos = 0
    for n in range(n_patrones):
        y = [x_test[n]]
        for p in range(len(w)):
            x = np.concatenate(([-1], y[p]))
            y.append(PropagacionAdelante(x, w[p], activacion))

        prediccion = signo(y[-1])
        if np.array_equal(prediccion, y_test[n]):
            aciertos += 1

    return aciertos, n_patrones

def clasificar(x, w, activacion):
    """Clasifica un solo patrón x (sin sesgo) con la red ya entrenada"""
    y = [x]
    for p in range(len(w)):
        entrada_con_bias = np.concatenate(([-1], y[p]))
        y.append(PropagacionAdelante(entrada_con_bias, w[p], activacion))
    return signo(y[-1])[0]

def graficar_resultado(w, activacion, x, d, x_min=0, x_max=1.2, y_min=0, y_max=1.2, resolucion=200):
    """Grafica el plano de decisión (fondo) y los datos reales (puntos) juntos"""
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucion),
                          np.linspace(y_min, y_max, resolucion))
    puntos = np.c_[xx.ravel(), yy.ravel()]
    clases = np.array([clasificar(p, w, activacion) for p in puntos]).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, clases, levels=[-1.5, 0, 1.5], colors=['blue', 'white'], alpha=0.4)


    clase_pos = d.flatten() == 1
    clase_neg = d.flatten() == -1
    plt.scatter(x[clase_pos, 0], x[clase_pos, 1], marker='x', color='black', label='+1')
    plt.scatter(x[clase_neg, 0], x[clase_neg, 1], marker='s', facecolors='none', edgecolors='red', label='-1')


    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.title('Plano de decisión + datos reales')


def Ejercicio2(capas, ruta_train, ruta_test, vel_aprendizaje, max_epocas, err_umbral, activacion):
    #leemos los datos pasandole la ruta y la cantidad de salidas que tiene la red (capas[-1] es la cantidad de salidas)
    x_train, y_train = leer_datos(ruta_train, capas[-1])
    x_test, y_test = leer_datos(ruta_test, capas[-1])

    w = inicializar_pesos(capas) #inicializamos los pesos de la red (matriz de filas=neuronas de la capa siguiente, columnas=neuronas de la capa anterior + 1 para el sesgo)
    n_patrones = x_train.shape[0] #cantidad de patrones de entrenamiento

    for epoca in range(max_epocas):
        error_epoca = 0 #acumulador de error para esta época
        aciertos_epoca = 0 #acumulador de aciertos para esta época
        for n in range(n_patrones): #n recorre todos los patrones de entrenamiento
            #propagacion hacia adelante
            y=[x_train[n]] #y[0] es la entrada del patrón n
            for p in range(len(capas)-1): #p recorre todas las capas de la red
                #agregamos el sesgo a la entrada de la capa p
                x = np.concatenate(([-1], y[p])) #agregamos el sesgo x0=-1
                y.append(PropagacionAdelante(x, w[p], activacion)) #y[p+1] es la salida de la capa p

            e = y_train[n] - y[-1]
            deltas = Retropropagacion(e, y, w) #calculamos los deltas de todas las capas

            #Actualizamos los pesos de todas las capas
            for p in range(len(w)): #p recorre todas las capas de la red
                entrada_con_bias = np.concatenate(([-1], y[p])) #agregamos el sesgo a la entrada de la capa p
                w[p] = ActualizarPesos(w[p], deltas[p], entrada_con_bias, vel_aprendizaje) #actualizamos los pesos de la capa p

            error_epoca += 0.5 * np.sum(e**2) #acumulamos el error cuadrático medio de todos los patrones de entrenamiento
            if np.array_equal(signo(y[-1]), y_train[n]):
                aciertos_epoca += 1

        tasa_aciertos = aciertos_epoca / n_patrones  
        print(f"Época {epoca}: xi = {error_epoca}, aciertos = {aciertos_epoca}, tasa_aciertos = {tasa_aciertos:.2f}")

        if tasa_aciertos >= 0.95 or error_epoca < err_umbral:
            print(f"Convergencia alcanzada en la época {epoca}. Error: {error_epoca}")
            break
    else:
        print("No convergio en", max_epocas, "épocas. Error final:", error_epoca)

    aciertos, n_patrones = Probar(x_test, y_test, w, activacion)
    print(f"Aciertos: {aciertos}/{n_patrones} ({aciertos/n_patrones*100:.2f}%)")

    return w  # Devolvemos los pesos finales de la red

np.random.seed(42) #Fijamos semilla aleatoria para analisis 
w = Ejercicio2([2,8,8,1], 'concent_trn.csv', 'concent_tst.csv', 0.1, 3000, 0.05, sigmoide)

print("Pesos finales:")
for i, wp in enumerate(w):
    print(f"  W[{i}] =\n{wp}")

x, d = leer_datos('concent_tst.csv', 1)
graficar_resultado(w, sigmoide, x, d)
plt.show()


