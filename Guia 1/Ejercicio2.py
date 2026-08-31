import numpy as np
import matplotlib.pyplot as plt

def funcActivacion(result):
    return 1 if result >= 0 else -1

def CargarDatosCSV(nombre_archivo):
    datos = np.loadtxt(nombre_archivo, delimiter=",")
    x = datos[:, :-1] #todas las columnas menos la última
    y = datos[:, -1] #la ultima columna

    x_extendido = np.hstack((-np.ones((x.shape[0], 1)), x)) # Agregar bias (-1) a las entradas
    return x_extendido, y

def GraficarPatrones(x, y, titulo):
    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 1], x[:, 2], c=-y, cmap='bwr', edgecolor='black', label='Patrones') #graficamos los patrones, el color depende del valor de y #cmap es para azul o rojo
    plt.ylim(x[:, 2].min() - 1, x[:, 2].max() + 1)   # limite para que grafique entre -2 y 2
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(titulo)
    plt.grid(True)
    plt.legend()

def dibujar_recta(pesos, color, alpha, label=None):
    # v = w0*(-1) + w1*x1 + w2*x2 = 0
    # x2 = (w0/w2) - ((w1/w2)*x1)
    x_vals = np.linspace(-2, 2, 100)
    y_vals = (pesos[0] - pesos[1] * x_vals) / pesos[2]
    plt.plot(x_vals, y_vals, color=color, alpha=alpha, label=label)


def entrenar(x_extendido, y, tasa_aprendizaje, max_epocas):
    pesos = np.random.uniform(-0.5, 0.5, x_extendido.shape[1]) # Inicializamos pesos aleatorios entre -0.5 y 0.5
    print("Pesos iniciales:", pesos)

    n_patrones = x_extendido.shape[0] # Cantidad de patrones de entrenamiento
    #Agregamos la recta de decisión al gráfico
    dibujar_recta(pesos, color='red', alpha=0.5) #recta inicial

    for epoca in range(1, max_epocas + 1):
        errores_epoca = 0 # Contador de errores por época
        for i in range(n_patrones):
            v = np.dot(x_extendido[i], pesos) # Producto punto entre el patrón de entrada y los pesos
            y_obtenida = funcActivacion(v) # Función de activación signo
            error = y[i] - y_obtenida 

            if error != 0:

                pesos += (tasa_aprendizaje / 2) * error * x_extendido[i] 
                errores_epoca += 1

        print(f"Época {epoca}: errores = {errores_epoca}")
        dibujar_recta(pesos, color="black", alpha=0.5) #una recta por epoca 

        if errores_epoca == 0:
            print("Entrenamiento completo, no hay errores en la época", epoca)
            break
    
    dibujar_recta(pesos, color="green", alpha=1.0, label="Recta final")  #ultima recta de decision en verde
    return pesos, epoca, errores_epoca

##LLAMAOS A LAS FUNCIONES


# --- OR ---
x_train, y_train = CargarDatosCSV("OR_trn.csv")
GraficarPatrones(x_train, y_train, "OR")
pesos_entrenados, epocas, errores = entrenar(x_train, y_train, 0.1, 100)
print("Pesos entrenados:", pesos_entrenados)
plt.show()
print("\n\n")
print("--------------------------------------------------")

# --- XOR ---
x_train, y_train = CargarDatosCSV("XOR_trn.csv")
GraficarPatrones(x_train, y_train, "XOR")
pesos_entrenados, epocas, errores = entrenar(x_train, y_train, 0.1, 100)
print("Pesos entrenados:", pesos_entrenados)
plt.show()
