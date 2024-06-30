# Programa proncipal para la parte 2

from BackPropagation import Backpropagation as bp
import numpy as np
import matplotlib.pyplot as plt

# Función para la lectura de datos


def leer_datos_desde_archivo(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        lineas = archivo.readlines()
        datos = [tuple(map(float, linea.strip().split())) for linea in lineas]
    return datos

# Función para cargar los datos


def load_data(datos):
    data = np.array(datos)
    X = data[:, :2]
    y = data[:, 2].reshape(-1, 1)
    return X, y


# Comienzo del programa principal

# Parte 2.1

# Cargar los datos de N = 2000
datos_entradaN2000 = leer_datos_desde_archivo('datos_P2_AJ2024_N2000.txt')
X_train_2000, y_train_2000 = load_data(datos_entradaN2000)

# Tasas de aprendizaje a evaluar y cantidad de iteraciones
learning_rates = [0.000001]
epochs = 1000

errors_dict = {}

# Entrenar la red para cada tasa de aprendizaje y registrar los errores
for lr in learning_rates:
    back_p = bp(2, 8, 1, lr)
    errors = back_p.train(X_train_2000, y_train_2000, epochs)
    errors_dict[lr] = errors

# Graficar la evolución de los errores de entrenamiento
plt.figure(figsize=(10, 6))
for lr, errors in errors_dict.items():
    plt.plot(errors, label=f'Tasa de aprendizaje = {lr}')
plt.title('Evolución de los errores de entrenamiento')
plt.xlabel('Iteraciones')
plt.ylabel('Error de entrenamiento (MSE)')
plt.legend()
plt.show()
