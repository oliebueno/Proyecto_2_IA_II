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

# Parte 2.1-------------------------------------------------------------

# Cargar los datos de N = 2000
datos_entradaN2000 = leer_datos_desde_archivo('datos_P2_AJ2024_N2000.txt')
X_train_2000, y_train_2000 = load_data(datos_entradaN2000)

# Tasas de aprendizaje a evaluar y cantidad de iteraciones
learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
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
plt.title('Evolución de los errores de entrenamiento (2.1.1)')
plt.xlabel('Iteraciones')
plt.ylabel('Error de entrenamiento (MSE)')
plt.legend()
plt.show()

# Parte 2.2-----------------------------------------------------------

# Cargar los datos
datos_entradaN500 = leer_datos_desde_archivo('datos_P2_AJ2024_N500.txt')
datos_entradaN1000 = leer_datos_desde_archivo('datos_P2_AJ2024_N1000.txt')
X_train_500, y_train_500 = load_data(datos_entradaN500)
X_train_1000, y_train_1000 = load_data(datos_entradaN1000)

# Tasa de aprendizaje
learning_rate = 0.1

# Entrenar la red para N=500
bp_500 = bp(2, 8, 1, learning_rate)
errors_500 = bp_500.train(X_train_500, y_train_500, epochs)

# Entrenar la red para N=1000
bp_1000 = bp(2, 8, 1, learning_rate)
errors_1000 = bp_1000.train(X_train_1000, y_train_1000, epochs)

# Graficar la evolución de los errores de entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(errors_500, label='N = 500')
plt.plot(errors_1000, label='N = 1000')
plt.title(
    'Evolución de los errores de entrenamiento (comparación para N=500 y N=1000) (2.2.1)')
plt.xlabel('Iteraciones')
plt.ylabel('Error de entrenamiento (MSE)')
plt.legend()
plt.show()


# Parte 2.2-----------------------------------------------------------------

errors_dict = {}

# Entrenar la red para diferentes cantidades de neuronas en la capa intermedia
for hidden_size in range(2, 11):
    back_p = bp(2, hidden_size, 1, learning_rate)
    errors = back_p.train(X_train_2000, y_train_2000, epochs)
    errors_dict[hidden_size] = errors

# Graficar la evolución de los errores de entrenamiento
plt.figure(figsize=(10, 6))
for hidden_size, errors in errors_dict.items():
    plt.plot(errors, label=f'{hidden_size} neuronas en la capa intermedia')
plt.title(
    'Evolución de los errores de entrenamiento para cada número de neuronas (2.3)')
plt.xlabel('Iteraciones')
plt.ylabel('Error de entrenamiento (MSE)')
plt.legend()
plt.show()
