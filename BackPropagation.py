import numpy as np

# Clase que implementa Backpropagation


class Backpropagation:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size        # Cantidad de neuronas en la capa iniciales
        self.hidden_size = hidden_size      # Cantidad de neuronas en la capa intermedia
        self.output_size = output_size      # Cantidad de neuronas en la capa final
        self.learning_rate = learning_rate  #

        # Inicialización de pesos
        self.hidden_weights = np.random.normal(
            0, 0.05, size=(self.input_size, self.hidden_size))
        self.output_weights = np.random.normal(
            0, 0.05, size=(self.hidden_size, self.output_size))

        # Inicialización de los bias
        self.hidden_bias = np.random.normal(
            0, 0.05, size=(1, self.hidden_size))
        self.output_bias = np.random.normal(
            0, 0.05, size=(1, self.output_size))

    # Función sigmoidal
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada de la función sigmoidal
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Procedimiento para entrenar la red
    def train(self, inputs, expected_output, epochs):
        errors = []
        for epoch in range(epochs):
            print(epoch)
            epoch_error = 0
            for input_vector, target_vector in zip(inputs, expected_output):

                # Forward Propagation
                # Capa oculta
                hidden_layer_activation = np.dot(
                    input_vector, self.hidden_weights) + self.hidden_bias
                hidden_layer_output = self.sigmoid(hidden_layer_activation)
                # Capa de salida
                output_layer_activation = np.dot(
                    hidden_layer_output, self.output_weights) + self.output_bias
                predicted_output = self.sigmoid(output_layer_activation)

                # Backpropagation
                # Cálculo del error en la capa de salida
                output_error = target_vector - predicted_output
                delta_output = output_error * \
                    self.sigmoid_derivative(predicted_output)

                # Cálculo del error en la capa oculta
                hidden_error = delta_output.dot(self.output_weights.T)
                delta_hidden = hidden_error * \
                    self.sigmoid_derivative(hidden_layer_output)

                # Actualización de pesos y sesgos
                self.output_weights += hidden_layer_output.T.reshape(-1, 1).dot(
                    delta_output.reshape(1, -1)) * self.learning_rate
                self.output_bias += delta_output * self.learning_rate
                self.hidden_weights += input_vector.T.reshape(-1, 1).dot(
                    delta_hidden.reshape(1, -1)) * self.learning_rate
                self.hidden_bias += delta_hidden * self.learning_rate

                epoch_error += np.mean(np.square(output_error))
            errors.append(epoch_error / len(inputs))
        return errors

    # Función que prueba la red
    def predict(self, inputs):
        hidden_layer_activation = np.dot(
            inputs, self.hidden_weights) + self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(
            hidden_layer_output, self.output_weights) + self.output_bias
        predicted_output = self.sigmoid(output_layer_activation)
        return predicted_output
