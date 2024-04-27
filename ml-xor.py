import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dane dla funkcji XOR
data = pd.DataFrame({
    'X1': [0, 0, 1, 1],
    'X2': [0, 1, 0, 1],
    'Y':  [0, 1, 1, 0]
})

# Wektory wejściowe i etykiety
X = data[['X1', 'X2']].values
Y = data['Y'].values.reshape(-1, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Parametry sieci
input_size = X.shape[1]
hidden_size = 2
output_size = 1

# Wagi i biasy
np.random.seed(42)
weights_input_to_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_to_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, output_size)

learning_rate = 0.5
epochs = 10000


losses = []
losses2 = []
for epoch in range(epochs):

    # Propagacja w przód
    hidden_layer_input = np.dot(X, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Oblicz błąd
    error = Y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # Błąd dla warstwy ukrytej
    error_hidden_layer = d_predicted_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Aktualizacja wag i biasów
    weights_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_to_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Co pewien czas wypisz błąd
    if epoch % 100 == 0:
        loss = np.mean(np.square(error))
        loss2 = np.mean(np.square(error_hidden_layer))
        losses.append(loss)
        losses2.append(loss2)
        print(f"{epoch/epochs*100:.2f}% done, loss: {loss} {loss2}")


plt.plot(losses)
plt.plot(losses2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.show()


# Sprawdzenie wyników
while True:
    x1 = int(input("Podaj x1: "))
    x2 = int(input("Podaj x2: "))

    hidden_layer_input = np.dot([x1, x2], weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    print(f"Przewidziana wartość: {int(predicted_output[0][0] * 100)}% na tak.")