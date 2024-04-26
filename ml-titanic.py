import numpy as np
import pandas as pd

# Dane dla funkcji titanica
data = pd.read_csv('titanic.csv', delimiter=';', decimal=',')

embarkedMap = {
    None: 0,  # Brak wartości
    'S': 1,
    'C': 2,
    'Q': 3
}
data['Embarked'] = data['Embarked'].map(lambda x: x in embarkedMap and embarkedMap[x] or 0)

columnsList = ['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Embarked']
# Wektory wejściowe i etykiety
X = data[columnsList].values
Y = data['Survived'].values.reshape(-1, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Parametry sieci
input_size = X.shape[1]
hidden_size = 5
output_size = 1

# Wagi i biasy
np.random.seed(42)
weights_input_to_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_to_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, output_size)

learning_rate = 0.001
epochs = 50000

def run_network(data=X):
    hidden_layer_input = np.dot(data, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    return hidden_layer_output, predicted_output


losses = []
for epoch in range(epochs):
    # Propagacja w przód
    hidden_layer_output, predicted_output = run_network()

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

    # Rejestrowanie straty
    if epoch % 100 == 0:
        loss = np.mean(np.square(Y - predicted_output))
        losses.append(loss)
        print(f"{epoch/epochs*100:.2f}% done, loss: {loss}");

# Wizlualizacja neuronów

import matplotlib.pyplot as plt

print("Wagi warstwy ukrytej:")
print(weights_input_to_hidden.shape)
print("Bias warstwy ukrytej:")
print(bias_hidden)
print("Wagi warstwy wyjściowej:")
print(weights_hidden_to_output)
print("Bias warstwy wyjściowej:")
print(bias_output)


plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.show()


bads = 0
for i in range(len(X)):
    data = X[i]
    result = Y[i]
    _, predicted_output = run_network(data)

    # print(f"Exp: {result}, Predicted: {predicted_output[0][0] * 100:.2f}% - age: {data[1]} sex: {data[0]}")
    if result != round(predicted_output[0][0]):
        bads += 1

print(f"Bad predictions: {bads} / {len(X)}")

# Sprawdzanie modelu
# note: u mnie działa
while True:
    print("Podaj dane pasażera:")
    pclass = float(input("Klasa: "))
    age = float(input("Wiek: "))
    sex = float(input("Płeć (m -> 0/f -> 1): "))
    sibsp = float(input("SibSp: "))
    parch = float(input("Parch: "))
    embarked = float(input("Embarked (S -> 1, C -> 2, Q -> 3): "))

    data = np.array([sex, age, pclass, sibsp, parch, embarked]).reshape(1, -1)

    _, predicted_output = run_network(data)

    print("Szansa na przeżycie: ", predicted_output[0][0])
