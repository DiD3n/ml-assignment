import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

data = pd.read_csv('Genshin Impact Character Stats.csv', delimiter=',')

elements = ['Anemo', 'Geo', 'Electro', 'Dendro', 'Hydro', 'Pyro', 'Cryo']

element_mapping = {'Anemo': 1, 'Geo': 2, 'Electro': 3, 'Dendro': 4, 'Hydro': 5, 'Pyro': 6, 'Cryo': 7}
weapon_mapping = {'Sword': 1, 'Claymore': 2, 'Polearm': 3, 'Bow': 4, 'Catalyst': 5}
role_mapping = {'DPS': 1, 'Sub DPS': 2, 'Healer': 4, 'Support': 3}
ascension_mapping = {'HP': 1, 'ATK': 2, 'DEF': 3, 'Elemental Mastery': 4, 'Energy Recharge': 5, 'CRIT Rate': 6, 'CRIT DMG': 7, 'Healing Bonus': 8,
                     'Anemo DMG': 9, 'Geo DMG': 10, 'Electro DMG': 11, 'Dendro DMG': 12, 'Hydro DMG': 13, 'Pyro DMG': 14, 'Cryo DMG': 15, 'Physical DMG': 16}

data['Element'] = data['Element'].map(element_mapping)
data['Weapon'] = data['Weapon'].map(weapon_mapping)
data['Main role'] = data['Main role'].map(role_mapping)
data['Ascension'] = data['Ascension'].map(ascension_mapping)

for element in elements:
    data[element + '_element'] = (data['Element'] == element_mapping.get(element)).astype(int)


inputColumnList = ['Rarity', 'Weapon', 'Main role', 'Ascension', 'Base HP', 'Base ATK', 'Base DEF']
outputColumnList = ['Anemo_element', 'Geo_element', 'Electro_element', 'Dendro_element', 'Hydro_element', 'Pyro_element', 'Cryo_element']
X = data[inputColumnList].values
Y = data[outputColumnList].values

if np.isnan(X).any():
    # find the index of the NaN value
    index = np.argwhere(np.isnan(X))
    raise ValueError("mapping gone wrong, something is NaN")

input_size = len(inputColumnList)
hidden_size = 7
output_size = len(outputColumnList)

np.random.seed(42)
weights_input_to_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(1, hidden_size)
weights_hidden_to_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(1, output_size)

learning_rate = 0.0001
epochs = 1000


def run_network(data):
    hidden_layer_input = np.dot(data, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    predicted_output = softmax(output_layer_input)

    return hidden_layer_output, predicted_output


losses1 = []
losses2 = []
for epoch in range(epochs):
    hidden_layer_output, predicted_output = run_network(X)

    error = Y - predicted_output
    d_predicted_output = error

    error_hidden_layer = d_predicted_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    weights_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_to_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 10 == 0:
        loss = cross_entropy_loss(Y, predicted_output)
        losses1.append(loss)
        print(f"{epoch / epochs * 100:.0f}% - Epoch {epoch}, Loss: {loss}")

def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return (predictions == labels).mean()

import matplotlib.pyplot as plt

plt.plot(losses1)
plt.plot(losses2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.show()

while True:
    print("Enter character stats:")
    print("Rarity (4/5) stars:")
    rarity = int(input())
    print("Weapon", weapon_mapping," (0-5):")
    weapon = int(input())
    print("Main", role_mapping," role (0-4):")
    role = int(input())
    print("Ascension", ascension_mapping ," (1-16):")
    ascension = int(input())
    print("Base HP:")
    hp = float(input())
    print("Base ATK:")
    atk = float(input())
    print("Base DEF:")
    defense = float(input())

    input_data = np.array([[rarity, weapon, role, ascension, hp, atk, defense]])
    hidden_layer_output, predicted_output = run_network(input_data)
    print("Predicted elements:")
    print(predicted_output)
    print("Predicted element:")
    print(elements[np.argmax(predicted_output)])
