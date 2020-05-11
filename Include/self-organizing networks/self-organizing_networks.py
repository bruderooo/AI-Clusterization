from matplotlib import pyplot as plt
import numpy as np
import sys
from Include.help_functions import increse, euclidean_distance, make_gif, prepare_data

file_number = 1


def draw(x_data, y_data, x_weights, y_weights):
    global file_number

    plt.cla()
    plt.grid(b=True)
    plt.plot(x_data, y_data, '.g', x_weights, y_weights, '^r')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(f"Plots/plot{increse(file_number)}.png")
    plt.show(block=False)
    file_number += 1


radius_0 = None

if len(sys.argv) <= 1:
    H = int(input("Podaj ilosc neuronow ukrytych: "))
    algorithm = input("Podaj algorytm [kohonen/gas]: ")
else:
    H = int(sys.argv[1])
    algorithm = sys.argv[2]

if algorithm == "kohonen":
    radius_0 = float(input("Podaj promien sasiedztwa: "))

# Number of dimension
N = 2

eta_max = 0.8
eta_min = 0.003
lambd_min = 0.01

# Load training set
if "1" == input("Choose data set. '1' - for random points, other for wikamp test data: "):
    data = np.loadtxt("../training_set.txt")
    weights = np.random.rand(N, H) * 2
else:
    prepare_data("../test.txt", "../test_prepared.txt", ",", 5)
    data = np.loadtxt("../test_prepared.txt")
    weights = np.random.rand(N, H) * 16 - 8

M = len(data)


draw(data.T[0], data.T[1], weights[0], weights[1])

epochs = 10

for epoch in range(epochs):
    for vector_index, vector in enumerate(data):

        distance = np.asarray(euclidean_distance(vector[0], vector[1], weights[0], weights[1]))
        sorted_distance = np.sort(distance.copy())

        for neuron_index, neuron in enumerate(sorted_distance):
            # i[0][0] - is index od neuron
            i = np.where(distance == neuron)[0][0]

            # Obliczanie eta dla danej iteracji
            eta_k = eta_max * (eta_min / eta_max) ** (epoch / epochs)

            # Obliczanie lambdy dla danej iteracji
            lambd = (H / 2.0) * (lambd_min / (H / 2.0)) ** (epoch / epochs)

            if algorithm == "gas":
                weights.T[i] += eta_k * np.exp(-neuron_index / lambd) * (vector - weights.T[i])

            elif algorithm == "kohonen":
                radius = radius_0 * np.exp(-epoch / epochs)
                # eta_k = eta_max * np.exp(-epoch / epochs)
                winner_index = np.where(distance == sorted_distance[0])[0][0]
                if abs(distance[winner_index] - distance[i]) <= radius:
                    weights.T[i] += eta_k * np.exp(-neuron_index / lambd) * (vector - weights.T[i])
                    # weights.T[i] += eta_k * np.exp(-(distance[i])**2 / (2 * radius ** 2)) * (vector - weights.T[i])

        if (vector_index % (M / 2)) == 0:
            draw(data.T[0], data.T[1], weights[0], weights[1])

make_gif("./Plots", algorithm)
