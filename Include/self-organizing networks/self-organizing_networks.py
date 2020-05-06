from matplotlib import pyplot as plt
import numpy as np
import sys
from Include import increse, euclidean_distance, make_gif, prepare_data

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


if len(sys.argv) <= 1:
    H = int(input("Podaj ilosc neuronow ukrytych: "))
    algorithm = input("Podaj algorytm [kohonen/gas]: ")
else:
    H = int(sys.argv[1])
    algorithm = sys.argv[2]

if algorithm == "kohonen":
    R = float(input("Podaj promien sasiedztwa: "))


# Load training set
if "1" == input("Choose data set. '1' - for random points, other for wikamp test data: "):
    data = np.loadtxt("../training_set.txt")
else:
    prepare_data("../test.txt", "../test_prepared.txt", ",", 5)
    data = np.loadtxt("../test_prepared.txt")


M = len(data)

# Number of dimension
N = 2

eta_max = 0.8
eta_min = 0.003
lambd_min = 0.01

weights = np.random.rand(N, H) * 15 - 7.5
# plt.xkcd()
draw(data.T[0], data.T[1], weights[0], weights[1])

for epoch in range(10):
    for vector in enumerate(data):

        distance = np.asarray(euclidean_distance(vector[1][0], vector[1][1], weights[0], weights[1]))
        sorted_distance = np.sort(distance.copy())

        for neuron in enumerate(sorted_distance):
            # i[0][0] - is index od neuron
            i = np.where(distance == neuron[1])[0][0]

            # Obliczanie eta dla danej iteracji
            eta_k = eta_max * (eta_min / eta_max) ** (vector[0] / M)

            if algorithm == "gas":
                # Obliczanie lambdy dla danej iteracji
                lambd = (H / 2.0) * (lambd_min / (H / 2.0)) ** (vector[0] / M)

                weights.T[i] += eta_k * np.exp(-neuron[0] / lambd) * (vector[1] - weights.T[i])

            elif algorithm == "kohonen":
                winner_index = np.where(distance == sorted_distance[0])[0][0]
                dist = euclidean_distance(weights.T[winner_index][1], weights.T[winner_index][1], weights.T[i][1], weights.T[i][1])
                weights.T[i] += eta_k * np.exp(-(dist ** 2 / (2 * R ** 2))) * (vector[1] - weights.T[i])

        if (vector[0] % (M / 2)) == 0:
            draw(data.T[0], data.T[1], weights[0], weights[1])

make_gif("./Plots/*", algorithm)
