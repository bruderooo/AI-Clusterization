from matplotlib import pyplot as plt
import numpy as np
import sys


file_number = 1


def increse(number):
    number = str(number)
    if len(number) == 4:
        return number
    elif len(number) == 3:
        return "0" + number
    elif len(number) == 2:
        return "00" + number
    elif len(number) == 1:
        return "000" + number
    elif len(number) == 5:
        return "0000" + number


# Distance function return euclidean distance
def euclidean_distance(A_x, A_y, B_x, B_y):
    return ((A_x - B_x) ** 2 + (A_y - B_y) ** 2) ** 0.5


def draw(x_data, y_data, x_weights, y_weights):
    global file_number

    plt.cla()
    plt.axis(ymin=-0.25, ymax=2.25, xmin=-0.25, xmax=2.25)
    plt.grid(b=True)
    plt.plot(x_data, y_data, '.g', x_weights, y_weights, '^r')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(f"Plots/plot{increse(file_number)}.png")
    # plt.draw()
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
data = np.loadtxt("../training_set.txt")
M = len(data)

# Number of dimension
N = 2

eta_max = 0.8
eta_min = 0.003
lambd_min = 0.01

weights = np.random.rand(N, H) * 2
# plt.xkcd()
draw(data.T[0], data.T[1], weights[0], weights[1])

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

    draw(data.T[0], data.T[1], weights[0], weights[1])
