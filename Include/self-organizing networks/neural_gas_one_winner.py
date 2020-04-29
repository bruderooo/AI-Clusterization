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


# Distance function return euclidean distance
def euclidean_distance(A_x, A_y, B_x, B_y):
    return ((A_x - B_x) ** 2 + (A_y - B_y) ** 2) ** 0.5


# Distance in 2 axis
def distance_xy(A_x, A_y, B_x, B_y):
    return [abs(A_x - B_x), abs(A_y - B_y)]


def draw(x_data, y_data, x_weights, y_weights):
    global file_number

    plt.axis(ymin=-0.25, ymax=2.25, xmin=-0.25, xmax=2.25)
    plt.grid(b=True)
    plt.plot(x_data, y_data, '.g', x_weights, y_weights, '^r')
    plt.gca().set_aspect('equal', 'box')

    plt.savefig(f"Plots_one_winner/plot{increse(file_number)}.png")
    plt.show()

    file_number += 1


if len(sys.argv) <= 1:
    # H = int(input("Podaj ilosc neuronow ukrytych: "))
    # epochs = int(input("Podaj ilość epok: "))
    # eta = float(input("Podaj krok: "))
    H = 50
    epochs = 1
    eta_k = 0.8 * np.random.rand(H)
else:
    H = int(sys.argv[1])
    epochs = int(sys.argv[2])
    eta_k = float(sys.argv[3])

# Load training set
data = np.loadtxt("../training_set.txt")
M = len(data)

# Number of dimension
N = 2

eta_max = 0.8
eta_min = 0.003
lambd_min = 0.01

weights = np.random.rand(N, H) * 1.5 + 0.25

draw(data.T[0], data.T[1], weights[0], weights[1])

for epoch in range(10):
    for vector in enumerate(data):
        distance = np.asarray(euclidean_distance(vector[1][0], vector[1][1], weights[0], weights[1]))
        sorted_distance = np.sort(distance.copy())

        # i[0][0] - is index od neuron
        i = np.where(distance == sorted_distance[0])[0][0]

        # Obliczanie eta dla danej iteracji
        eta_k[i] = eta_max * (eta_min / eta_max) ** (vector[0] / M)

        # print("distance:", distance)
        # print("sorted_distance:", sorted_distance)
        # print("Dane:", vector[1])
        # print("Indeks wygranego neuronu:", i)
        # print("Neuron wygrany", weights.T[i])
        # print("To dodamy do wagi ^:", eta_k[i] * (distance_2D.T[i] - weights.T[i]))

        # input()

        weights.T[i] += eta_k[i] * (vector[1] - weights.T[i])

        draw(data.T[0], data.T[1], weights[0], weights[1])
