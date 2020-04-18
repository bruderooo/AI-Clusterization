from matplotlib import pyplot as plt
import numpy as np
import sys


# Distance function return euclidean distance
def distance(A_x, A_y, B_x, B_y):
    # print(A_x, A_y, B_x, B_y)
    return ((A_x - B_x)**2 + (A_y - B_y)**2)**0.5


if len(sys.argv) <= 1:
    # H = int(input("Podaj ilosc neuronow ukrytych: "))
    # epochs = int(input("Podaj ilość epok: "))
    # eta = float(input("Podaj krok: "))
    H = 3
    epochs = 1
    eta = 1
else:
    H = int(sys.argv[1])
    epochs = int(sys.argv[2])
    eta = float(sys.argv[3])

# Load training set
data = np.loadtxt("../training_set.txt")

# Number of dimension
N = 2

weights = 2 * np.random.rand(N, H) - 1

plt.figure().add_subplot().add_patch(plt.Polygon(weights.T, "r", facecolor="w", edgecolor="r"))
plt.axis(ymin=-1.25, ymax=1.25, xmin=-1.25, xmax=1.25)
plt.grid(b=True)
plt.plot(data.T[0], data.T[1], '.g')
plt.gca().set_aspect('equal', 'box')
plt.show()

for vector in data:
    print(distance(vector[0], vector[1], weights[0], weights[1]))
