from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
import numpy as np


# Distance function return euclidean distance
def euclidean_distance(A_x, A_y, B_x, B_y):
    return ((A_x - B_x) ** 2 + (A_y - B_y) ** 2) ** 0.5


k = 3

clusters = np.random.rand(k, 2) + 0.5

vor = Voronoi(clusters)

# lista = []
# for i in range(1, 10_000, 20):
#     lista.append(i)
#
# lista = tuple(lista)
# data = np.loadtxt("../test.txt", delimiter=',', usecols=lista)

data = np.loadtxt("../training_set.txt")
M = len(data)
data = np.concatenate((data, np.zeros((M, 1))), axis=1)

voronoi_plot_2d(vor, show_vertices=False, line_colors='red', line_width=2, line_alpha=1, point_size=20)

plt.axis(ymin=-0.25, ymax=2.25, xmin=-0.25, xmax=2.25)
plt.grid(b=True)
plt.gca().set_aspect('equal', 'box')
plt.plot(data.T[0], data.T[1], '.g')
plt.show()

for vector in enumerate(data):
    distance = np.asarray(euclidean_distance(vector[1][0], vector[1][1], clusters.T[0], clusters.T[1]))
    sorted_distance = np.sort(distance.copy())

    # Index "która grupa"
    i = np.where(distance == sorted_distance[0])[0][0]

    vector[1][2] = i

print(data)
