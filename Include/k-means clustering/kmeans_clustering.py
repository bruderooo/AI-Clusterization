from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
import numpy as np
from Include.help_functions import euclidean_distance, increse, make_gif, prepare_data

file_number = 1


def draw(x_data, y_data, weights, axis_range):
    global file_number

    voronoi_plot_2d(weights, show_vertices=False, line_colors='red', line_width=2, line_alpha=1, point_size=20)
    plt.axis(ymin=axis_range[0], ymax=axis_range[1], xmin=axis_range[2], xmax=axis_range[3])
    plt.grid(b=True)
    plt.plot(x_data, y_data, '.g')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(f"Plots/plot{increse(file_number)}.png")
    plt.show(block=False)
    file_number += 1


def how_much_var_in_array(array, var):
    n = 0
    for i in array:
        if i[2] == var:
            n += 1
    return n


k = int(input("Choose number of clusters: "))

# Load training set
if "1" == input("Choose data set. '1' - for random points, other for wikamp test data: "):
    data = np.loadtxt("../training_set.txt")
    clusters = np.random.rand(k, 2) * 2
    axis_for_draw = (0, 2, 0, 2)
else:
    prepare_data("../test.txt", "../test_prepared.txt", ",", 5)
    data = np.loadtxt("../test_prepared.txt")
    clusters = np.random.rand(k, 2) * 20 - 10
    axis_for_draw = (-12, 9, -11, 14)

vor = Voronoi(clusters)

M = len(data)
data = np.concatenate((data, np.zeros((M, 1))), axis=1)

draw(data.T[0], data.T[1], vor, axis_for_draw)

epochs = 20

for epoch in range(epochs):
    tmp_clusters = np.zeros((k, 2))

    for vector_index, vector in enumerate(data):
        distance = np.asarray(euclidean_distance(vector[0], vector[1], clusters.T[0], clusters.T[1]))
        sorted_distance = np.sort(distance)

        # Index "która grupa"
        i = np.where(distance == sorted_distance[0])[0][0]

        vector[2] = i

        tmp_clusters[i][0] += vector[0]
        tmp_clusters[i][1] += vector[1]

    for i in range(k):
        tmp = how_much_var_in_array(data, i)
        if 0 != tmp:
            clusters[i][0] = tmp_clusters[i][0] / tmp
            clusters[i][1] = tmp_clusters[i][1] / tmp

    vor = Voronoi(clusters)

    draw(data.T[0], data.T[1], vor, axis_for_draw)

make_gif("./Plots", "clustry")
