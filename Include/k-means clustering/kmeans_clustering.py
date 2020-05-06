from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
import numpy as np
from Include import euclidean_distance, increse, make_gif, prepare_data

file_number = 1


def draw(x_data, y_data, weights):
    global file_number

    plt.cla()
    voronoi_plot_2d(weights, show_vertices=False, line_colors='red', line_width=2, line_alpha=1, point_size=20)
    # plt.axis(ymin=-0.25, ymax=2.25, xmin=-0.25, xmax=2.25)
    plt.axis(ymin=-10, ymax=10, xmin=-10, xmax=10)
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

clusters = np.random.rand(k, 2) + 0.5

vor = Voronoi(clusters)

# Load training set
if "1" == input("Choose data set. '1' - for random points, other for wikamp test data: "):
    data = np.loadtxt("../training_set.txt")
else:
    prepare_data("../test.txt", "../test_prepared.txt", ",", 5)
    data = np.loadtxt("../test_prepared.txt")

M = len(data)
data = np.concatenate((data, np.zeros((M, 1))), axis=1)

draw(data.T[0], data.T[1], vor)

for epoch in range(10):
    tmp_clusters = np.zeros((k, 2))

    for vector in enumerate(data):
        distance = np.asarray(euclidean_distance(vector[1][0], vector[1][1], clusters.T[0], clusters.T[1]))
        sorted_distance = np.sort(distance.copy())

        # Index "która grupa"
        i = np.where(distance == sorted_distance[0])[0][0]

        vector[1][2] = i

        tmp_clusters[i][0] += vector[1][0]
        tmp_clusters[i][1] += vector[1][1]

    for i in range(k):
        tmp = how_much_var_in_array(data, i)
        if 0 != tmp:
            clusters[i][0] = tmp_clusters[i][0] / tmp
            clusters[i][1] = tmp_clusters[i][1] / tmp

    vor = Voronoi(clusters)

    draw(data.T[0], data.T[1], vor)

make_gif("./Plots/*", "clustry")
