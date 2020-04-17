import numpy as np
import matplotlib.pyplot as plt


def geometric_points(how_much_points, figure):
    i, x_axis, y_axis = 0, [], []
    while i < how_much_points:
        tmp = 2 * np.random.rand(2) - 1
        if choose(tmp, figure):
            x_axis.append(tmp[0])
            y_axis.append(tmp[1])
            i += 1
    return np.array([x_axis, y_axis])


def choose(list, type):
    if type == 'circle':
        return (list[0] ** 2 + list[1] ** 2) <= 1
    elif type == 'rhombus':
        return (abs(list[0]) + abs(list[1])) <= 1
    elif type == 'funny_sq':
        return (list[0] ** 4 + list[1] ** 4) <= 1
    elif type == 'star':
        return (abs(list[0]) ** 0.5 + abs(list[1]) ** 0.5) <= 1


choose_list = ['circle', 'rhombus', 'funny_sq', 'star']

matrix = geometric_points(500, choose_list[0])

np.savetxt("training_set.txt", matrix.T)

plt.grid(b=True)
plt.plot(matrix[0], matrix[1], '.r')
plt.gca().set_aspect('equal', 'box')
plt.show()
