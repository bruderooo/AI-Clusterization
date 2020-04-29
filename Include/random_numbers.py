import numpy as np
import matplotlib.pyplot as plt


def geometric_points(how_much_points, figure):
    i, x_axis, y_axis = 0, [], []
    while i < how_much_points:
        tmp = 2 * np.random.rand(2)
        if choose(tmp, figure):
            x_axis.append(tmp[0])
            y_axis.append(tmp[1])
            i += 1
    return np.array([x_axis, y_axis])


def choose(list, type):
    if type == 'circle':
        return ((list[0] - 1) ** 2 + (list[1] - 1) ** 2) <= 1
    elif type == 'rhombus':
        return (abs(list[0] - 1) + abs(list[1] - 1)) <= 1
    elif type == 'funny_sq':
        return ((list[0] - 1) ** 4 + (list[0] - 1) ** 4) <= 1
    elif type == 'star':
        return (abs(list[0] - 1) ** 0.5 + abs(list[0] - 1) ** 0.5) <= 1
    elif type == '4circle':
        return (abs(list[0] - 1) - 0.5) ** 2 + (abs(list[1] - 1) - 0.5) ** 2 <= 0.09
    elif type == 'small_circle':
        return (list[0] ** 2 + list[1] ** 2) <= 0.1


choose_list = ['circle', 'rhombus', 'funny_sq', 'star', '4circle', 'small_circle']

matrix = geometric_points(100, choose_list[4])

np.savetxt("training_set.txt", matrix.T)

plt.axis(ymin=-0.25, ymax=2.25, xmin=-0.25, xmax=2.25)
plt.grid(b=True)
plt.plot(matrix[0], matrix[1], '.g')
plt.gca().set_aspect('equal', 'box')
plt.show()
