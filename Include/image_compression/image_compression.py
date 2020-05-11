from matplotlib.image import imread
import numpy as np


def euclidean_distance(A, B):
    sum = 0.0
    for i in range(len(A)):
        sum += (A[i] - B[i]) ** 2
    return sum ** 0.5


image_path = "./Lena.gif"

image = imread(image_path)
print(image)

# Define the window size
frame_row = 4
frame_col = 4

data = np.zeros(((image.shape[0] // frame_row - 1) * (image.shape[0] // frame_col - 1), frame_row * frame_col))

i = 0

# Crop out the window and calculate the histogram
for r in range(0, image.shape[0] - frame_row, frame_row):
    for c in range(0, image.shape[1] - frame_col, frame_col):
        data[i] = (image[r:r + frame_row, c:c + frame_col].flatten())
        i += 1

weights = np.random.rand(frame_row * frame_col, image.shape[0])

eta_max = 0.8
eta_min = 0.003
lambd_min = 0.01

epochs = 15

# print(weights)

for epoch in range(epochs):
    for vector in enumerate(data):

        distance = euclidean_distance(vector[1], weights)
        # print(distance)
        # input()
        sorted_distance = np.sort(distance.copy())

        for neuron in enumerate(sorted_distance):
            # i[0][0] - is index od neuron
            i = np.where(distance == neuron[1])[0][0]

            # Obliczanie eta dla danej iteracji
            eta_k = eta_max * (eta_min / eta_max) ** (epoch / epochs)

            # Obliczanie lambdy dla danej iteracji
            lambd = (image.shape[0] / 2.0) * (lambd_min / (image.shape[0] / 2.0)) ** (epoch / epochs)

            # print(weights.T[i])
            # print(" ")
            # print(vector[1])
            # input()

            weights.T[i] += eta_k * np.exp(-neuron[0] / lambd) * (vector[1] - weights.T[i])

# print(weights.shape)


# weights = weights.astype('int32')

# print(weights)

np.savetxt("obrazek.txt", weights, fmt='%i')
