from matplotlib.image import imread
import numpy as np

image_path = "./Lena.gif"
image = imread(image_path)

print(image)

wymiar = 3

lista = []
for i in range(wymiar):
    tmp_list = []
    for j in range(wymiar):
        tmp_list.append(image[i][j])
