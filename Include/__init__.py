import numpy as np
from PIL import Image
import glob


# Distance function return euclidean distance
def euclidean_distance(A_x, A_y, B_x, B_y):
    return ((A_x - B_x) ** 2 + (A_y - B_y) ** 2) ** 0.5


def increse(number):
    number = str(number)
    if len(number) == 5:
        return number
    elif len(number) == 4:
        return "0" + number
    elif len(number) == 3:
        return "00" + number
    elif len(number) == 2:
        return "000" + number
    elif len(number) == 1:
        return "0000" + number


def make_gif(path_to_plots, file_name):
    # Create the frames
    frames = []
    imgs = sorted(glob.glob(path_to_plots))
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(file_name + '.gif', format='GIF', save_all=True, duration=1, loop=0, append_images=frames[1:])


def prepare_data(file_to_change, new_file, delimiter, every_n_row):
    test_data = np.loadtxt(file_to_change, delimiter=delimiter)

    tmp = []

    for x in enumerate(test_data):
        if x[0] % every_n_row == 0:
            tmp.append(x[1])

    tmp = np.asarray(tmp)

    np.savetxt(new_file, tmp)
