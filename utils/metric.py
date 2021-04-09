import numpy as np

def euclidean_distance(x, y):
    return np.mean(np.square(x-y), axis=-1)