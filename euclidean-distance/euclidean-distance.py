import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.

    """
    x=np.array(x)
    y=np.array(y)
    return float(np.sqrt(np.sum((x-y)**2)))