import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    
    """
    points=np.array(points)
    centroids=np.array(centroids)

    diff=points[:,np.newaxis,:]- centroids

    distances=np.sum(diff**2,axis=2)

    return np.argmin(distances,axis=1).tolist()

    