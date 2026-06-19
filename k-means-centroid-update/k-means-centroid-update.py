import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points=np.array(points)
    assignments=np.array(assignments)

    num_features=points.shape[1]
    new_center=[]

    for cluster_idx in range (k):
        cluster_points=points[assignments==cluster_idx]

        if len(cluster_points)==0:
            centroid=np.zeros(num_features)
        else:
            centroid=np.mean(cluster_points,axis=0)
        new_center.append(centroid.tolist())
    return new_center

    