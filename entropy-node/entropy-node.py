import numpy as np

def entropy_node(y):
    
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0
    unique,count=np.unique(y, return_counts =True)
    prob=count/len(y)
    return -(np.sum(prob*(np.log2(prob))))
    