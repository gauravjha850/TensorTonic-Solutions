import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split for classification labels.
    
    Parameters:
    y (array-like): Parent labels
    split_mask (array-like): Boolean mask where True indicates Left partition 
                             and False indicates Right partition.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)
    
    # Total number of samples
    N = y.size
    if N == 0:
        return 0.0
        
    # Split the labels into left and right subsets
    y_L = y[split_mask]
    y_R = y[~split_mask]
    
    # Handle edge case: If either side is empty, the split provides no information gain
    if y_L.size == 0 or y_R.size == 0:
        return 0.0
        
    # Calculate weights for the child nodes
    w_L = y_L.size / N
    w_R = y_R.size / N
    
    # Information Gain = H(Parent) - [w_L * H(Left) + w_R * H(Right)]
    ig = _entropy(y) - (w_L * _entropy(y_L) + w_R * _entropy(y_R))
    
    return float(ig)

    
