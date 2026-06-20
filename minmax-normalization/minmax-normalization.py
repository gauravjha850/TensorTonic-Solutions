import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    
    X=np.asarray(X,dtype=float)

    X_min=np.min(X,axis=axis,keepdims=True)
    X_max=np.max(X,axis=axis,keepdims=True)

    denominator=X_max-X_min
    denominator=np.maximum(denominator,eps)

    return (X-X_min)/denominator

    