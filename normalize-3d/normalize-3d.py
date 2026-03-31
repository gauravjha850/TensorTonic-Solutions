import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v=np.array(v,dtype=float)
    norm=np.sqrt(np.sum(v*v,axis=-1, keepdims=True))
    return np.where(norm>0,v/norm,v)
    