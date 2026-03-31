import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x=np.array(x,dtype=float)
    if x.ndim == 1:
        exp_x=np.exp(x-np.max(x))
        result=exp_x/np.sum(exp_x)
    elif x.ndim == 2:
        exp_x= np.exp(x-np.max(x,axis=1, keepdims=True))
        result=exp_x/np.sum(exp_x,axis=1,keepdims=1)
    else:
        return None
    return np.round(result,4).tolist()
    
    
        