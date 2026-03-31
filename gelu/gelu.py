import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x=np.asarray(x,dtype=float)
    
    
    result=(0.5)*(x)*(1+np.vectorize(math.erf)(x/np.sqrt(2)))
    return result 
