import numpy as np

def leaky_relu(arr, alpha=0.01):
    
    arr=np.array(arr)
    return np.where(arr>=0,arr,alpha*arr)
       
       
    
    