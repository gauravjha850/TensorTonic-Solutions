import numpy as np
def sigmoid(x):
    return 1/(1+ np.exp(-x))    

def swish(x):
    x= np.asarray(x, dtype=float)
    result=np.zeros_like(x)
    for i in range(len(x)):
        
        result[i]=sigmoid(x[i])*x[i]
        
    
    return np.round(result,4).tolist()


        