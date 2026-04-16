import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:


    W1_=np.dot(x,W1)
    total=W1_ + b1
    k=np.maximum(0,total)
    final=np.dot(k,W2)
    pre=final+b2
    return pre