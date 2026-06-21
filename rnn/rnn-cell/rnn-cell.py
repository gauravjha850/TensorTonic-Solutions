import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    c=np.dot(x_t,W_xh.T)
    k=np.dot(h_prev,W_hh.T)
    return np.tanh(c+k+b_h)
    