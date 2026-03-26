import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    
    """
    Generate sinusoidal positional encodings.
    """
    """ seq_length: Number of positions (sequence length)
        d_model: Dimensionality of the model/embeddings
    
    Returns:
        Positional encoding matrix of shape (seq_length, d_model) """
    
    
    pe=np.zeros((seq_length,d_model))  
# this line initilize the output matrix that will hold all the positional encoding
    positions =np.arange(seq_length).reshape(-1,1)
    dims =np.arange(0,d_model,2).reshape(1,-1)
    div_term=np.exp(dims*(-np.log(10000)/d_model))

    pe[:,0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions*div_term)


    return pe
    
    
    