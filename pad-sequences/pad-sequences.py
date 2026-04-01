import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs :
        return np.empty((0,0),dtype=object)
    N=len(seqs)

    if max_len is None :
        max_len=max(len(seq) for seq in seqs)
        
    elif max_len < 0 :
        raise ValueError(" please enter a vaild word ")
    L=max_len

    padded = np.full((N,L),pad_value,dtype=object)

    for i , seq in enumerate(seqs):
        
        
        seq_len= min(len(seq),max_len)
        padded[i, :seq_len]=list(seq)[:seq_len]
    return padded
    
        
    