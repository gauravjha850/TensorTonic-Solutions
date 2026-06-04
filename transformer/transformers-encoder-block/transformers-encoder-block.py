import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply layer normalization across the last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray, 
                         num_heads: int) -> np.ndarray:
    """Multi-head attention using np.matmul to handle 4D dimensions correctly."""
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # 1. Project inputs to Q, K, V matrices
    q_proj = np.dot(Q, W_q)  
    k_proj = np.dot(K, W_k)  
    v_proj = np.dot(V, W_v)  
    
    # 2. Reshape and transpose to split into heads: (batch, num_heads, seq, d_k)
    q_heads = q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    k_heads = k_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    v_heads = v_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # 3. Scaled dot-product attention
    scores = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    
    # Compute attention context matrix
    context_heads = np.matmul(attn_weights, v_heads)
    
    # 4. Concatenate heads back to original shape: (batch, seq, d_model)
    context = context_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # 5. Final output linear projection
    output = np.dot(context, W_o)
    return output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Position-wise feed-forward network with ReLU activation."""
    hidden = np.maximum(0, np.dot(x, W1) + b1)
    output = np.dot(hidden, W2) + b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray, 
                  W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, 
                  gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, 
                  num_heads: int) -> np.ndarray:
    """Complete encoder block: MHA + FFN with residuals and layer norms."""
    # Sublayer 1: Multi-Head Attention + Residual Connection + Layer Norm
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_prime = layer_norm(x + attn_out, gamma1, beta1)
    
    # Sublayer 2: Feed-Forward Network + Residual Connection + Layer Norm
    ffn_out = feed_forward(x_prime, W1, b1, W2, b2)
    output = layer_norm(x_prime + ffn_out, gamma2, beta2)
    
    return output