import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.asarray(matrix, dtype=float)
    
    # 1. Validation check (Return None for invalid/empty inputs or axes)
    if matrix.ndim != 2 or matrix.size == 0:
        return None
        
    if axis not in [0, 1, None]:
        return None
        
    # 2. Direct Norm Calculations
    if norm_type == 'l1':
        norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == 'l2':
        norm = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
    elif norm_type == 'max':
        norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else:
        return None
        
    # 3. Handle zero vectors safely to avoid division by zero
    norm_safe = np.where(norm == 0, 1.0, norm)
    
    # 4. Divide and return the result
    return matrix / norm_safe
    
        