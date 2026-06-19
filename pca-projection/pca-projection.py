import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X=np.array(X)

    X_mean=np.mean(X,axis=0)
    X_centered=X-X_mean
    n=X.shape[0]

    covariance_matrix=(X_centered.T@X_centered)/(n-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance_matrix)
    idx=np.argsort(eigenvalues)[::-1]
    top_k_eigenvectors=eigenvectors[:, idx[:k]]
    Xprojected=X_centered@top_k_eigenvectors
    return Xprojected.tolist()
    
    