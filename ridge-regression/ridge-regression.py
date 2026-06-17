def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-fsolution.
    
    """
    X=np.array(X)
    y=np.array(y)

    w=np.linalg.inv(X.T@X + lam*np.eye(X.shape[1]))@(X.T)@y
    return w
    