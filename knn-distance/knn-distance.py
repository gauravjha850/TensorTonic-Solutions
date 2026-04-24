import numpy as np

def knn_distance(X_train, X_test, k):
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    if X_train.ndim==1 and X_train.size >0:
        X_train=X_train.reshape(-1,1)
    if X_test.ndim==1 and X_test.size >0:
        X_test=X_test.reshape(-1,1)
    results=[]
    for q in X_test:
        diff=X_train-q
        dist=np.sqrt(np.sum(diff**2,axis=1))

        k_nearest_indices=np.argsort(dist)[:k].tolist()
        while len(k_nearest_indices) < k :
            k_nearest_indices.append(-1)

        results.append(k_nearest_indices)
    return np.array(results,dtype=int).reshape(len(X_test),k)
    