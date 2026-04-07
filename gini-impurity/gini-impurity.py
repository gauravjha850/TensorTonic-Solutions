import numpy as np

def gini_impurity(y_left, y_right):
    def calculate_node_gini(y):
        n=len(y)
        if n==0:
            return 0.0

        _, count= np.unique(y,return_counts=True)

        prob=count/n

        return 1.0-np.sum(prob**2)
    gini_left= calculate_node_gini(y_left)
    gini_right= calculate_node_gini(y_right)

    n_left = len(y_left)
    n_right = len(y_right)

    total_n=n_left+n_right

    if total_n == 0:
        return 0.0
    weighted_gini= (n_left/total_n * gini_left) +(n_right/total_n*gini_right)
    return weighted_gini


        
        
        
    
    