import numpy as np
def calculate_gini_old_style(y):
        if len(y)==0:
            return 0.0 
        label_counts={}
        for label in y:
            if label in label_counts:
                label_counts[label]+=1
            else:
                label_counts[label]=1
        n=len(y)
        gini=1.0

        for count in label_counts.values():
            prob=count/n

            gini= gini - prob**2
        return gini

def decision_tree_split(X, y):
    X=np.array(X)
    y=np.array(y)
    """
    Find the best feature and threshold to split the data.
    """
    n_samples, n_features=X.shape
    best_gini=float('inf')
    best_feature=None
    best_threshold =None
    
    
    # if all samples have the same label no need to split  
    if len(np.unique(y))<=1:
        return [None,None] 
    for feature in range(n_features):
        feature_values = X[:,feature]
        sorted_values = np.sort(np.unique(feature_values))
        thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
        


        for threshold in thresholds:
            left_mask = feature_values <= threshold

            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = n_samples - n_left

            if n_left == 0 or n_right == 0:
                continue
            


            

            

            

                    
            left_gini=calculate_gini_old_style(y[left_mask])
            
            right_gini=calculate_gini_old_style(y[right_mask])
            
            weighted_gini= (n_left/n_samples)*left_gini + (n_right/n_samples)*right_gini
            if weighted_gini < best_gini :
                best_gini= weighted_gini
                best_feature=feature
                best_threshold= threshold
    return [best_feature, best_threshold ]
    
            
        
        
            
            

            
    
    
    