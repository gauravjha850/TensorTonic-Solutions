import numpy as np

def majority_classifier(y_train, X_test):
    classes,counts=np.unique(y_train,return_counts=True)
    majority_index=np.argmax(counts)
    majority_class=classes[majority_index]

    predictions=np.full(len(X_test),majority_class,dtype =int)

    return predictions
    
    