import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    pred_array=np.array(predictions)

    num_samples=pred_array.shape[1]

    final_pred=[]

    for i in range (num_samples):
        sample_votes= pred_array[:,i]

        classes,counts=np.unique(sample_votes,return_counts=True)

        max_votes=np.max(counts)

        winning_classes=classes[counts== max_votes]

        best_class= np.min(winning_classes)

        final_pred.append(int(best_class))
    return final_pred

        
        
        