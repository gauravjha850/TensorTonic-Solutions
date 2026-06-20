import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Convert inputs to NumPy arrays for vectorized operations
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    # Identify the unique classes
    classes = np.unique(y_train)
    num_samples = X_train.shape[0]
    
    # Dictionaries to store training statistics per class
    priors = {}
    means = {}
    variances = {}
    
    epsilon = 1e-9
    
    # 1. Training Phase: Compute priors, means, and variances
    for c in classes:
        # Filter training data belonging to class c
        X_c = X_train[y_train == c]
        
        # Prior probability: P(c) = n_c / n
        priors[c] = X_c.shape[0] / num_samples
        
        # Mean of each feature given class c
        means[c] = np.mean(X_c, axis=0)
        
        # Population variance of each feature given class c (+ epsilon)
        variances[c] = np.var(X_c, axis=0) + epsilon

    # 2. Inference Phase: Compute log posteriors for each test sample
    predictions = []
    
    for x in X_test:
        log_posteriors = {}
        
        for c in classes:
            # Start with log of the prior: log P(c)
            log_prior = np.log(priors[c])
            
            # Retrieve parameters for class c
            mean = means[c]
            var = variances[c]
            
            # Vectorized Gaussian log-likelihood calculation for all features:
            # -0.5 * log(2 * pi * var) - ((x - mean)**2) / (2 * var)
            log_likelihood = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
            
            # Total log posterior proportional sum
            log_posteriors[c] = log_prior + np.sum(log_likelihood)
            
        # Select the class with the highest log posterior probability
        best_class = max(log_posteriors, key=log_posteriors.get)
        predictions.append(int(best_class))
        
    return predictions
    