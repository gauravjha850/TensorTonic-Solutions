import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    X_train =np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)

    n_samples,n_features= X_train.shape
    classes =np.unique(y_train)

    class_log_priors = []
    feature_log_probs = []


    for c in classes:
        X_c = X_train[y_train==c]
        n_c = X_c.shape[0]

        class_log_priors.append(np.log(n_c/n_samples))



        counts = np.sum(X_c, axis=0)


        theta=  (counts + 1 )/(n_c + 2)


        feature_log_probs.append({
            'log_pos' : np.log(theta), 'log_neg' : np.log(1-theta)
        })
    all_test_scores = []
    for row in X_test :
        row_scores = []

        for i in range (len(classes)):
            current_score = class_log_priors[i]


            for  j in range (n_features):
                if row[j]==1:
                    current_score+=feature_log_probs[i]['log_pos'][j]
                else:
                    current_score+=feature_log_probs[i]['log_neg'][j]
            row_scores.append(current_score)
        all_test_scores.append(row_scores)

    return np.array(all_test_scores)
                
                
                
    