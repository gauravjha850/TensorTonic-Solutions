def jaccard_similarity(set_a, set_b):
    set_a=set(set_a)
    set_b=set(set_b)
    t=len(set_a.intersection(set_b))
    k=len(set_a.union(set_b))
    if k==0:
        return 0.00
    else:
        return t/k
    
    
    