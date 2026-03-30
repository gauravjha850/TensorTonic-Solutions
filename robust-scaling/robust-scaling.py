def robust_scaling(arr):
    """
    Scale values using median and interquartile range.
    """
    
    arr_sorted=sorted(arr)
    n=len(arr)
    if n<=1:
        return [0 for _ in arr]
    if n%2==0:
        median=(arr_sorted[n//2-1]+arr_sorted[n//2])/2.0
    else:
        median=(arr_sorted[n//2])
    mid=n//2
    if n%2==0:
        lower_half=arr_sorted[:mid]
        upper_half=arr_sorted[mid:]
    else:
        lower_half=arr_sorted[:mid]
        upper_half=arr_sorted[mid+1:]
    def find_median(sub):
        m=len(sub)
        if m==0:
            return 0

        
        if m % 2 == 0:
            
            return (sub[m//2-1]+sub[m//2])/2.0
        else:
            return (sub[m//2])
    Q3=find_median(upper_half)
    Q1=find_median(lower_half)
    IQR=Q3-Q1
    if IQR==0:
        return [0 for _ in arr]
    return [(x-median)/IQR for  x in arr]
        