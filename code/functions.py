

def get_gce(bin_score, bin_prop, norm):

    return(np.power(np.sum(np.multiply(bin_prop, np.power(np.abs(bin_score), norm))), (1.0/norm)))

def get_bin_scores(prob_array, accuracies_array, n_bins, adaptive_bins):

    #clean out negative values
    #neg_idx = np.where(array1 < 0)
    prob_array_subset = prob_array[prob_array < 0]
    accuracies_array_subset = accuracies_array[prob_array < 0]

    if adaptive_bins:
        bins_dicttionary = get_adaptive_bins(prob_array_subset, n_bins)
        bin_lowers = bins_dicttionary['bin_lowers']
        bin_uppers = bins_dicttionary['bin_lowers']
    else:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

    bin_n = np.zeros(n_bins)
    bin_prop = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_score = np.zeros(n_bins)

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = np.greater(prob_array_subset, bin_lower.item()) * np.less_equal(prob_array_subset, bin_upper.item())
        bin_n[i] = np.sum(in_bin)
        bin_prop[i] = np.mean(in_bin)
        if bin_prop[i].item() > 0:
            bin_acc[i] = np.mean(accuracies_array_subset[in_bin])
            bin_conf[i] = np.mean(prob_array_subset[in_bin])
            bin_score[i] = bin_conf[i] - bin_acc[i]

    d = dict()
    d['bin_prop'] = bin_prop
    d['bin_acc'] = bin_acc
    d['bin_conf'] = bin_conf
    d['bin_score'] = bin_score
    return(d)

def get_adaptive_bins(prob_array, n_bins):
    #size of bins 
    bin_n = int(np.size(prob_array)/n_bins)
    bin_boundaries = np.array([])

    probabilities_sort = np.sort(prob_array)  
    #TODO numpy way to avoid for loop
    for i in range(0,n_bins):
        bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
    bin_boundaries = np.append(bin_boundaries,1.0)

    d = dict()
    d['bin_lowers'] = bin_boundaries[:-1]
    d['bin_lowers'] = bin_boundaries[1:]
    return(d) 

def get_binary_matrix(prob, labels):
    labels_2d = np.zeros_like(prob)

    desired_cols = labels_2d.flatten()
    desired_rows = np.arange(len(desired_cols))
    labels_2d[desired_rows, desired_cols] = 1

    return(labels_2d)

def get_prob_top_k(prob, top_k_class):

    for i, inner_list in enumerate(prob):

        A = inner_list        
        idx = np.argpartition(A, -top_k_class)
        arr_threshold = np.max(A[idx[:-top_k_class]])
        
        A[A <= arr_threshold] = -1

        prob[i] = A
    
    return(prob)





