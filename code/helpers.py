
def get_top_k(prob, k):
    #iterate over each vector in prob
    for i in len(prob):
        #get kth highest value
        prob_k = np.partition(prob[i], -k)[-k]
        #filter values
        prob[i][prob[i] < prob_k] = -2.0

    return (prob)

def get_threshold(prob, threshold):
    #filter values
    prob[prob < threshold] = -3.0

def get_gce(prob, label, n_bins, adaptive_bins):
    
    #calculate bin boundaries (lower and upper)
    if adaptive_bins:
        get_adaptive_bin_boundaries(prob, n_bins)
    else:
        get_uniform_bin_boundaries(n_bins)

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]



def get_uniform_bin_boundaries(n_bins):
    #uniform bin spacing
    return(np.linspace(0, 1, self.n_bins + 1))

def get_adaptive_bin_boundaries(n_bins, prob):
    #size of bins 
    bin_n = int(len(prob)/n_bins)

    bin_boundaries = np.array([])
    prob_sort = np.sort(prob)  

    for i in range(0,self.n_bins):
        bin_boundaries = np.append(bin_boundaries,prob_sort[i*bin_n])
    bin_boundaries = np.append(bin_boundaries,1.0)

    return(bin_boundaries)