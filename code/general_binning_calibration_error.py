#inputs
##numpy matrix of logits (require columns to sum to 1, i.e. probabilities)
##vector of answers
##bins (total or classwise)

#five properties of GCE
##class conditionality
##adaptive bin intervals
##max probability (note top k is from another paper)
##norm
##thresholding (how to work with top k above?)

#design decisions - class based?


def general_binning_calibration_error(prob, labels, n_bins = 15, class_cond = False, adaptive_bins = False, top_k_classes = 1, norm = 2, thresholding = 0.0):

    n_data = len(prob)
    n_classes = len(prob[0])

    #validate (each item sums to 1, etc.)
    ##class_cond boolean
    ##adaptive_bins boolean
    ##top_k integer (less than or equal to nrows of prob, or "all")
    ##norm non-negative integer or "inf"
    ##thresholding double [0,1)

    prob = get_prob_top_k(prob, top_k_classes)
    prob = get_threshold(prob, threshold)

    GCE = 0.0

    if(class_cond):
        for i in n_classes:
            class_i_prob = prob[:,i]
            class_i_label = labels[:,i]
            GCE += get_gce(class_i_prob, class_i_label, n_bins, adaptive_bins)
    else:
        GCE = get_gce(prob.flatten(), labels.flatten(), n_bins, adaptive_bins)

