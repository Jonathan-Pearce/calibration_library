
def general_calibration_error(prob, labels, n_bins = 15, adaptive_bins = False, class_conditional = False, top_k_class = 1, norm = 1, thresholding = 0.0):

    n_data = np.size(prob)
    n_classes = np.size(labels)

    predictions = np.argmax(prob, axis=1)
    accuracies = np.equal(predictions, labels)

    labels_binary_matrix = get_binary_matrix(prob, labels)
    predictions_binary_matrix = get_binary_matrix(prob, predictions)
    accuracies_binary_matrix = np.equal(predictions_binary_matrix, labels_binary_matrix)

    #Apply Top K classes criteria
    #TODO validation top_k_classes <= n_classes, top_k_class integer
    #TODO how to handle tiebreakers in get_prob_top_k (current code is correct I think, we want to remove 0's in tiebreakers)
    if top_k_class < n_classes:
        prob_top_k = get_prob_top_k(prob, top_k_class)
    elif top_k_class == n_classes:
        prob_top_k = prob


    #Apply Thresholding
    prob_threshold = prob_top_k[prob_top_k < thresholding] = -2

    #Apply class conditional logic
    if class_conditional:
        #loop over classes
        for i in range(n_classes):
            get_bin_scores(prob_array, labels_array, n_bins, adaptive_bins)

    else:
        #flatten and run
        prob_array = prob_threshold.flatten()
        accuracies_array = accuracies_binary_matrix.flatten()

        bin_score_dict = get_bin_scores(prob_array, accuracies_array, n_bins, adaptive_bins)

        bin_n = bin_score_dict['bin_n']
        bin_prop = bin_score_dict['bin_prop']
        bin_acc = bin_score_dict['bin_acc']
        bin_conf = bin_score_dict['bin_conf']
        bin_score = bin_score_dict['bin_score']

        gce = get_gce(bin_score, bin_prop, norm)


