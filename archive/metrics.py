class CELoss(object):

    def __init__(self, n_bins, uniform_range):
        """
        n_bins (int): number of confidence interval bins
        """
        self.n_bins = n_bins
        if uniform_range:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def bin_boundaries(self):
        return 0

    def get_probabilities(self, output, labels, probabilities):

        #If not probabilities apply softmax!
        if not probabilities:
            softmaxes = softmax(output, axis=1)
        else:
            softmaxes = output

        self.confidences = np.max(softmaxes, axis=1)
        self.predictions = np.argmax(softmaxes, axis=1)
        self.accuracies = np.equal(self.predictions,labels)

    def compute_bins(self):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(self.confidences,bin_lower.item()) * np.less_equal(self.confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(self.accuracies[in_bin])
                self.bin_conf[i] = np.mean(self.confidences[in_bin])
                self.bin_score[i] = np.abs(avg_confidence_in_bin - self.bin_acc[i])


#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)

    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)
    
    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.max(self.bin_score)


#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)

    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))

#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):
    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)

    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.max(self.bin_scores)

class ACELoss(CELoss):
    def __init__(self, n_bins=15):
        super().__init__(n_bins, False)

    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.max(self.bin_scores)

class TACELoss(CELoss):
    def __init__(self, n_bins=15):
        super().__init__(n_bins, False)

    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.max(self.bin_scores)