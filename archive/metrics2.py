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

    def get_probabilities(self, output, labels, logits):

        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)
        self.n_data = len(self.output)
        self.n_class = len(self.output[0])

    def binary_matrices(self):
        #make matrices of zeros
        self.pred_matrix = np.zeros([self.n_data,self.n_class])
        self.label_matrix = np.zeros([self.n_data,self.n_class])
        self.acc_matrix = np.zeros([self.n_data,self.n_class])

        idx = np.arange(self.n_data)
        self.pred_matrix[idx,self.predictions] = 1
        self.label_matrix[idx,self.labels] = 1
        self.acc_matrix = np.equal(pred_matrix,label_matrix)


    def compute_bins(self, confidences, accuracies):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(avg_confidence_in_bin - self.bin_acc[i])


#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)

    def loss(self, output, labels, logits = True):
        super().get_probabilities(output, labels, logits)
        super().compute_bins(self.confidences, self.accuracies)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)
    
    def loss(self, output, labels, logits = True):
        super().get_probabilities(output, labels, logits)
        super().compute_bins(self.confidences, self.accuracies)
        return np.max(self.bin_score)


#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)

    def loss(self, output, labels, logits = True):
        super().get_probabilities(output, labels, logits)
        super().compute_bins(self.confidences, self.accuracies)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))

#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):
    def __init__(self, n_bins=15):
        super().__init__(n_bins, True)

    def loss(self, output, labels, logits = True):
        sce = 0.0

        super().get_probabilities(output, labels, logits)
        super().self.binary_matrices()

        for i in range(self.n_class):
            #ECELoss(self.probabilities[i,:],self.acc_matrix[i,:])
            #ith row or column? of matrices
            super().compute_bins(self.probabilities[i,:],self.acc_matrix[i,:])
            sce += np.dot(self.bin_prop,self.bin_score)

        return sce/self.n_class


#create TACELoss with threshold fixed at 0!!
class ACELoss(CELoss):
    def __init__(self, n_bins=15):
        TACELoss = TACELoss(0)

    def loss(self, output, labels, logits = True):
        return TACELoss.loss(output, labels, logits)


class TACELoss(CELoss):
    def __init__(self, threshold, n_bins=15):
        self.threshold = threshold
        super().__init__(n_bins, False)

    def loss(self, output, labels, logits = True):
        sce = 0.0

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < self.threshold] = 0
        super().self.binary_matrices()

        for i in range(self.n_class):
            #ECELoss(self.probabilities[i,:],self.acc_matrix[i,:])
            #ith row or column? of matrices
            compute_ranges()
            super().compute_bins(self.probabilities[i,:],self.acc_matrix[i,:])
            sce += np.dot(self.bin_prop,self.bin_score)

        return sce/self.n_class