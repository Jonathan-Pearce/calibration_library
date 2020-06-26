class CELoss(object):

    def __init__(self, n_bins):
        """
        n_bins (int): number of confidence interval bins
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

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
        self.bin_scores = np.zeros(len(self.bin_lowers))
        self.bin_prop = np.zeros(len(self.bin_lowers))
        self.bin_acc = np.zeros(len(self.bin_lowers))

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
        #for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(self.confidences,bin_lower.item()) * np.less_equal(self.confidences,bin_upper.item())
            prop_in_bin = np.mean(in_bin)
            self.bin_prop[i] = prop_in_bin

            if prop_in_bin.item() > 0:
                self.bin_acc[i] = np.mean(self.accuracies[in_bin])
                avg_confidence_in_bin = np.mean(self.confidences[in_bin])
                self.bin_scores[i] = np.abs(avg_confidence_in_bin - self.bin_acc[i])


class ECELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins)

    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.dot(self.bin_scores,self.bin_prop)

class MCELoss(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins)
    
    def loss(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()
        return np.max(self.bin_scores)


class ConfidenceHistogram(CELoss):

    def __init__(self, n_bins=15):
        super().__init__(n_bins)

    def plot(self, output, labels, bins, vertical_lines = True, x_min = 0.0, x_max = 1.0, y_max = None, scaled = True, probabilities = False):
        assert bins > 0
        super().get_probabilities(output, labels, probabilities)

        n = len(labels)
        w = np.ones(n)
        if scaled:
            w /= n


        plt.rcParams["font.family"] = "serif"
        plt.figure(figsize=(3,3))
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)    

        plt.hist(self.confidences,bins,weights = w,color='b',range=(x_min,x_max),edgecolor = 'k')
        plt.ylim((0,y_max))

        if vertical_lines:
            #accuracy
            acc = np.mean(self.accuracies)
            #confidence
            conf = np.mean(self.confidences)
                           
            plt.axvline(x=acc, color='k', linestyle='--',label='H')
            plt.axvline(x=conf, color='k', linestyle='--',label='H')

            if acc > conf:
                plt.text(acc+0.01,0.3,'Accuracy',rotation=90)
                plt.text(conf-0.05,0.3,'Confidence',rotation=90)
            else:
                plt.text(acc-0.05,0.3,'Accuracy',rotation=90)
                plt.text(conf+0.01,0.3,'Confidence',rotation=90)


        plt.ylabel('% of Samples',fontfamily='serif',fontsize=13)
        plt.xlabel('Confidence',fontfamily='serif',fontsize=13)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title('Model on Dataset',fontfamily='serif',fontsize=16)
        plt.show()

    def get_bins(self):
        return 0

class ReliabilityDiagram(CELoss):

    def __init__(self, n_bins=10):
        super().__init__(n_bins)

    def plot(self, output, labels, probabilities = False):
        super().get_probabilities(output, labels, probabilities)
        super().compute_bins()

        x = np.arange(0,1,0.1)

        mid = np.linspace(0.05,0.95,10)

        error = np.abs(np.subtract(mid,self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        plt.figure(figsize=(3,3))
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        plt.bar(x, self.bin_acc, color = 'b', width=0.1,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=0.1,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        plt.ylabel('Accuracy',fontfamily='serif',fontsize=13)
        plt.xlabel('Confidence',fontfamily='serif',fontsize=13)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title('Model on Dataset',fontfamily='serif',fontsize=16)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='large')

        plt.show()


