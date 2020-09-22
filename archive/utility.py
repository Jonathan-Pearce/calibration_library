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


