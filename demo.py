import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.special import softmax

import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.init as init

import torchvision
import torchvision as tv
import torchvision.transforms as transforms

sys.path.insert(1, 'models/')
from models import resnet
import metrics
import recalibration
import visualization

np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = './pretrained_models/cifar10_resnet20.pth'
net_trained = resnet.ResNet(resnet.BasicBlock, [3, 3, 3])
net_trained.load_state_dict(torch.load(PATH))
net_trained.to(device)

# Data transforms

mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]

test_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=mean, std=stdv),
])

# IMPORTANT! We need to use the same validation set for temperature
# scaling, so we're going to save the indices for later
test_set = tv.datasets.CIFAR10(root='./data', train=False, transform=test_transforms, download=True)
testloader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=4)

correct = 0
total = 0

# First: collect all the logits and labels for the validation set
logits_list = []
labels_list = []


with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        #outputs are the the raw scores!
        logits = net_trained(images)
        #add data to list
        logits_list.append(logits)
        labels_list.append(labels)
        #convert to probabilities
        output_probs = F.softmax(logits,dim=1)
        #get predictions from class
        probs, predicted = torch.max(output_probs, 1)
        #total
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
print(total)

################
#metrics

ece_criterion = metrics.ECELoss()
#Torch version
logits_np = logits.cpu().numpy()
labels_np = labels.cpu().numpy()

#Numpy Version
print('ECE: %f' % (ece_criterion.loss(logits_np,labels_np, 15)))

softmaxes = softmax(logits_np, axis=1)

print('ECE with probabilties %f' % (ece_criterion.loss(softmaxes,labels_np,15,False)))

mce_criterion = metrics.MCELoss()
print('MCE: %f' % (mce_criterion.loss(logits_np,labels_np)))

oe_criterion = metrics.OELoss()
print('OE: %f' % (oe_criterion.loss(logits_np,labels_np)))

sce_criterion = metrics.SCELoss()
print('SCE: %f' % (sce_criterion.loss(logits_np,labels_np, 15)))

ace_criterion = metrics.ACELoss()
print('ACE: %f' % (ace_criterion.loss(logits_np,labels_np,15)))

tace_criterion = metrics.TACELoss()
threshold = 0.01
print('TACE (threshold = %f): %f' % (threshold, tace_criterion.loss(logits_np,labels_np,threshold,15)))



############
#recalibration

model = recalibration.ModelWithTemperature(net_trained)
# Tune the model temperature, and save the results
model.set_temperature(testloader)


############
#visualizations

conf_hist = visualization.ConfidenceHistogram()
plt_test = conf_hist.plot(logits_np,labels_np,title="Confidence Histogram")
plt_test.savefig('plots/conf_histogram_test.png',bbox_inches='tight')
#plt_test.show()

rel_diagram = visualization.ReliabilityDiagram()
plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Diagram")
plt_test_2.savefig('plots/rel_diagram_test.png',bbox_inches='tight')
#plt_test_2.show()


