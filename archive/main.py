import numpy as np
import matplotlib.pyplot as plt


from scipy.special import softmax

import torch
from torch import nn, optim
from torch.nn import functional as F

import torchvision
import torchvision as tv
import torchvision.transforms as transforms

np.random.seed(0)
torch.manual_seed(0)


PATH = './cifar_net.pth'
net_trained = Net()
net_trained.load_state_dict(torch.load(PATH))

#########################################################################

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

#########################################################################

correct = 0
total = 0

# First: collect all the logits and labels for the validation set
logits_list = []
labels_list = []

with torch.no_grad():
    for images, labels in testloader:
        #outputs are the the raw scores!
        logits = net_trained(images)
        #add data to list
        logits_list.append(logits)
        labels_list.append(labels)
        #convert to probabilities
        output_probs = F.softmax(logits,dim=1)
        #get predictions from class 
        probs, predicted = torch.max(output_probs.data, 1)
        #total
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
print(total)
print(logits.shape)
print(labels.shape)


#########################################################################

ece_criterion = ECELoss()
#Torch version
logits_np = logits.numpy()
labels_np = labels.numpy()

#Numpy Version
print(ece_criterion.loss(logits_np,labels_np))

softmaxes = softmax(logits_np, axis=1)

print(ece_criterion.loss(softmaxes,labels_np,True))

mce_criterion = MCELoss()
print(mce_criterion.loss(logits_np,labels_np))


confHist = ConfidenceHistogram()
confHist.plot(logits_np,labels_np,15)

relDia = ReliabilityDiagram()
relDia.plot(logits_np,labels_np)

#########################################################################

net_calibrate = ModelWithTemperature(net_trained)
net_calibrate.set_temperature(testloader)
