import numpy as np

data = np.load("/workspaces/calibration-toolbox/data/cifar10h-probs.npy")

print(data)

print(data[1])

print(len(data))

k=1
prob_k = np.partition(data[1], -k)[-k]
print(prob_k)
print(np.shape(prob_k))

data[1][data[1] < prob_k] = 0.0
print(data[1]) 