import torch
print(torch.version)

a=torch.randn(1,3,5)
print(a.max(1))

