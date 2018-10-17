import torch

a = torch.rand(5,6)
print(a)

idx = torch.tensor([2,3])

q = a[idx]
print(q)