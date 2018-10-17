import torch
import torch.nn.functional as F


a = torch.rand(5,6).cuda()
print(a)
print("*"*10)
print(F.softmax(a,dim=-1))