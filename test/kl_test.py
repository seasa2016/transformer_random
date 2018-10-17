import torch
import torch.nn as nn
import torch.nn.functional as F


a = torch.randn(4,5)
b = torch.randn(4,5)

out = F.kl_div(a,b,size_average=False)
print(out)

out = F.kl_div(a,b,reduction="sum")
print(out)