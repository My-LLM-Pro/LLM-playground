import torch

x = torch.rand((4,5))
mean = x.mean(dim=-1,keepdim=True)
var = x.var(dim=-1,keepdim=True,unbiased=False)

print(x)
print(mean)
print(var)

print(mean.shape)
print(var.shape)