import torch

attn = torch.arange(0,6).view(2,3)
print(attn)
mask = torch.tensor([[True,True,True],[False,False,False]])
attn = attn.masked_fill(mask, -1)
print(attn)