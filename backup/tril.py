import torch

seq = 5
casual_mask = torch.tril(torch.ones(seq, seq))
print(casual_mask)
casual_mask = torch.tril(torch.arange(0,9).view(3,3))
print(casual_mask)