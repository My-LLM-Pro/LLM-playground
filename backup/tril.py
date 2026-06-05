import torch

seq = 5
casual_mask = torch.tril(torch.ones(seq, seq))
print(casual_mask)
casual_mask = torch.triu(torch.ones(seq, seq))
print(casual_mask)