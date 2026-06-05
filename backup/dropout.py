import torch

dropout = torch.nn.Dropout(0.1)
example = torch.ones(6,6)
print(dropout(example))