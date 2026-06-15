from torch.utils.data import Dataset
import torch
import tiktoken

class LLMDataset(Dataset):
    def __init__(self, cfg):
        self.path = cfg.path
        self.training_data = []
        self.gt = []
        enc = tiktoken.get_encoding("gpt2")
        with open(cfg.path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = enc.encode(text)
        for i in range(0, len(tokens)-cfg.context_length, cfg.context_length):
            self.training_data.append(tokens[i:i+cfg.context_length])
            self.gt.append(tokens[i+1:i+cfg.context_length+1])

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return torch.Tensor(self.training_data[idx], dtype=torch.long), torch.Tensor(self.gt[idx], dtype=torch.long)
        

def llm_dataloader(torch.dataloader):
    pass

