import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

import math

@dataclass
class GPTconfig:
    max_seq_len: int = 512 # the most token numbers can be processed by GPT
    batch_size: int = 12 
    n_layer: int = 6 # the number of transformer block
    n_head: int = 12 # the number of heads in multi-head attention
    n_embed: int = 768 # the dimention of embedding
    head_size: int = n_embed // n_head # the dimention of every head
    dropout: float = 0.1 # the dropout rate
    vocab_size: int = 50257 # the vocabulary size of GPT-2


class SingleHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head_size = cfg.head_size
        self.queue_proj = nn.Linear(cfg.n_embed, cfg.head_size)
        self.key_proj = nn.Linear(cfg.n_embed, cfg.head_size)
        self.value_proj = nn.Linear(cfg.n_embed, cfg.head_size)
        self.register_buffer(
            'causal_mask',
            torch.tril(
                cfg.max_seq_len, cfg.max_seq_len
            ) 
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.shape
        Q = self.queue_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        attn = Q @ K.transpose(-2,-1)/math.sqrt(self.head_size)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask==0,float('-inf'))
        attn = attn.masked_fill(self.causal_mask[:seq_len, :seq_len]==0,float('-inf'))
        attn = torch.softmax(attn,dim=-1)
        out = attn @ V
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(cfg) for _ in cfg.n_head]
        )
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out_list = [head(x) for head in self.heads]
        out = torch.concat(out_list, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        out = out.view(cfg.batch_size, cfg.max_seq_len, -1)

        return out


class Feedforward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Linear(cfg.n_embed, cfg.n_embed * 4),
            nn.ReLU(),
            nn.Linear(cfg.n_embed * 4, cfg.n_embed),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        x = self.ffd(x)
        return x

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_layer = MultiHeadAttention(cfg)
        self.ffd = Feedforward(cfg)
        self.norm1 = nn.LayerNorm(cfg.n_embed)
        self.norm2 = nn.LayerNorm(cfg.n_embed)

    def forward(self, x):
        x = self.norm1(x + self.attn_layer(x))
        x = self.norm2(x + self.ffd(x))

        return x

class MiniGPT(nn.Module):
    pass