import torch
import torch.nn as nn
import math

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.queue_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        #? x shape: batch, num_tokens, dimensions
        Q = self.queue_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        #? attn shape: batch, num_tokens, num_tokens
        attn = torch.matmul(Q, K.transpose(-1,-2))/math.sqrt(hidden_dim)
        attn = torch.softmax(attn, dim=-1)
        #? attn shape:
        out = torch.matmul(attn, V)
        return out


class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim*3)

    def forward(self, x):
        #? x shape: batch, num_tokens, dimensions
        #? QKV shape: batch, dimensions, dimensions*3
        QKV = self.proj(x)
        Q,K,V = torch.split(QKV,hidden_dim,dim=-1)
        #? attn shape: batch, num_tokens, num_tokens
        attn = torch.matmul(Q, K.transpose(-1,-2))/math.sqrt(hidden_dim)
        attn = torch.softmax(attn, dim=-1)
        #? attn shape:
        out = torch.matmul(attn, V)
        return out


class SelfAttentionV3(nn.Module):
    def __init__(self, hidden_dim, attention_drop=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim*3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_out = nn.Dropout(attention_drop)

    def forward(self, x, attention_mask=None):
        #? x shape: batch, num_tokens, dimensions
        #? QKV shape: batch, dimensions, dimensions*3
        QKV = self.proj(x)
        # Q,K,V = torch.split(QKV,hidden_dim,dim=-1)
        Q,K,V = QKV.chunk(3, dim=-1)
        #? attn shape: batch, num_tokens, num_tokens
        attn = torch.matmul(Q, K.transpose(-1,-2))/math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask==0,float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop_out(attn)
        #? attn shape:
        out = torch.matmul(attn, V)
        out = self.out_proj(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_drop=0.1):
        super().__init__()
        self.queue_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_out = nn.Dropout(attention_drop)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

    def forward(self, x, attention_mask=None):
        #? x: batch, n_tokens, sequence
        batch, n_tokens, _ = x.shape
        Q = self.queue_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        #? x_state: batch, head_num, n_tokens, head_dim
        q_state = Q.view(batch, n_tokens, self.head_num, self.head_dim).transpose(1,2)
        k_state = K.view(batch, n_tokens, self.head_num, self.head_dim).transpose(1,2)
        v_state = V.view(batch, n_tokens, self.head_num, self.head_dim).transpose(1,2)
        #? attn_weight: batch, head_num, n_tokens, n_tokens
        attn_weight = q_state @ k_state.transpose(-1,-2)/math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weight = attn_weight.masked_fill(attention_mask==0,float('-inf'))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.drop_out(attn_weight)
        out = attn_weight @ v_state
        out = out.transpose(1,2).contiguous()
        out = out.view(batch, n_tokens, -1)
        out = self.out_proj(out)

        return out



        
