import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size
        self.multi_head_attention = \
                nn.MultiheadAttention(h_size, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm([h_size])
        self.fc =  nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )

    def forward(self, x):
        x_ln = self.layer_norm(x)
        attention_value, _ = self.multi_head_attention(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.fc(attention_value) + attention_value

        return attention_value
    
class SAWrapper(nn.Module):
    def __init__(self, h_size, num_s):
        super().__init__()
        self.self_attention = nn.Sequential(*[SelfAttention(h_size) for _ in range (1)])
        self.num_s = num_s
        self.h_size = h_size

    def forward(self, x):
        x = x.view(-1, self.h_size, self.num_s * self.num_s).swapaxes(1, 2)
        x = self.self_attention(x)
        x = x.swapaxes(2, 1).view(-1, self.h_size, self.num_s, self.num_s)
        
        return x