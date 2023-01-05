import torch
import math
from torch import nn
import torch.nn.functional as F
import os

''' Multi Layer Perceptron
'''
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)

''' SubGraph in VectorNet
'''
class MapNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, feature_size=10):
        super(MapNet, self).__init__()
        local_graph=[]
        for i in range(3):
            local_graph.append(SubGraph(in_dim, hidden_dim, hidden_dim, feature_size))
            in_dim = hidden_dim
        self.local_graph = nn.ModuleList(local_graph)

    def forward(self, lane_graph):
        whole_polys_feature = lane_graph
        for i in range(3):
            whole_polys_feature = self.local_graph[i](whole_polys_feature)

        kernel_size = whole_polys_feature.shape[1]
        maxpool = nn.MaxPool1d(kernel_size)
        poly_feature = maxpool(whole_polys_feature.transpose(1,2)).squeeze()
        return poly_feature

''' SubGraph in VectorNet
'''
class SubGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, feature_size=10):
        super(SubGraph, self).__init__()
        self.feature_size = feature_size
        self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim//2),
                nn.LayerNorm(out_dim//2),
                nn.ReLU()
                )
    def forward(self, polylines):
        whole_polys_feature = self.mlp(polylines)

        kernel_size = whole_polys_feature.shape[1]
        maxpool = nn.MaxPool1d(kernel_size)
        poly_feature = maxpool(whole_polys_feature.transpose(1,2)).transpose(1,2)

        whole_polys_feature = torch.cat([whole_polys_feature, poly_feature.repeat(1,self.feature_size,1)], -1)
        return whole_polys_feature

''' MultiHead Attention Layer
'''
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1, d_reduce=None):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        if d_reduce != None:
            self.q_linear = nn.Linear(d_reduce, d_model)
        else:
            self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # self.norm = nn.LayerNorm(d_model)
        
        if d_reduce != None:
            self.out = nn.Linear(d_model, d_reduce)
        else:
            self.out = nn.Linear(d_model, d_model)
    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        self.scores = scores

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, iq, ik, iv, attn_mask=None):

        bs = iq.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(ik).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(iq).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(iv).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, attn_mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        # ResNet structure
        concat = concat + iq
        # concat = self.norm(concat)

        output = self.out(concat) + concat

        return output
