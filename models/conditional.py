import torch
from torch import nn
import torch.nn.functional as F

from conditional_subnet import MLP, MultiheadAttention, MapNet

class Conditional(pl.LightningModule):
    def __init__(self):
        self.motion_encoder = MLP(300, 128, 128)
        self.lane_encoder = MapNet(2, 128, 128, 10)
        self.lane_attn = MultiheadAttention(128, 8)
        self.neighbor_encoder = MLP(66, 128, 128)
        self.neighbor_attn = MultiheadAttention(128, 8)

    def forward(self, data):
        x = data['x'].reshape(-1, 300) #50*6
        
        lane = data['lane_graph']
        lane = self.lane_encoder(lane)
        
        neighbor = data['neighbor_graph'].reshape(-1, 66) #11*6
        neighbor = self.neighbor_encoder(neighbor)
        
        x = x.unsqueeze(0)
        lane = lane.unsqueeze(0)
        
        lane_mask = data['lane_mask']
        x = self.lane_attn(x, lane, lane, attn_mask=lane_mask) 
        
        neighbor_mask = data['neighbor_mask']
        x = self.neighbor_attn(x, neighbor, neighbor, attn_mask=neighbor_mask) 
        return x
