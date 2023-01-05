import torch
from torch import nn
import torch.nn.functional as F

from .conditional_subnet import MLP, MultiheadAttention, MapNet

class Conditional(nn.Module):
    def __init__(self):
        super(Conditional, self).__init__()
        self.motion_encoder = MLP(300, 240, 240)
        self.lane_encoder = MapNet(2, 240, 240, 10)
        self.lane_attn = MultiheadAttention(240, 8)
        self.neighbor_encoder = MLP(66, 240, 240)
        self.neighbor_attn = MultiheadAttention(240, 8)

    def forward(self, data):
        batch_size = data['past_traj'].shape[0]

        x = data['past_traj'].reshape(-1, 300) #50*6
        x = self.motion_encoder(x)

        lane = data['lane']
        lane = self.lane_encoder(lane)
        
        neighbor = data['neighbor'].reshape(-1, 66) #11*6
        neighbor = self.neighbor_encoder(neighbor)
        
        x = x.unsqueeze(0)
        lane = lane.unsqueeze(0)
        
        lane_mask = data['lane_mask']
        x = self.lane_attn(x, lane, lane, attn_mask=lane_mask) 
        
        neighbor_mask = data['neighbor_mask']
        x = self.neighbor_attn(x, neighbor, neighbor, attn_mask=neighbor_mask) 
        return x.reshape(batch_size, -1, 60, 4)
