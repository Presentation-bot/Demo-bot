from Bricks.graph_bricks.spatial.GAT import GAT
from Bricks.graph_bricks.temporal.DilatedTCN import DilatedTCN

import torch
import torch.nn as nn
import torch.nn.functional as F

class STModel(nn.Module):
    def __init__(self, num_nodes, input_seq_length, horizon, spatial_layers, temporal_layers, input_dim, output_dim, fusion_type, adj_idx, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_seq_length = input_seq_length
        self.horizon = horizon
        self.adj = adj_idx
        self.fusion_type = fusion_type

        hidden_dim = int(hidden_dim)

        self.spatial_blocks = nn.ModuleList([
            GAT(num_nodes, input_seq_length, hidden_dim, input_seq_length, adj_idx)
            for _ in range(spatial_layers)
        ])

        self.bridge = nn.Linear(input_seq_length, horizon)

        self.temporal_blocks = nn.ModuleList([
            DilatedTCN(horizon)
            for _ in range(temporal_layers)
        ])

        self.fusion = nn.Linear(horizon * 2, horizon)

    def forward(self, x):
        for spatial in self.spatial_blocks:
            x = F.relu(spatial(x))
        spatial_out = self.bridge(x)
        temporal_out = spatial_out
        for temporal in self.temporal_blocks:
            temporal_out = F.relu(temporal(temporal_out))

        if self.fusion_type == 'concat':
            fusion_input = torch.cat([spatial_out, temporal_out], dim=-1)
            fused = self.fusion(fusion_input)
        else:
            fused = spatial_out + temporal_out
        return fused
