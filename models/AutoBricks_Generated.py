# ========== Spatial Module ==========
import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim,input_seq_length,adj_idx, hidden_dim=64):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes
        self.input_seq_length = input_seq_length
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Ensure adjacency is a float tensor
        self.adj = adj_idx.clone().detach().float() if isinstance(adj_idx, torch.Tensor) else torch.tensor(adj_idx, dtype=torch.float32)

        # Input projection: project temporal dim to hidden dim per node
        self.linear_in = nn.Linear(input_seq_length, hidden_dim)

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, input_seq_length)  # Back to input_seq_length

    def forward(self, x):
        # x: (B, N, T)
        B, N, T = x.shape

        # Project input from (B, N, T) -> (B*N, T)
        x = x.reshape(B * N, T)
        x = self.linear_in(x)  # (B*N, hidden_dim)
        x = x.reshape(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        # Graph convolution: A @ x
        A = self.adj.to(x.device)  # (N, N)
        support = torch.einsum("ij,bjf->bif", A, x)  # (B, N, hidden_dim)

        # Output projection: (B, N, hidden_dim) -> (B, N, T)
        out = self.output_layer(support)  # (B, N, T)
        return out


# ========== Temporal Module ==========
import torch.nn as nn


class DilatedTCN(nn.Module):
    """
    Dilated Temporal Convolution for sequence modeling.
    Input: (B, N, H)
    Output: (B, N, H)
    """
    def __init__(self, horizon, hidden_channels=64, num_layers=3):
        super().__init__()
        self.horizon = horizon

        class TCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, 1),
                    dilation=(dilation, 1),
                    padding=(dilation * (kernel_size - 1), 0)
                )
                self.relu = nn.ReLU()

            def forward(self, x):
                out = self.conv(x)
                return self.relu(out)[..., :x.shape[2], :]

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = 1 if i == 0 else hidden_channels
            layers.append(TCNBlock(in_ch, hidden_channels, dilation=dilation))
        self.network = nn.Sequential(*layers)
        self.projector = nn.Linear(hidden_channels, horizon)

    def forward(self, x):
        # x: (B, N, H)
        B, N, H = x.shape
        x = x.unsqueeze(1)  # (B, 1, N, H)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, 1, H, N)
        x = self.network(x)  # (B, C, H, N)
        x = x.permute(0, 3, 2, 1)  # (B, N, H, C)
        x = x[:, :, -1, :]  # (B, N, C)
        return self.projector(x)  # (B, N, H)

# LLM generated description:
# ### Model Explanation
# 
# The proposed model is a hybrid architecture that combines the strengths of a Graph Convolutional Network (GCN) for spatial modeling and a Dilated Temporal Convolutional Network (DilatedTCN) for temporal modeling. The model is designed to process spatio-temporal data, such as traffic flow or sensor data, where both spatial relationships and temporal dynamics are crucial.
# 
# #### Encoder Module: GCN (Graph Convolutional Network)
# 
# The GCN encoder is responsible for capturing the spatial dependencies among nodes (e.g., sensors or traffic intersections). Here's a breakdown of its components and functionality:
# 
# - **Input**: The input to the GCN is a tensor of shape `(B, N, T)`, where `B` is the batch size, `N` is the number of nodes, and `T` is the input sequence length.
# - **Linear Transformation**: The input sequence is first projected from `(B, N, T)` to `(B, N, hidden_dim)` using a linear transformation.
# - **Graph Convolution**: The graph convolution operation is performed by multiplying the adjacency matrix `A` with the transformed input. This operation captures the influence of each node on its neighbors.
# - **Output Projection**: The output of the graph convolution

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
       Insert spatial module(s) here.
       You may stack multiple spatial layers based on `spatial_layers`.
"""

"""
       Insert temporal module(s) here.
       You may stack multiple temporal layers based on `temporal_layers`.
"""

class STModel(nn.Module):
    def __init__(self, num_nodes, input_seq_length, horizon, spatial_layers, temporal_layers, input_dim, output_dim, fusion_type, adj_idx, hidden_dim=64):
        """
        Spatiotemporal Model Template

        !!! Do NOT change the constructor parameters. They are required for consistent integration.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.input_seq_length = input_seq_length
        self.horizon = horizon
        self.adj = adj_idx
        self.fusion_type = fusion_type

        # ---------------- Spatial Block ----------------
        # Replace 'GCN' with a specific spatial modeling component
        self.spatial_blocks = nn.ModuleList([
            GCN(num_nodes, input_dim, output_dim,input_seq_length,adj_idx)
            for _ in range(spatial_layers)
        ])

        # ---------------- Bridge Block ----------------
        # Bridge layer: converts (B, N, T) -> (B, N, H)
        self.bridge = nn.Linear(input_seq_length, horizon)

        # ---------------- Temporal Block ----------------
        # Replace 'DilatedTCN' with a specific temporal modeling component
        self.temporal_blocks = nn.ModuleList([
            DilatedTCN(horizon)
            for _ in range(temporal_layers)
        ])

        # ---------------- Fusion Block ----------------
        self.fusion = nn.Linear(horizon * 2, horizon)  # e.g., concat fusion


    def forward(self, x):
        # x: Tensor (B, N, T)

        # ----- Spatial Processing -----
        for spatial in self.spatial_blocks:
            x = F.relu(spatial(x))  # (B, N, T)

        # ----- Bridge to temporal -----
        spatial_out = self.bridge(x)  # (B, N, H)

        # ----- Temporal Processing -----
        temporal_out = spatial_out
        for temporal in self.temporal_blocks:
            temporal_out = F.relu(temporal(temporal_out))  # (B, N, H)

        # ----- Fusion -----
        if self.fusion_type == 'concat':
            fusion_input = torch.cat([spatial_out, temporal_out], dim=-1)  # (B, N, 2H)
            fused = self.fusion(fusion_input)
        else:
            fused = spatial_out + temporal_out  # (B, N, H)

        return fused
