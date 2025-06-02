from typing import Callable, Optional
import torch
from torch import Tensor, nn

class DiTBlock(nn.Module):
    def __init__(self, edge_mask, in_features, dim_hidden):
        super().__init__()
        self.edge_mask = edge_mask
        self.proj = nn.Linear(in_features, dim_hidden)

    def forward(self, tokens, t):
        return self.proj(tokens)

class Transformer(nn.Module):
    def __init__(self, edge_mask):
        super().__init__()

    def forward(self, tokens, t):
        pass

class Simformer(nn.Module):
    def __init__(
            self,
            in_features,
            num_nodes,
            edge_mask,
            dim_val=64,
            dim_id=32,
            dim_cond=16,
            dim_t=16,
            dim_hidden=128,
            num_blocks=4,
            num_heads=8
        ):
        super().__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.edge_mask = edge_mask
        self.dim_val = dim_val
        self.dim_id = dim_id
        self.dim_cond = dim_cond
        self.dim_t = dim_t
        self.dim_hidden = dim_hidden
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        # Tokenize on val
        #? Should this be a repeat rather than Linear?
        self.val_linear = nn.Linear(in_features, dim_val)

        # Tokenize on id
        self.id_embedding = nn.Embedding(num_nodes, dim_id)

        # Conditioning parameter
        self.conditioning_parameter = nn.Parameter(torch.randn(1, 1, dim_cond) * 0.5)

        # Time embedding
        self.time_embedding = RandomFourierTimeEmbedding(dim_t)

        # Project input tokens to hidden dim
        self.in_proj = nn.Linear(dim_val + dim_id + dim_cond, dim_hidden)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(edge_mask, dim_hidden, dim_hidden) for _ in range(num_blocks)
        ])


        # Output projection
        self.out_linear = nn.Linear(dim_hidden, in_features)


    def forward(self, theta, x, t, condition_mask):

        # Checks for shapes and device
        B_theta, m, F_theta = theta.shape
        B_x, n, F_x = x.shape
        assert B_theta == B_x, f"theta and x must have the same batch size: {B_theta=} != {B_x=}"
        assert F_theta == F_x == self.in_features, f"theta and x must have the same feature dimension as the one declared in init{self.in_features}: {F_theta=}, {F_x=}"

        B = B_theta
        T = m + n
        F = self.in_features

        assert condition_mask.shape == (B, T), "condition_mask must have the same batch size and sequence length as theta"

        theta_device = theta.device
        x_device = x.device
        assert theta_device == x_device, "theta and x must be on the same device"

        device = theta_device

        # Tokenize on val
        #? Should this rather be cat > linear?
        h = torch.cat([theta, x], dim=1)  # [B, T, F]
        val_h = self.val_linear(h)  # [B, T, dim_val]

        # Tokenize on id
        ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
        id_h = self.id_embedding(ids)  # [B, T, dim_id]

        # Conditioning
        # conditioning_parameter: [1, 1, dim_cond]
        # condition_mask: [B, T]
        conditioning_h = self.conditioning_parameter.expand(B, T, self.dim_cond) * condition_mask.unsqueeze(-1)  # [B, T, dim_cond]

        # Time embedding
        #? Normalize time
        t_h = self.time_embedding(t)  # [B, dim_t]

        # Concatenate tokens
        tokens = torch.cat([val_h, id_h, conditioning_h], dim=-1)  # [B, T, dim_val+dim_id+dim_cond]
        h = self.in_proj(tokens)  # [B, T, dim_hidden]

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h, t_h)

        # Output projection
        #? Should this be flattened as [B, T*F]?
        out = self.out_linear(h)  # [B, T, F]
        return out
