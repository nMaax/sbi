from typing import Callable, Optional
import torch
from torch import Tensor, nn

class DiTBlock(nn.Module):
    def __init__(self, edge_mask):
        super().__init__()

    def forward(self, tokens, t):
        pass

class Transformer(nn.Module):
    def __init__(self, edge_mask):
        super().__init__()

    def forward(self, tokens, t):
        pass

class Simformer(VectorFieldNet):
    def __init__(self, edge_mask):
        super().__init__()

    def forward(self, theta, x, t, condition_mask):

        # Input: theta -> [B, m, F], x -> [B, n, F], t -> [B,],
        # condition_mask: representing L, C (Latent and Conditional) variables -> [B, T],
        # edge_mask: representing DAG of nodes -> [B, T, T]

        # Tokenize on val -> val(theta), val(x)
        # theta: Tensor, shape [B, m, F]
        # x: Tensor, shape [B, n, F]
        # val: Linear layer(m+n, dim_val)
        # val_h = linear([theta, x])

        # Tokenize on id -> id(theta), id(x)
        # theta: Tensor, shape [B, m, F]
        # x: Tensor, shape [B, n, F]
        # id: Embedding layer(m+n, dim_id)
        # id_h = embedding([theta, x])

        # condition_mask: Tensor, shape [B, T]
        # Produce a signal for conditioning
        # conditioning = nn.Parameter(torch.randn(1, 1, dim_cond) * 0.5)
        # Mask conditioning based on the condition_mask
        # conditioning_h = conditioning * condition_mask, shape [B, T, dim_cond]

        # t: Tensor, shape [B,]
        # time -> Fourier embedding
        # t_h = embedding(t) [B, dim_t]

        # Concatenate all tokens *on the feature dimension*
        # tokens = [val_h, id_h, conditioning_h], shape [B, T, dim_tokens=dim_val+dim_id+dim_cond]

        # Transformer: Sequence of DiT Blocks with self attention, masked on dependencies mask
        # Block(tokens, t_h, edge_mask) -> h2
            # h1 = norm(tokens) -> [B, T, dim_tokens]
            # h1 = att_block(tokens, edge_mask) -> [B, T, dim_hidden]
                # att_block: multi-head attention(x, x, x, edge_mask)
            # h1 = residual(h1, tokens)

            # h2 = norm(h1) -> [B, T, dim_hidden]
            # h2 = dense_block(h2, t_h) -> [B, T, dim_hidden]
                # Time Linear layer(dim_t, dim_hidden)
                # dense_block: MLP(h2) + Linear(t_h) -> [B, T, dim_hidden]
            # h2 = residual(h2, h1) -> [B, T, dim_hidden]

            # (Nx times)..., where tokens = h2

        # Produce the vector field, $\nabla_{x_t} log p(x_t | theta, t)$
        # So, output should be of same shape as input, i.e., [B, T, F]
        # Linear layer(dim_tokens, F)
        # return linear(h2)
        pass
