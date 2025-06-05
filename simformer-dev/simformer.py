from typing import Callable, Optional, Union
import math
from typing import Callable, Union

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.score_estimator import MaskedConditionalScoreEstimator
from sbi.utils.vector_field_utils import MaskedVectorFieldNet
from sbi.neural_nets.net_builders.vector_field_nets import DiTBlock
from sbi.neural_nets.net_builders.vector_field_nets import RandomFourierTimeEmbedding


class MaskedVEScoreEstimator(MaskedConditionalScoreEstimator):
    def __init__(
        self,
        net: Union[MaskedVectorFieldNet, nn.Module],
        input_shape: torch.Size,
        embedding_net: nn.Module = nn.Identity(),
        weight_fn: Union[str, Callable] = "max_likelihood",
        sigma_min: float = 1e-4,
        sigma_max: float = 10.0,
        mean_0: float = 0.0,
        std_0: float = 1.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(
            net,
            input_shape,
            embedding_net=embedding_net,
            weight_fn=weight_fn,
            mean_0=mean_0,
            std_0=std_0,
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        """Conditional mean function for variance exploding SDEs, which is always 1.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Conditional mean at a given time.
        """
        # Handle case when times has 3 dimensions during sampling
        original_shape = times.shape
        has_sample_dim = len(original_shape) == 3

        #! Is this what I need to do in my setting?
        if has_sample_dim:
            # Create ones tensor
            phi = torch.ones_like(times.reshape(-1), device=times.device)
            # Add necessary dimensions
            for _ in range(len(self.input_shape)):
                phi = phi.unsqueeze(-1)
            # Reshape back to original
            phi = phi.reshape(*original_shape[:-1], *phi.shape[1:])
            return phi
        else:
            phi = torch.ones_like(times, device=times.device)
            for _ in range(len(self.input_shape)):
                phi = phi.unsqueeze(-1)
            return phi

    def std_fn(self, times: Tensor) -> Tensor:
        """Standard deviation function for variance exploding SDEs.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """
        # Handle case when times has 3 dimensions during sampling
        original_shape = times.shape
        has_sample_dim = len(original_shape) == 3

        #! Is this what I need to do in my setting?
        if has_sample_dim:
            # Flatten for computation
            times_flat = times.reshape(-1)
            std = self.sigma_min * (self.sigma_max / self.sigma_min) ** times_flat
            # Add necessary dimensions
            for _ in range(len(self.input_shape)):
                std = std.unsqueeze(-1)
            # Reshape back to original
            std = std.reshape(*original_shape[:-1], *std.shape[1:])
            return std
        else:
            std = self.sigma_min * (self.sigma_max / self.sigma_min) ** times
            for _ in range(len(self.input_shape)):
                std = std.unsqueeze(-1)
            return std

    def _sigma_schedule(self, times: Tensor) -> Tensor:
        """Geometric sigma schedule for variance exploding SDEs.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Sigma schedule at a given time.
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** times

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for variance exploding SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        return torch.tensor([0.0])

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for variance exploding SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        g = self._sigma_schedule(times) * math.sqrt(
            (2 * math.log(self.sigma_max / self.sigma_min))
        )

        #! Is this what I need to do in my setting?
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)

        return g

class MaskedDiTBlock(DiTBlock):
    def __init__(
        self,
        hidden_dim,
        cond_dim,
        num_heads,
        mlp_ratio=2,
        activation=nn.GELU
    ):
        super().__init__(hidden_dim, cond_dim, num_heads, mlp_ratio, activation)

    def forward(self, x, cond, mask):

        ada_params = self.ada_affine(cond)
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = ada_params.chunk(6, dim=-1)
        B, T, D = x.shape

        attn_scale = attn_scale.view(B, 1, D)
        attn_shift = attn_shift.view(B, 1, D)
        attn_gate = attn_gate.view(B, 1, D)
        mlp_scale = mlp_scale.view(B, 1, D)
        mlp_shift = mlp_shift.view(B, 1, D)
        mlp_gate = mlp_gate.view(B, 1, D)

        # Adaptive LayerNorm before attention
        x_norm = self.norm1(x)
        x_norm = x_norm * (attn_scale + 1) + attn_shift

        # Prepare attention mask
        if mask is not None:
            # Ensure the mask is boolean: True for masked, False for allowed
            mask = mask.bool()
            #? nn.MultiheadAttention expects [B*num_heads, T, T] or [T, T], so flatten batch if needed?

        # Self-attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_gate * attn_out

        # Adaptive LayerNorm before MLP
        x_norm = self.norm2(x)
        x_norm = x_norm * (mlp_scale + 1) + mlp_shift

        # MLP
        mlp_out = self.mlp(x_norm)
        x = x + mlp_gate * mlp_out

        return x

class Simformer(MaskedVectorFieldNet):
    def __init__(
            self,
            in_features,
            num_nodes,
            dim_val=64,
            dim_id=32,
            dim_cond=16,
            dim_t=16,
            dim_hidden=128,
            num_blocks=4,
            num_heads=8
        ):
        super().__init__()
        self.in_features = in_features # Number of features by each node (F)
        self.num_nodes = num_nodes # Number of nodes in the DAG (T = m + n)
        self.dim_val = dim_val # Dimension of the value token
        self.dim_id = dim_id # Dimension of the id token
        self.dim_cond = dim_cond # Dimension of the conditioning token
        self.dim_t = dim_t # Dimension of the time embedding
        self.dim_hidden = dim_hidden # Dimension of the latent space in the transformer blocks
        self.num_blocks = num_blocks # Number of transformer blocks to stack
        self.num_heads = num_heads # Number of attention heads per each transfrormer block

        assert in_features > 0, "in_features must be greater than 0"
        assert num_nodes > 0, "num_nodes must be greater than 0"
        assert dim_val > 0, "dim_val must be greater than 0"
        assert dim_id > 0, "dim_id must be greater than 0"
        assert dim_cond > 0, "dim_cond must be greater than 0"
        assert dim_t > 0, "dim_t must be greater than 0"
        assert dim_hidden > 0, "dim_hidden must be greater than 0"
        assert num_blocks > 0, "num_blocks must be greater than 0"
        assert num_heads > 0, "num_heads must be greater than 0"

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
            MaskedDiTBlock(dim_hidden, dim_t, num_heads) for _ in range(num_blocks)
        ])

        # Output projection
        self.out_linear = nn.Linear(dim_hidden, in_features)

    def forward(self, inputs, t, condition_mask, edge_mask):

        device = inputs.device
        B, T, F = inputs.shape

        assert condition_mask.shape == (B, T), "condition_mask must have the same batch size and sequence length as inputs"
        assert edge_mask.shape == (T, T), "edge_mask must have same shape as the sequence length"

        # Tokenize on val
        #? Should this rather be cat[linear(theta), linear(x)]?
        val_h = self.val_linear(inputs) # [B, T, dim_val]

        # Tokenize the nodes' id
        ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
        id_h = self.id_embedding(ids)  # [B, T, dim_id]

        # Conditioning
        # conditioning_parameter: [1, 1, dim_cond]
        # condition_mask: [B, T]
        conditioning_h = self.conditioning_parameter.expand(B, T, self.dim_cond) * condition_mask.unsqueeze(-1)  # [B, T, dim_cond]

        # Time embedding
        #? Should I normalize time?
        t_h = self.time_embedding(t)  # [B, dim_t]

        # Concatenate tokens
        tokens = torch.cat([val_h, id_h, conditioning_h], dim=-1)  # [B, T, dim_val+dim_id+dim_cond]
        h = self.in_proj(tokens)  # [B, T, dim_hidden]

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h, t_h, edge_mask)

        # Output projection
        #? Should this be flattened as [B, T*F]?
        #! Answer: No, as you will use your own score estimator
        out = self.out_linear(h)  # [B, T, F]
        return out

def _test_time_embedding():
    print("\n--- Testing RandomFourierTimeEmbedding ---")

    # Create a dummy time tensor
    times = torch.linspace(0, 1, steps=4)
    # Instantiate the embedding
    emb = RandomFourierTimeEmbedding(embed_dim=8)
    # Forward pass
    out = emb(times)
    assert out.shape == (4, 8), f"Expected output shape (4, 8), got {out.shape}"
    print("RandomFourierTimeEmbedding test passed!")

def _test_masked_dit_block():
    print("\n--- Testing DiTBlock ---")

    # Define dummy parameters for DiTBlock initialization
    dim_hidden_block = 128  # Output dimension of the block (same as input for stacked blocks)
    dim_t = 16 # Dimension of the time embedding
    num_heads = 8 # Number of attention heads

    # Create DiTBlock instance
    masked_dit_block = MaskedDiTBlock(
        dim_hidden_block,
        dim_t,
        num_heads,
    )
    print(f"MaskedDiTBlock initialized: {masked_dit_block}")

    # Define dummy input tensor shapes and values
    batch_size = 4
    sequence_length = 15 # T from Simformer (m_theta + n_x)

    # tokens (after init projection): [B, T, dim_hidden_block]
    tokens = torch.randn(batch_size, sequence_length, dim_hidden_block)
    # t_h: [B, dim_t] (time embedding)
    t_h = torch.randn(batch_size, dim_t)

    edge_mask = None # Not used in this dummy DiTBlock

    print(f"Input shapes: tokens={tokens.shape}, t_h={t_h.shape}")

    # Move to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        masked_dit_block.to(device)
        tokens = tokens.to(device)
        t_h = t_h.to(device)
        if edge_mask:
            edge_mask = edge_mask.to(device)
        print(f"Moved tensors and MaskedDiTBlock to {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on CPU (CUDA not available)")

    # Perform forward pass
    output = masked_dit_block(tokens, t_h, edge_mask)
    print(f"MaskedDiTBlock forward pass successful. Output shape: {output.shape}")

    # Assert the output shape
    # Expected output shape: [B, T, dim_hidden_block]
    expected_output_shape = (batch_size, sequence_length, dim_hidden_block)
    assert output.shape == expected_output_shape, \
        f"Expected output shape {expected_output_shape}, but got {output.shape}"
    print("MaskedDiTBlock output shape assertion passed!")

def _test_simformer():
    print("\n--- Testing Simformer ---")

    # Define dummy parameters for Simformer initialization
    in_features = 5      # Dimension of input features for theta and x
    num_nodes = 20       # Total possible nodes (m + n should be less than or equal to this)

    # Create Simformer instance
    simformer = Simformer(
        in_features=in_features,
        num_nodes=num_nodes,
    )
    print(f"Simformer initialized: {simformer}")

    # Define dummy input tensor shapes and values
    batch_size = 4
    m_theta = 10            # Number of nodes/elements in theta
    n_x = 5                 # Number of nodes/elements in x
    total_T = m_theta + n_x # Total sequence length

    # theta: [B, m, F_theta]
    theta = torch.randn(batch_size, m_theta, in_features)
    # x: [B, n, F_x]
    x = torch.randn(batch_size, n_x, in_features)
    # inputs: [theta, x] on second dimension
    inputs = torch.cat([theta, x], dim=1)
    # t: [B] (time value for each batch item)
    t = torch.rand(batch_size) # Random time between 0 and 1
    # Not used in dummy DiTBlock, but needed for init
    edge_mask = torch.ones(total_T, total_T)
    # condition_mask: [B, T] (boolean mask)
    # Example: First 'm_theta' elements are always conditioned, rest are random
    condition_mask = torch.cat([
        torch.ones(batch_size, m_theta, dtype=torch.bool),
        torch.randint(0, 2, (batch_size, n_x), dtype=torch.bool)
    ], dim=1)

    print(f"Input shapes: theta={theta.shape}, x={x.shape}, t={t.shape}, condition_mask={condition_mask.shape}")

    # Move to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        simformer.to(device)
        inputs = inputs.to(device)
        t = t.to(device)
        condition_mask = condition_mask.to(device)
        edge_mask = edge_mask.to(device)
        print(f"Moved tensors and model to {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on CPU (CUDA not available)")


    # Perform forward pass

    output = simformer(inputs, t, condition_mask, edge_mask)
    print(f"Simformer forward pass successful. Output shape: {output.shape}")

    # Assert the output shape
    # Expected output shape: [B, T, F]
    expected_output_shape = (batch_size, total_T, in_features)
    assert output.shape == expected_output_shape, \
        f"Expected output shape {expected_output_shape}, but got {output.shape}"

    print("Simformer output shape assertion passed!")

def _test_masked_ve_score_estimator():
    print("\n--- Testing MaskedVEScoreEstimator ---")

    # Dummy parameters
    in_features = 4
    num_nodes = 8
    input_shape = torch.Size([num_nodes, in_features])

    # Create a dummy MaskedVectorFieldNet (Simformer)
    net = Simformer(
        in_features=in_features,
        num_nodes=num_nodes,
    )

    # Instantiate MaskedVEScoreEstimator
    estimator = MaskedVEScoreEstimator(
        net=net,
        input_shape=input_shape,
        sigma_min=1e-3,
        sigma_max=1.0,
    )
    print(f"MaskedVEScoreEstimator initialized: {estimator}")

    # Dummy input
    batch_size = 2
    x = torch.randn(batch_size, num_nodes, in_features)
    t = torch.rand(batch_size)

    # Test mean_t_fn
    mean = estimator.mean_t_fn(t)
    print(f"mean_t_fn output shape: {mean.shape}")

    # Test std_fn
    std = estimator.std_fn(t)
    print(f"std_fn output shape: {std.shape}")

    # Test drift_fn
    drift = estimator.drift_fn(x, t)
    print(f"drift_fn output: {drift}")

    # Test diffusion_fn
    diffusion = estimator.diffusion_fn(x, t)
    print(f"diffusion_fn output shape: {diffusion.shape}")

    # Test forward method
    output = estimator(x, t)
    print(f"Estimator forward output shape: {output.shape}")

    # Check shapes
    assert mean.shape[0] == t.shape[0], "mean_t_fn output batch size mismatch"
    assert std.shape[0] == t.shape[0], "std_fn output batch size mismatch"
    assert diffusion.shape[0] == t.shape[0], "diffusion_fn output batch size mismatch"
    assert output.shape == x.shape, f"Expected output shape {x.shape}, got {output.shape}"
    print("MaskedVEScoreEstimator tests passed!")

# Run the test when this file is executed directly
if __name__ == "__main__":
    _test_time_embedding()
    _test_masked_dit_block()
    _test_simformer()
    _test_masked_ve_score_estimator()
