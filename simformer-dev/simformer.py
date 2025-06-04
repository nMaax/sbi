from typing import Callable, Optional
import torch
from torch import Tensor, nn

from sbi.utils.vector_field_utils import MaskedVectorFieldNet
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator

from sbi.neural_nets.net_builders.vector_field_nets import RandomFourierTimeEmbedding
from sbi.neural_nets.net_builders.vector_field_nets import DiTBlock

class MaskedConditionalScoreEstimator(ConditionalScoreEstimator):
    #! Extend from Conditional, and override
    # class MyConditionalScoreEstimator(ConditionalScoreEstimator):
    #     def __init__(self, ...):
    #         ...
    #
    #     def forward(self, ..., -cond, +masks):
    #         # 3D Tensors, not 2D, as input to the net
    #
    #     def loss(self, ...):
    #         # 3D Tensors, not 2D, as input to the net

    #! Note:
    #!  NO   --> input = ALL THE NODES, condition is a subset of them (those observed)
    #!  THIS --> input = latent NODEs,  condition is observed;
    #!           i.e., inputs union condition = ALL NODES,
    #!                 but input intersect condition = void

    def forward(
        self,
        input: Tensor,                              # [B, T, F]
        time: Tensor,                               # [B] or [B, 1]
        condition_mask: Optional[Tensor] = None,    # [B, T]
        edge_mask: Optional[Tensor] = None          # [T, T] or [B, T, T]
    ) -> Tensor:
        """
        Forward pass of the score estimator for Simformer.
        Args:
            input: [B, T, F] - all nodes (latent and observed).
            time: [B] or [B, 1] - diffusion times.
            condition_mask: [B, T] - True for observed/conditioned nodes.
            edge_mask: [T, T] or [B, T, T] - attention mask for the transformer.
        Returns:
            Score (gradient of the density) at a given time, shape [B, T, F].
        """
        device = input.device
        B, T, F = input.shape

        # Ensure time shape is [B, 1]
        if time.dim() == 1:
            time = time.view(B, 1)

        # Compute time-dependent mean and std for z-scoring
        mean = self.approx_marginal_mean(time)      # [B, 1, F] or broadcastable
        std = self.approx_marginal_std(time)        # [B, 1, F] or broadcastable

        #! Skipped, as Simformer expects time as it is, not as a level of std
        # time_enc = self.std_fn(time)

        # Z-score the input
        input_enc = (input - mean) / std

        # "Skip connection" Gaussian score
        score_gaussian = (input - mean) / (std ** 2)

        # Model prediction (no flattening, pass masks)
        score_pred = self.net(input_enc, time.squeeze(-1), condition_mask, edge_mask)  # [B, T, F]

        # Output pre-conditioned score (same scaling as in reference)
        scale = self.mean_t_fn(time) / self.std_fn(time)
        # Ensure scale is broadcastable to [B, T, F]
        while scale.dim() < input.dim():
            scale = scale.unsqueeze(-1)
        output_score = -scale * score_pred - score_gaussian

        return output_score

    def score(
        self,
        input: Tensor,                            # [B, T, F]
        time: Tensor,                             # [B] or [B, 1]
        condition_mask: Optional[Tensor] = None,  # [B, T]
        edge_mask: Optional[Tensor] = None        # [T, T] or [B, T, T]
    ) -> Tensor:
        return self(input=input, time=time, condition_mask=condition_mask, edge_mask=edge_mask)

    def loss(
        self,
        input: Tensor,
        times: Optional[Tensor] = None, #! Where do I generate it? In the training loop!
        condition_mask: Optional[Tensor] = None, #! Where do I generate it? In the training loop!
        edge_mask: Optional[Tensor] = None, #! Where do I generate it? In the training loop!
        control_variate=True,
        control_variate_threshold=0.3,
    ) -> Tensor:
        """
        Denoising score matching loss for Simformer.
        Args:
            input: [B, T, F] - all nodes (latent and observed).
            times: [B, 1] or [B] - diffusion times. If None, sampled uniformly.
            condition_mask: [B, T] - True for observed/conditioned nodes.
            edge_mask: [T, T] or [B, T, T] - attention mask for the transformer.
            weight_fn: Callable for weighting the loss over time (optional).
            rebalance_loss: Whether to rebalance loss by number of unmasked elements.
        Returns:
            Scalar loss.
        """
        device = input.device
        B, T, F = input.shape

        # Sample times if not provided
        if times is None:
            times = torch.rand(B, 1, device=device) * (self.t_max - self.t_min) + self.t_min  # [B, 1]
        times = times.view(B, 1) # Ensure shape [B, 1]

        # Sample noise
        eps = torch.randn_like(input)  # [B, T, F]

        # Compute mean and std for the SDE
        mean_t = self.mean_fn(input, times)  # [B, T, F]
        std_t = self.std_fn(times)           # [B, 1, 1] or [B, 1] or [B]
        while std_t.dim() < input.dim():
            std_t = std_t.unsqueeze(-1)
        std_t = std_t.expand_as(input)       # [B, T, F]

        # Get noised input
        input_noised = mean_t + std_t * eps  # [B, T, F]

        # True score
        score_target = -eps / std_t

        # Apply condition mask: where condition_mask is True, use input (observed)
        if condition_mask is not None:
            mask = condition_mask.bool()
            input_noised = torch.where(mask.unsqueeze(-1), input, input_noised)

        # Model prediction
        score_pred = self.forward(input_noised, times.squeeze(-1), condition_mask, edge_mask)  # [B, T, F]

        # Compute MSE loss, mask out observed entries
        loss = torch.sum((score_pred - score_target) ** 2.0, dim=-1)  # [B, T] (sum tensor of shape [B, T, F] over F)
        if condition_mask is not None:
            loss = torch.where(mask, torch.zeros_like(loss), loss)

        # Weight and sum over T and F
        weights = self.weight_fn(times)


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

    #! The original DiTBlock do not use mask
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
            #! nn.MultiheadAttention expects a boolean mask with True for masked positions
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

    #! NOTE: Both VectorFieldNet and ConditionalScoreEstimator were designed expecting a posterior sampling only
    #! Practically confounding theta == latent, x == conditioning always!
    #? Maybe you should make a more general class, from which they both extend? Or rather a separate, parallel class
    #! NOTE: VectorFieldNet forward() method does not expect masks
    #? Can I still pass it?
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

def _test_dit_block():
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



# Run the test when this file is executed directly
if __name__ == "__main__":
    _test_time_embedding()
    _test_dit_block()
    _test_simformer()
