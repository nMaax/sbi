import torch

from sbi.neural_nets.estimators.score_estimator import ( # type: ignore
    MaskedVEScoreEstimator,
)
from sbi.neural_nets.net_builders.vector_field_nets import (  # type: ignore
    MaskedDiTBlock,
    MaskedTimeAdditiveBlock,
    SimformerNet,
)


def _test_masked_simformer_block():
    print("\n--- Testing SimformerNet Block ---")

    # Define dummy parameters for SimformerNet Block initialization
    dim_hidden_block = (
        128  # Output dimension of the block (same as input for stacked blocks)
    )
    dim_t = 16  # Dimension of the time embedding
    num_heads = 8  # Number of attention heads

    # Create SimformerNet Block instance
    masked_simformer_block = MaskedTimeAdditiveBlock(
        dim_hidden_block,
        dim_t,
        num_heads,
    )
    print(f"MaskedTimeAdditiveBlock initialized: {masked_simformer_block}")

    # Define dummy input tensor shapes and values
    batch_size = 4
    sequence_length = 15  # T from SimformerNet (m_theta + n_x)

    # tokens (after init projection): [B, T, dim_hidden_block]
    tokens = torch.randn(batch_size, sequence_length, dim_hidden_block)
    # t_h: [B, dim_t] (time embedding)
    t_h = torch.randn(batch_size, dim_t)

    edge_mask = None  # Not used in this dummy DiTBlock

    print(f"Input shapes: tokens={tokens.shape}, t_h={t_h.shape}")

    # Move to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        masked_simformer_block.to(device)
        tokens = tokens.to(device)
        t_h = t_h.to(device)
        if edge_mask:
            edge_mask = edge_mask.to(device)
        print(f"Moved tensors and MaskedTimeAdditiveBlock to {device}")
    else:
        device = torch.device("cpu")
        print("Running on CPU (CUDA not available)")

    # Perform forward pass
    output = masked_simformer_block(tokens, t_h, edge_mask)
    print(f"MaskedTimeAdditiveBlock forward pass successful. Output shape: {output.shape}")

    # Assert the output shape
    # Expected output shape: [B, T, dim_hidden_block]
    expected_output_shape = (batch_size, sequence_length, dim_hidden_block)
    assert output.shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape}, but got {output.shape}"
    )
    print("MaskedTimeAdditiveBlock output shape assertion passed!")

    return True


def _test_masked_dit_block():
    print("\n--- Testing DiTBlock ---")

    # Define dummy parameters for DiTBlock initialization
    dim_hidden_block = (
        128  # Output dimension of the block (same as input for stacked blocks)
    )
    dim_t = 16  # Dimension of the time embedding
    num_heads = 8  # Number of attention heads

    # Create DiTBlock instance
    masked_dit_block = MaskedDiTBlock(
        dim_hidden_block,
        dim_t,
        num_heads,
    )
    print(f"MaskedDiTBlock initialized: {masked_dit_block}")

    # Define dummy input tensor shapes and values
    batch_size = 4
    sequence_length = 15  # T from SimformerNet (m_theta + n_x)

    # tokens (after init projection): [B, T, dim_hidden_block]
    tokens = torch.randn(batch_size, sequence_length, dim_hidden_block)
    # t_h: [B, dim_t] (time embedding)
    t_h = torch.randn(batch_size, dim_t)

    edge_mask = None  # Not used in this dummy DiTBlock

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
        print("Running on CPU (CUDA not available)")

    # Perform forward pass
    output = masked_dit_block(tokens, t_h, edge_mask)
    print(f"MaskedDiTBlock forward pass successful. Output shape: {output.shape}")

    # Assert the output shape
    # Expected output shape: [B, T, dim_hidden_block]
    expected_output_shape = (batch_size, sequence_length, dim_hidden_block)
    assert output.shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape}, but got {output.shape}"
    )
    print("MaskedDiTBlock output shape assertion passed!")

    return True


def _test_simformer():
    print("\n--- Testing SimformerNet ---")

    # Define dummy parameters for SimformerNet initialization
    in_features = 5  # Dimension of input features for theta and x
    num_nodes = 20  # Total possible nodes (m + n should be less than or equal to this)

    # Create SimformerNet instance
    simformer = SimformerNet(
        in_features=in_features,
        num_nodes=num_nodes,
    )
    print(f"SimformerNet initialized: {simformer}")

    # Define dummy input tensor shapes and values
    batch_size = 4
    m_theta = 10  # Number of nodes/elements in theta
    n_x = 5  # Number of nodes/elements in x
    total_T = m_theta + n_x  # Total sequence length

    # theta: [B, m, F_theta]
    theta = torch.randn(batch_size, m_theta, in_features)
    # x: [B, n, F_x]
    x = torch.randn(batch_size, n_x, in_features)
    # inputs: [theta, x] on second dimension
    inputs = torch.cat([theta, x], dim=1)
    # t: [B] (time value for each batch item)
    t = torch.rand(batch_size)  # Random time between 0 and 1
    # Not used in dummy DiTBlock, but needed for init
    edge_mask = torch.ones(total_T, total_T)
    # condition_mask: [B, T] (boolean mask)
    # Example: First 'm_theta' elements are always conditioned, rest are random
    condition_mask = torch.cat(
        [
            torch.ones(batch_size, m_theta, dtype=torch.bool),
            torch.randint(0, 2, (batch_size, n_x), dtype=torch.bool),
        ],
        dim=1,
    )

    print(
        f"Input shapes: theta={theta.shape}, x={x.shape}, t={t.shape}, condition_mask={condition_mask.shape}"
    )

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
        print("Running on CPU (CUDA not available)")

    # Perform forward pass
    output = simformer(inputs, t, condition_mask, edge_mask)
    print(f"SimformerNet forward pass successful. Output shape: {output.shape}")

    # Assert the output shape
    # Expected output shape: [B, T, F]
    expected_output_shape = (batch_size, total_T, in_features)
    assert output.shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape}, but got {output.shape}"
    )

    print("SimformerNet output shape assertion passed!")

    return True


def _test_masked_ve_score_estimator():
    print("\n--- Testing MaskedVEScoreEstimator ---")

    # Dummy parameters
    batch_size = 2
    num_nodes = 8
    in_features = 4
    input_shape = torch.Size([num_nodes, in_features])

    # Create a dummy MaskedVectorFieldNet (SimformerNet)
    net = SimformerNet(
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
    x = torch.randn(batch_size, num_nodes, in_features)
    t = torch.rand(batch_size)

    # Test mean_t_fn
    mean_t = estimator.mean_t_fn(t)
    print(f"mean_t_fn output shape: {mean_t.shape}")

    # Test std_fn
    std = estimator.std_fn(t)
    print(f"std_fn output shape: {std.shape}")

    # Test mean_fn
    mean = estimator.mean_fn(x, t)
    print(f"mean_fn output shape: {mean.shape}")

    # Test drift_fn
    drift = estimator.drift_fn(x, t)
    print(f"drift_fn output shape: {drift.shape}")

    # Test diffusion_fn
    diffusion = estimator.diffusion_fn(x, t)
    print(f"diffusion_fn output shape: {diffusion.shape}")

    # Test approx_marginal_mean
    approx_mean = estimator.approx_marginal_mean(t)
    print(f"approx_marginal_mean output shape: {approx_mean.shape}")

    # Test approx_marginal_std
    approx_std = estimator.approx_marginal_std(t)
    print(f"approx_marginal_std output shape: {approx_std.shape}")

    # Test forward method
    edge_mask = torch.ones(num_nodes, num_nodes)
    condition_mask = torch.bernoulli(torch.full((batch_size, num_nodes), 0.33))
    output = estimator(x, t, condition_mask, edge_mask)
    print(f"Estimator forward output shape: {output.shape}")

    # Test loss method
    loss = estimator.loss(x)
    print(f"Estimator loss output shape: {loss.shape}")

    # Check shapes
    broadcasted_t_shape = torch.Size([batch_size, 1, 1])
    assert mean_t.shape == broadcasted_t_shape, "mean_t output size mismatch"
    assert mean.shape == x.shape, "mean_fn output size mismatch"
    assert std.shape == broadcasted_t_shape, "std_fn output size mismatch"
    assert drift.shape == x.shape, "drift output shape mismatch"
    assert diffusion.shape == broadcasted_t_shape, (
        "diffusion_fn output batch size mismatch"
    )
    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}"
    )
    assert approx_mean.shape == broadcasted_t_shape, (
        "approx_marginal_mean output batch size mismatch"
    )
    assert approx_std.shape == broadcasted_t_shape, (
        "approx_marginal_std output batch size mismatch"
    )
    print("MaskedVEScoreEstimator tests passed!")

    return True


# Run the test when this file is executed directly
if __name__ == "__main__":
    _test_masked_simformer_block()
    _test_masked_dit_block()
    _test_simformer()
    _test_masked_ve_score_estimator()
