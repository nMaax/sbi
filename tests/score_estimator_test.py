# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi.neural_nets.net_builders import (
    build_masked_score_matching_estimator,
    build_score_matching_estimator,
)


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("input_sample_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("score_net", ["mlp"])
def test_score_estimator_loss_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    score_net,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    score_estimator, inputs, conditions = _build_score_estimator_and_tensors(
        sde_type,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        net=score_net,
    )

    losses = score_estimator.loss(inputs[0], condition=conditions)
    assert losses.shape == (batch_dim,)


@pytest.mark.gpu
@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("score_net", ["mlp"])
def test_score_estimator_on_device(sde_type, device, score_net):
    """Test whether DensityEstimators can be moved to the device."""
    score_estimator = build_score_matching_estimator(
        torch.randn(100, 1),
        torch.randn(100, 1),
        sde_type=sde_type,
        net=score_net,
    )
    score_estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 1, device=device)
    condition = torch.randn(100, 1, device=device)
    time = torch.randn(1, device=device)
    out = score_estimator(inputs, condition, time)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."

    # Test loss
    loss = score_estimator.loss(inputs, condition)
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.parametrize("sde_type", ["vp", "ve", "subvp"])
@pytest.mark.parametrize("input_sample_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("score_net", ["mlp"])
def test_score_estimator_forward_shapes(
    sde_type,
    input_sample_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
    score_net,
):
    """Test whether `forward` of DensityEstimators follow the shape convention."""
    score_estimator, inputs, conditions = _build_score_estimator_and_tensors(
        sde_type,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_sample_dim,
        net=score_net,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = score_estimator(inputs[0], condition=conditions, time=times)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = score_estimator(inputs[0], condition=conditions, time=time)
    assert outputs.shape == (batch_dim, *input_event_shape), "Output shape mismatch."


def _build_score_estimator_and_tensors(
    sde_type: str,
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    input_sample_dim: int = 1,
    **kwargs,
):
    """Helper function for all tests that deal with shapes of density estimators."""

    # Use discrete thetas such that categorical density esitmators can also use them.
    building_thetas = torch.randint(
        0, 4, (1000, *input_event_shape), dtype=torch.float32
    )
    building_xs = torch.randn((1000, *condition_event_shape))

    if len(condition_event_shape) > 1:
        embedding_net = CNNEmbedding(condition_event_shape, kernel_size=1)
    else:
        embedding_net = torch.nn.Identity()

    score_estimator = build_score_matching_estimator(
        torch.randn_like(building_thetas),
        torch.randn_like(building_xs),
        sde_type=sde_type,
        embedding_net=embedding_net,
        **kwargs,
    )

    inputs = building_thetas[:batch_dim]
    condition = building_xs[:batch_dim]

    inputs = inputs.unsqueeze(0)
    inputs = inputs.expand(
        [
            input_sample_dim,
        ]
        + [-1] * (1 + len(input_event_shape))
    )
    condition = condition
    return score_estimator, inputs, condition


@pytest.mark.parametrize("sde_type", ["ve"])
@pytest.mark.parametrize("input_event_shape", ((5, 1), (5, 4)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("score_net", ["simformer"])
def test_masked_score_estimator_loss_shapes(
    sde_type,
    input_event_shape,
    batch_dim,
    score_net,
):
    """Test whether `loss` of MaskedScoreEstimator follows the shape convention."""
    (
        score_estimator,
        inputs,
        condition_masks,
        edge_masks,
    ) = _build_masked_score_estimator_and_tensors(
        sde_type,
        input_event_shape,
        batch_dim,
        net=score_net,
    )

    losses = score_estimator.loss(
        inputs, condition_mask=condition_masks, edge_mask=edge_masks
    )
    assert losses.shape[0] == batch_dim, "Loss shape mismatch."


@pytest.mark.gpu
@pytest.mark.parametrize("sde_type", ["ve"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("score_net", ["simformer"])
def test_masked_score_estimator_on_device(sde_type, device, score_net):
    """Test whether MaskedScoreEstimator can be moved to the device."""
    score_estimator = build_masked_score_matching_estimator(
        torch.randn(100, 5, 1),
        torch.randn(100, 5, 1),
        sde_type=sde_type,
        net=score_net,
    )
    score_estimator.to(device)

    # Test forward
    inputs = torch.randn(100, 5, 1, device=device)
    condition_masks = torch.ones(100, 5, device=device)
    edge_masks = torch.ones(100, 5, 5, device=device)
    time = torch.randn(1, device=device)
    out = score_estimator(inputs, time, condition_masks, edge_masks)

    assert str(out.device).split(":")[0] == device, "Output device mismatch."

    # Test loss
    loss = score_estimator.loss(inputs, condition_masks, edge_masks)
    assert str(loss.device).split(":")[0] == device, "Loss device mismatch."


@pytest.mark.parametrize("sde_type", ["ve"])
@pytest.mark.parametrize("input_event_shape", ((5, 1), (5, 4)))
@pytest.mark.parametrize("batch_dim", (1, 10))
@pytest.mark.parametrize("score_net", ["simformer"])
def test_masked_score_estimator_forward_shapes(
    sde_type,
    input_event_shape,
    batch_dim,
    score_net,
):
    """Test whether `forward` of MaskedScoreEstimator follows the shape convention."""
    (
        score_estimator,
        inputs,
        condition_masks,
        edge_masks,
    ) = _build_masked_score_estimator_and_tensors(
        sde_type,
        input_event_shape,
        batch_dim,
        net=score_net,
    )
    # Batched times
    times = torch.rand((batch_dim,))
    outputs = score_estimator(
        inputs, time=times, condition_mask=condition_masks, edge_mask=edge_masks
    )
    assert outputs.shape == (
        batch_dim,
        *input_event_shape,
    ), "Output shape mismatch."

    # Single time
    time = torch.rand(())
    outputs = score_estimator(
        inputs, time=time, condition_mask=condition_masks, edge_mask=edge_masks
    )
    assert outputs.shape == (
        batch_dim,
        *input_event_shape,
    ), "Output shape mismatch."


def _build_masked_score_estimator_and_tensors(
    sde_type: str,
    input_event_shape: Tuple[int, int],
    batch_dim: int,
    **kwargs,
):
    """
    Helper function for all tests that deal with shapes of masked score estimators.
    """

    num_nodes, num_features = input_event_shape
    building_inputs = torch.randn((1000, num_nodes, num_features))

    score_estimator = build_masked_score_matching_estimator(
        building_inputs,
        building_inputs,  # not used
        sde_type=sde_type,
        **kwargs,
    )

    # ? Why using slices? This is done in the original build_score_estimator_and_tensors
    inputs = building_inputs[:batch_dim]
    condition_masks = torch.bernoulli(torch.rand(batch_dim, num_nodes))
    edge_masks = torch.ones(batch_dim, num_nodes, num_nodes)

    return score_estimator, inputs, condition_masks, edge_masks
