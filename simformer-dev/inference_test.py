# %%
import torch

from sbi.inference import Simformer  # type: ignore
from sbi.utils import BoxUniform
from torch import nn

_ = torch.manual_seed(0)

NUM_SIM_NODES = 4
NUM_NODE_FEATURES = 5
NUM_OBS_NODES = 2
NUM_LAT_NODES = NUM_SIM_NODES - NUM_OBS_NODES

def simformer_simulator(num_simulations):
    theta1 = torch.randn(num_simulations, NUM_NODE_FEATURES) * 3.0 + 12
    theta2 = torch.randn(num_simulations, NUM_NODE_FEATURES) * 1.5 + 4.0

    x1 = theta1 + torch.randn_like(theta1)
    x2 = theta2 + torch.randn_like(theta2)

    inputs_tensor = torch.stack([theta1, theta2, x1, x2], dim=1)

    condition_masks = torch.bernoulli(
        torch.full((num_simulations, NUM_SIM_NODES), 0.5)
    ).bool()
    for i in range(num_simulations):
        if not condition_masks[i].any():
            rand_idx = torch.randint(0, NUM_SIM_NODES, (1,))
            condition_masks[i, rand_idx] = True

    edge_mask_single_sample = torch.ones(
        (NUM_SIM_NODES, NUM_SIM_NODES), dtype=torch.bool
    )
    edge_masks = edge_mask_single_sample.unsqueeze(0).expand(
        num_simulations, NUM_SIM_NODES, NUM_SIM_NODES
    )

    return inputs_tensor, condition_masks, edge_masks


# %%

# The actual diffusion will use an implicit Gaussian.
# This prior is used for bounding box checks if samples go out of reasonable range
prior_low = -25 * torch.ones(NUM_LAT_NODES * NUM_NODE_FEATURES)
prior_high = 25 * torch.ones(NUM_LAT_NODES * NUM_NODE_FEATURES)
prior = BoxUniform(low=prior_low, high=prior_high, device="gpu")

# %%

inference: Simformer = Simformer(
    prior=prior,
    vf_estimator="simformer",
    sde_type="ve",
    device="gpu",
    hidden_features=100,
    num_layers=5,
    num_heads=4,
    mlp_ratio=2,
    time_embedding_dim=32,
    embedding_net=nn.Identity(),
    dim_val=64,
    dim_id=32,
    dim_cond=16,
    ada_time=False, # TODO: fix, raises error at inference time
    time_emb_type="sinusoidal",
    sinusoidal_max_freq=0.01,
    fourier_scale=30.0,
    activation=nn.SiLU,
)

print(inference)

# %%
num_simulations = 1000
sim_inputs, sim_condition_masks, sim_edge_masks = simformer_simulator(num_simulations)
print("sim_inputs.shape", sim_inputs.shape)  # Expected: [2000, 2, 3]
print("sim_condition_masks.shape", sim_condition_masks.shape)  # Expected: [2, 3]
print("sim_edge_masks.shape", sim_edge_masks.shape)  # Expected: [2, 2]

# %%
inference.append_simulations(
    inputs=sim_inputs,
    condition_masks=sim_condition_masks,
    edge_masks=sim_edge_masks,
)

# %%
density_estimator = inference.train()

print(density_estimator)

# %%
import matplotlib.pyplot as plt

# Plot the validation loss from the inference summary
validation_loss = inference.summary['validation_loss']
plt.plot(validation_loss)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss over Epochs')
plt.show()

# %%

condition_mask_single_sample = torch.zeros((NUM_SIM_NODES,), dtype=torch.bool)
condition_mask_single_sample[2] = True  # Index 2 is observed
condition_mask_single_sample[3] = True  # Index 3 is observed

edge_mask_single_sample = torch.ones((NUM_SIM_NODES, NUM_SIM_NODES), dtype=torch.bool)

posterior = inference.build_posterior(
    condition_mask=condition_mask_single_sample,
    #edge_mask=edge_mask_single_sample,
)

# %%

# Prepare data
x_obs = torch.tensor([
    [*[12.7]*NUM_NODE_FEATURES, *[4.3]*NUM_NODE_FEATURES],
    [*[11.8]*NUM_NODE_FEATURES, *[4.5]*NUM_NODE_FEATURES],
    [*[12.8]*NUM_NODE_FEATURES, *[3.5]*NUM_NODE_FEATURES],
], device="cuda")
print(f"{x_obs.shape=}")

# %%

samples = posterior.sample_batched(torch.Size((1000,)), x=x_obs)

print(f"{samples.shape=}")

# %%

from sbi.analysis import pairplot
import numpy as np

num_samples = samples.shape[0]
last_batch_samples = samples[:, -1, :]  # Select samples for the last batch

_ = pairplot(
    last_batch_samples.reshape(-1, NUM_LAT_NODES * NUM_NODE_FEATURES),
    limits=[[0, 16]] * (NUM_LAT_NODES * NUM_NODE_FEATURES),
    figsize=(8, 8),
    labels=[rf"$\theta_{{{i + 1}}}$" for i in range(NUM_LAT_NODES * NUM_NODE_FEATURES)],
    ticks=[[4, 12] for _ in range(NUM_LAT_NODES * NUM_NODE_FEATURES)],
)

# %%

def simulate_from_theta(theta_samples):
    num_samples = theta_samples.shape[0]
    theta1 = theta_samples[:, 0].unsqueeze(1)
    theta2 = theta_samples[:, 1].unsqueeze(1)
    x1 = theta1 + torch.randn_like(theta1)
    x2 = theta2 + torch.randn_like(theta2)
    x_obs_sim = torch.cat([x1, x2], dim=1)
    return x_obs_sim

x_predictive = simulate_from_theta(samples.cpu())

print("Posterior mean theta:", samples.mean(dim=0))
print("Posterior predictives mean: ", torch.mean(x_predictive, axis=0)) # type: ignore
print("Observation: ", x_obs)

# %%
