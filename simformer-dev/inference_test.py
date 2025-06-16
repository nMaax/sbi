# %%
import torch
import numpy as np
from sbi.inference import NPE, Simformer # type: ignore
from sbi.utils import BoxUniform

_ = torch.manual_seed(0)

NUM_SIM_NODES = 2 # e.g., node 0 is 'theta', node 1 is 'x'
NUM_NODE_FEATURES = 3 # e.g., both theta1, x1 have 3 features

def simformer_simulator(num_simulations):
    theta_raw = torch.randn(num_simulations, NUM_NODE_FEATURES) * 2.0
    x_raw = theta_raw * 0.5 + 1.0 + torch.randn(num_simulations, NUM_NODE_FEATURES) * 0.1

    inputs_tensor = torch.stack([theta_raw, x_raw], dim=1)

    # True for observed, False for latent.
    # Note: This mask is for a single 'input' sample. It gets broadcasted for the batch.
    condition_mask_single_sample = torch.zeros((NUM_SIM_NODES,), dtype=torch.bool)
    condition_mask_single_sample[1] = True # Node 1 (index 1) is observed
    condition_masks = condition_mask_single_sample.unsqueeze(0).expand(num_simulations, NUM_SIM_NODES)

    # True for an edge from row to column.
    edge_mask_single_sample = torch.zeros((NUM_SIM_NODES, NUM_SIM_NODES), dtype=torch.bool)
    edge_mask_single_sample[0, 1] = True # Edge from node 0 to node 1
    edge_masks = edge_mask_single_sample.unsqueeze(0).expand(num_simulations, NUM_SIM_NODES, NUM_SIM_NODES)

    return inputs_tensor, condition_masks, edge_masks

# %%

# The actual diffusion will use an implicit Gaussian.
# This prior is used for bounding box checks if samples go out of reasonable range.
prior_low = -5 * torch.ones(NUM_SIM_NODES * NUM_NODE_FEATURES)
prior_high = 5 * torch.ones(NUM_SIM_NODES * NUM_NODE_FEATURES)
prior = BoxUniform(low=prior_low, high=prior_high)

# %%

inference = Simformer(
    prior=prior,
    vf_estimator="simformer",
    sde_type="ve",
    device="cpu",
)

# %%
num_simulations = 2000
sim_inputs, sim_conditioning_masks, sim_edge_masks = simformer_simulator(num_simulations)
print("sim_inputs.shape", sim_inputs.shape) # Expected: [2000, 2, 3]
print("sim_conditioning_masks.shape", sim_conditioning_masks.shape) # Expected: [2, 3]
print("sim_edge_masks.shape", sim_edge_masks.shape) # Expected: [2, 2]

# %%
inference = inference.append_simulations(
    inputs=sim_inputs,
    conditioning_masks=sim_conditioning_masks,
    edge_masks=sim_edge_masks,
)

# %%
density_estimator = inference.train()

print(density_estimator)

# %%
posterior = inference.build_posterior()

print(posterior)

# %%
x_obs = torch.as_tensor([0.8, 0.6, 0.4])

# %%
samples = posterior.sample((10000,), x=x_obs)

# %%
from sbi.analysis import pairplot

_ = pairplot(
    samples,
    limits=[[-2, 2], [-2, 2], [-2, 2]],
    figsize=(5, 5),
    labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
)

# %%
theta_posterior = posterior.sample((10000,), x=x_obs)  # Sample from posterior.
x_predictive = simulator(theta_posterior)  # Simulate data from posterior.

# %%
print("Posterior predictives: ", torch.mean(x_predictive, axis=0))
print("Observation: ", x_obs)

# %%
sim_edge_mask
