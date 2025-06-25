# %%
import torch

from sbi.inference import Simformer  # type: ignore
from sbi.utils import BoxUniform

_ = torch.manual_seed(0)

NUM_SIM_NODES = 4
NUM_NODE_FEATURES = 2
NUM_OBS_NODES = 2
NUM_LAT_NODES = NUM_SIM_NODES - NUM_OBS_NODES

def simformer_simulator(num_simulations):
    theta1 = torch.randn(num_simulations, NUM_NODE_FEATURES)
    theta2 = torch.randn(num_simulations, NUM_NODE_FEATURES)
    #theta1 = torch.randn(num_simulations, NUM_NODE_FEATURES) * 3.0
    #theta2 = torch.randn(num_simulations, NUM_NODE_FEATURES) * 1.5 + 2.0
    x1 = theta1 + torch.randn(num_simulations, NUM_NODE_FEATURES) * 0.2
    x2 = theta2 + torch.randn(num_simulations, NUM_NODE_FEATURES) * 0.2
    # x1 = 2 * torch.sin(theta1) + torch.randn(num_simulations, NUM_NODE_FEATURES) * 0.5
    # x2 = 0.1 * theta1**2 + 0.3 * theta2 + 0.5 * torch.abs(x1) * torch.randn(
    #     num_simulations, NUM_NODE_FEATURES
    # )

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
prior_low = -50 * torch.ones(NUM_LAT_NODES * NUM_NODE_FEATURES)
prior_high = 50 * torch.ones(NUM_LAT_NODES * NUM_NODE_FEATURES)
prior = BoxUniform(low=prior_low, high=prior_high, device="gpu")

# %%

inference: Simformer = Simformer(
    prior=prior,
    vf_estimator="simformer",
    sde_type="ve",
    device="gpu",
)

# %%
num_simulations = 1000
sim_inputs, sim_condition_masks, sim_edge_masks = simformer_simulator(
    num_simulations
)
print("sim_inputs.shape", sim_inputs.shape)  # Expected: [2000, 2, 3]
print("sim_conditioning_masks.shape", sim_condition_masks.shape)  # Expected: [2, 3]
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

# ! Take the network I have, and do the sampling from scratch

posterior = inference.build_posterior(
    condition_mask=condition_mask_single_sample,
    edge_mask=edge_mask_single_sample,
)

# %%
x_obs = torch.as_tensor([
    [0.5,], #  0.8, 1.5, -1.8, -1.6, 0.1],
    [-0.8,], #  1.6, 0.9, -0.3, 0.25, 0.4],
])
x_obs = x_obs.view(1, -1)

# %%

samples = posterior.sample((1000,), x=x_obs)
# samples = samples.reshape(-1, NUM_LAT_NODES, NUM_NODE_FEATURES)

print(f"{samples.shape=}")

# %%

# TODO: Do not work with Simformer shapes

from sbi.analysis import pairplot

_ = pairplot(
    samples.reshape(-1, NUM_LAT_NODES * NUM_NODE_FEATURES),
    #limits=[[-10, 10]] * (NUM_LAT_NODES * NUM_NODE_FEATURES),
    figsize=(8, 8),
    labels=[fr"$\theta_{{{i+1}}}$" for i in range(NUM_LAT_NODES * NUM_NODE_FEATURES)],
)

# %%
# theta_posterior = posterior.sample((10000,), x=x_obs)  # Sample from posterior.
# x_predictive = simulator(theta_posterior)  # Simulate data from posterior.

# # %%
# print("Posterior predictives: ", torch.mean(x_predictive, axis=0))
# print("Observation: ", x_obs)
