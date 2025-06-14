# %%
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform

_ = torch.manual_seed(0)

num_dim = 3
def simulator(theta):
    # Linear Gaussian.
    return theta + 1.0 + torch.randn_like(theta) * 0.1

# %%

prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

# %%

inference = NPE(prior=prior)

# %%
num_simulations = 2000
theta = prior.sample((num_simulations,))
x = simulator(theta)
print("theta.shape", theta.shape)
print("x.shape", x.shape)

# %%
inference = inference.append_simulations(theta, x)

# %%
density_estimator = inference.train()

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
