# %%
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from sbi.neural_nets.estimators.score_estimator import (
    MaskedVEScoreEstimator,  # type: ignore
)
from sbi.neural_nets.net_builders.vector_field_nets import (  # type: ignore
    SimformerNet,
)


# %%
class LinearGaussian(Dataset):
    def __init__(self, num_features, n):
        self.num_nodes = 3
        self.num_features = num_features

        theta1 = np.random.normal(0, 3, size=(n, num_features))
        x1 = 2 * np.sin(theta1) + np.random.normal(0, 0.5, size=(n, num_features))
        x2 = 0.1 * theta1**2 + 0.5 * np.abs(x1) * np.random.normal(
            0, 1, size=(n, num_features)
        )

        data = np.concatenate([theta1, x1, x2], axis=1).reshape(n, -1, num_features)
        self.data = torch.from_numpy(data).float()

        self.mean_0_dataset = self.data.mean(dim=0)  # Shape: (num_nodes, num_features)
        self.std_0_dataset = self.data.std(dim=0)  # Shape: (num_nodes, num_features)
        self.std_0_dataset[self.std_0_dataset == 0] = 1e-8  # A small epsilon

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        edge_mask = torch.ones((self.num_nodes, self.num_nodes))
        condition_mask = torch.bernoulli(torch.full((self.num_nodes,), 0.33))
        return {
            'input': self.data[idx],
            'edge_mask': edge_mask,
            'condition_mask': condition_mask,
        }


# %%
# Set number of training epochs and learning rate
n = 50000
batch_size = 1024
num_epochs = 200
lr = 1e-4

# Feature dimension size
in_features = 5

# Instantiate dataset and dataloader
train_dataset = LinearGaussian(num_features=in_features, n=n)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Sequence dimension size
num_nodes = train_dataset.num_nodes

# Define model, optimizer, and any other necessary components.
# Instantiate SimformerNet
net = SimformerNet(
    in_features=in_features,
    num_nodes=num_nodes,
    ada_time=False,
)

# Instantiate score estimator
model = MaskedVEScoreEstimator(
    net=net,
    input_shape=torch.Size([num_nodes, in_features]),
    mean_0=train_dataset.mean_0_dataset,
    std_0=train_dataset.std_0_dataset,
    sigma_min=1e-3,  # Ensure sigma_min is not too small
    sigma_max=10.0,
)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=lr)

# Define scheduler
scheduler = None  # CosineAnnealingLR(optimizer, T_max=num_epochs)

# Move model to device (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Moved model to {device}")

print(model)

# %%

# TODO: Improve SimformerNet and MaskedVectorFieldEstimator
# to manage input shapes of [T, F]  and [T] (B=1, F=1), must be unsqueezed()

# Test single sample inference (no batch dimension)
# with torch.no_grad():
#     # Create a random input sample (num_nodes, in_features)
#     sample_input = torch.randn(num_nodes, in_features).to(device)
#     # Create edge mask (num_nodes, num_nodes)
#     sample_edge_mask = torch.ones(num_nodes, num_nodes).to(device)
#     # Create condition mask (num_nodes,)
#     sample_condition_mask = torch.bernoulli(torch.full((num_nodes,), 0.33)).to(device)

#     # Forward pass through the model's loss function (should work without batch dimension)
#     test_loss = model.loss(
#         input=sample_input,
#         edge_mask=sample_edge_mask,
#         condition_mask=sample_condition_mask,
#     )
#     print("Test loss (single sample, no batch dimension):", test_loss)

# %%
# Losses
lossi = []

# Loop for each epoch
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Initialize variables for tracking training loss
    total_train_loss = 0.0
    num_train_batches = 0

    # Loop over each mini-batch in the training data
    for batch_idx, data_batch in enumerate(train_loader):
        # Extract inputs, and potentially masks if they are part of your dataset
        input = data_batch['input'].to(device)
        edge_mask = (
            data_batch['edge_mask'].to(device) if 'edge_mask' in data_batch else None
        )
        condition_mask = (
            data_batch['condition_mask'].to(device)
            if 'condition_mask' in data_batch
            else None
        )

        # Clear any accumulated gradients from the previous step
        optimizer.zero_grad()

        # Call the loss method score estimator.
        # This method will sample 'times', 'eps', and generate masks if they are None.
        # Remember that model.loss() returns loss *per batch element* ([B])
        # Remember that the forward() method is within the loss, i.e., you do *not* need to call forward()
        batch_losses = model.loss(
            input=input,
            condition_mask=condition_mask,
            edge_mask=edge_mask,
        )

        # Take the mean of the batch-wise losses to get a single scalar for backprop
        # print(f"Batch {batch_idx+1} with loss: {batch_losses}")
        loss = batch_losses.mean()
        # print(f"Batch {batch_idx+1} with loss: {loss}")

        # Accumulate loss for logging
        total_train_loss += loss.item()
        num_train_batches += 1

        # Compute gradients with respect to model parameters
        loss.backward()

        # Update model parameters using the calculated gradients
        optimizer.step()

    # Step the learning rate scheduler
    if scheduler:
        scheduler.step()

    # Compute average train loss
    avg_train_loss = total_train_loss / num_train_batches

    # Save current loss
    lossi.append(avg_train_loss)

    # Print stats for loss for the current epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"\tAverage training Loss: {avg_train_loss:.4e}")
# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(lossi, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# %%
