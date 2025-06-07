# %%
import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from simformer import Simformer, MaskedVEScoreEstimator
from torch.optim.lr_scheduler import CosineAnnealingLR


# %%

class LinearGaussian(Dataset):
    def __init__(self, num_features, n):
        self.num_nodes = 3
        theta1 = np.random.normal(0, 3, size=(n, num_features))
        x1 = 2 * np.sin(theta1) + np.random.normal(0, 0.5, size=(n, num_features))
        x2 = 0.1 * theta1 ** 2 + 0.5 * np.abs(x1) * np.random.normal(0, 1, size=(n, num_features))
        data = np.concatenate([theta1, x1, x2], axis=1).reshape(n, -1, num_features)
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {'input': self.data[idx]}

# %%
# Set number of training epochs and learning rate
num_epochs = 100
lr = 1e-1

# Feature dimension size
in_features = 5

# Instantiate dataset and dataloader
train_dataset = LinearGaussian(num_features=in_features, n=10000)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Sequence dimension size
num_nodes = train_dataset.num_nodes

# Define model, optimizer, and any other necessary components.
# Instantiate Simformer
net = Simformer(
    in_features= in_features,
    num_nodes= num_nodes,
)

# Instantiate score estimator
model = MaskedVEScoreEstimator(net=net, input_shape=torch.Size([in_features, num_nodes]))

# Define optimizer
optimizer = AdamW(model.parameters(), lr=lr)

# Define scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Move model to device (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Moved model to {device}")

print(model)

# %%
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
        condition_mask = data_batch['condition_mask'].to(device) if 'condition_mask' in data_batch else None
        edge_mask = data_batch['edge_mask'].to(device) if 'edge_mask' in data_batch else None

        # Clear any accumulated gradients from the previous step
        optimizer.zero_grad()

        # Call the loss method score estimator.
        # This method will sample 'times', 'eps', and generate masks if they are None.
        # Remember that model.loss() returns loss *per batch element* ([B])
        # Remember that the forward() method is within the loss, i.e., you do *not* need to call forward()
        batch_losses = model.loss(
            input=input,
        )

        # Take the mean of the batch-wise losses to get a single scalar for backprop
        #print(f"Batch {batch_idx+1} with loss: {batch_losses}")
        loss = batch_losses.mean()
        #print(f"Batch {batch_idx+1} with loss: {loss}")

        # Accumulate loss for logging
        total_train_loss += loss.item()
        num_train_batches += 1

        # Compute gradients with respect to model parameters
        loss.backward()

        # Update model parameters using the calculated gradients
        optimizer.step()

    # Step the learning rate scheduler
    scheduler.step()

    # Compute average train loss
    avg_train_loss = total_train_loss / num_train_batches

    # Print stats for loss for the current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"\tTotal train loss: {total_train_loss}")
    print(f"\tNumber of train batches: {num_train_batches}")
    print(f"\tAverage train Loss: {avg_train_loss:.4f}")
# %%
