import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_reader import DataReader


class Trainer:
    """
    Class responsible for training a neural network model.
    """
    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the Trainer with a model and an optional learning rate.
        
        Parameters:
            model (nn.Module): The neural network model to be trained.
        """
        self.model = model

        # Binary Cross-Entropy loss for multi-label classification
        self.criterion = nn.BCELoss()

        # Adam optimizer with learning rate 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, data_reader: DataReader) -> None:
        """
        Trains the model on the dataset provided by the DataReader instance.

        Parameters:
            data_reader: Contains normalized tensors for training (X_tensor, y_tensor)
        """
        # Wrap training data into a PyTorch DataLoader for mini-batching
        train_dataset = TensorDataset(data_reader.X_tensor, data_reader.y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Number of training epochs
        epochs = 20

        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # 1. Zero gradients from the previous iteration
                self.optimizer.zero_grad()

                # 2. Forward pass
                outputs = self.model(inputs)

                # 3. Compute loss
                loss = self.criterion(outputs, targets)

                # 4. Backward pass
                loss.backward()

                # 5. Optimizer step (update weights)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")