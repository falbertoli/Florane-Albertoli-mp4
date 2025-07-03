import torch.nn as nn
import torch

class PowerSystemNN(nn.Module):
    """
    Neural network model for electric power system branch overload prediction.
    This is a feedforward fully connected neural network using ReLU activations
    and sigmoid output to perform multi-label binary classification.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the network layers and activation functions.

        Parameters:
            input_dim (int): Number of selected input features (e.g., 53)
            output_dim (int): Number of binary output labels (e.g., 46 branches)
        """
        super(PowerSystemNN, self).__init__()

        # First fully connected layer: input_dim → 256 neurons
        self.fc1 = nn.Linear(input_dim, 256)

        # Second fully connected layer: 256 → 128 neurons
        self.fc2 = nn.Linear(256, 128)

        # Output layer: 128 → output_dim (one per branch)
        self.output_layer = nn.Linear(128, output_dim)

        # Activation functions
        self.relu = nn.ReLU()         # ReLU introduces non-linearity
        self.sigmoid = nn.Sigmoid()   # Sigmoid for multi-label binary outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the network — how data flows through the layers.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), with sigmoid applied
        """
        # Apply first hidden layer + ReLU
        x = self.relu(self.fc1(x))

        # Apply second hidden layer + ReLU
        x = self.relu(self.fc2(x))

        # Apply output layer + sigmoid (to return probability per branch)
        x = self.sigmoid(self.output_layer(x))

        return x

if __name__ == "__main__":
    model = PowerSystemNN(input_dim=53, output_dim=46)
    dummy_input = torch.randn(8, 53)  # Simulate a batch of 8 inputs
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be: torch.Size([8, 46])
