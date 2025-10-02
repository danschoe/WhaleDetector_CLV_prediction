import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """
    CNN Model for LTV prediction as described in the paper.

    Architecture:
    - Input layer
    - Conv1D(32 filters, kernel_size=7) + ReLU
    - MaxPool1D(pool_size=2)
    - Conv1D(16 filters, kernel_size=3) + ReLU
    - Conv1D(1 filter, kernel_size=1) + ReLU
    - Flatten
    - FC(300) + ReLU
    - FC(150) + ReLU
    - FC(60) + ReLU
    - Output(1)
    """

    def __init__(self, input_channels: int, sequence_length: int):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 1, kernel_size=1)

        # Calculate flattened size after convolutions
        conv_output_size = sequence_length // 2

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc3 = nn.Linear(150, 60)
        self.output = nn.Linear(60, 1)

        self.relu = nn.ReLU()

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization as mentioned in the paper."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)

        return x.squeeze(-1)  # Remove last dimension for output


# Exemplary code of DNN from paper below
# class DNNModel(nn.Module):
#     """
#     Deep Multilayer Perceptron for LTV prediction as described in the paper.

#     Architecture can be scaled based on input features:
#     - Original paper: 203 features → 300 → 200 → 100 → 1
#     """

#     def __init__(self, input_features: int = 203, scale_architecture: bool = True):
#         super(DNNModel, self).__init__()

#         # Original paper architecture
#         hidden1, hidden2, hidden3 = 300, 200, 100

#         self.fc1 = nn.Linear(input_features, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, hidden3)
#         self.output = nn.Linear(hidden3, 1)

#         self.sigmoid = nn.Sigmoid()

#         # Initialize weights using Xavier initialization
#         self._initialize_weights()

#     def _initialize_weights(self):
#         """Xavier initialization as mentioned in the paper."""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.sigmoid(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         x = self.sigmoid(self.fc3(x))
#         x = self.output(x)
#         return x
