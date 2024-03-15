import torch
import torch.nn as nn

class ConvolutionalNN(nn.Module):
    def __init__(self, width = 28):
        super(ConvolutionalNN, self).__init__()
        self.width=width
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * width//4 * width//4, 128)
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-3)
        # Forward pass through the convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # Reshape the tensor for the fully connected layers
        x = x.view(-1, 32 * self.width//4 * self.width//4)
        # Forward pass through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x