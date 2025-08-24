import torch
import torch.nn as nn
import numpy as np

class VisionCortex(nn.Module):
    """
    A cortex for processing image-based observations using a Convolutional Neural Network (CNN).
    The architecture is inspired by the Nature DQN paper (Mnih et al., 2015).
    """
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        # Input shape is expected to be (H, W, C)
        h, w, c = input_shape

        # The CNN architecture
        self.cnn = nn.Sequential(
            # PyTorch expects input as (N, C, H, W)
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # To determine the input size of the linear layer, we perform a forward pass with a dummy tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output_size = self.cnn(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of image tensors.
        Args:
            x (torch.Tensor): A tensor of shape (N, H, W, C).
        Returns:
            torch.Tensor: A tensor of shape (N, output_dim).
        """
        # PyTorch CNNs expect (N, C, H, W), but Gym envs often give (N, H, W, C).
        if x.dim() == 4 and x.shape[3] in [1, 3]: # A reasonable check for (N,H,W,C)
            x = x.permute(0, 3, 1, 2)

        # Normalize pixel values
        x = x / 255.0

        x = self.cnn(x)
        x = self.fc(x)
        return x

    def process(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Processes a single raw observation from the environment.
        This is typically used during the agent's "perceive" step.
        """
        # Add a batch dimension, convert to tensor
        obs_tensor = torch.from_numpy(raw_obs).float().unsqueeze(0)

        # Pass through the network
        with torch.no_grad():
            processed_obs_tensor = self.forward(obs_tensor)

        # Return as a numpy array, removing the batch dimension
        return processed_obs_tensor.squeeze(0).cpu().numpy()
