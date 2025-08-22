import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActionHead(nn.Module):
    """
    A feed-forward network to map an internal state representation to a vector
    of action logits, implemented as a PyTorch module.
    """
    def __init__(self, input_dim, n_actions=256, learning_rate=0.001):
        """
        Initializes the Action Head.

        Args:
            input_dim (int): Dimension of the input state vector.
            n_actions (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(ActionHead, self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions

        self.layer = nn.Linear(input_dim, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state_vector: np.array):
        """
        Performs a forward pass to get action logits.

        Args:
            state_vector (np.array): The internal state from the agent's brain.

        Returns:
            torch.Tensor: A tensor of logits for each possible action.
        """
        if not isinstance(state_vector, torch.Tensor):
            state_vector = torch.from_numpy(state_vector).float()

        # The input can be a single vector or a batch of vectors.
        # The feature dimension is always the last one.
        if state_vector.shape[-1] != self.input_dim:
            raise ValueError(f"Input vector feature dimension {state_vector.shape[-1]} does not match expected {self.input_dim}")

        return self.layer(state_vector)

    def get_state(self):
        """Returns the serializable state of the ActionHead."""
        return {
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

    def set_state(self, state):
        """Sets the state of the ActionHead from a dictionary."""
        self.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
