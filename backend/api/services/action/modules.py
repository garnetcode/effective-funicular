import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActionHead(nn.Module):
    """
    A feed-forward network to map an internal state representation to a
    distribution over actions.
    """
    def __init__(self, input_dim, n_actions=256, learning_rate=0.001):
        super(ActionHead, self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions

        self.layer = nn.Linear(input_dim, n_actions)
        # Optimizer is no longer managed here. It's managed in the agent's training loop.

    def forward(self, state_vector: torch.Tensor):
        """
        Performs a forward pass to get a distribution over actions.

        Args:
            state_vector (torch.Tensor): The internal state from the agent's brain.

        Returns:
            torch.distributions.Categorical: A distribution over actions.
        """
        if state_vector.shape[-1] != self.input_dim:
            raise ValueError(f"Input vector feature dimension {state_vector.shape[-1]} does not match expected {self.input_dim}")

        logits = self.layer(state_vector)
        return Categorical(logits=logits)

    def get_log_probs(self, state_vector, action):
        """
        Returns the log probability of taking a specific action given a state.
        """
        dist = self.forward(state_vector)
        return dist.log_prob(action)

    def get_entropy(self, state_vector):
        """
        Returns the entropy of the action distribution for a given state.
        """
        dist = self.forward(state_vector)
        return dist.entropy()

    # The get_state and set_state methods are no longer needed here,
    # as the agent will use self.action_head.state_dict() directly.
