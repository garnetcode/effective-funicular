import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActionHead(nn.Module):
    """
    A feed-forward network to map an internal state representation to a
    distribution over actions. Can be conditioned on a goal.
    """
    def __init__(self, input_dim, n_actions=256, goal_dim=None, learning_rate=0.001):
        super(ActionHead, self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.goal_dim = goal_dim

        if self.goal_dim:
            self.layer = nn.Linear(input_dim + goal_dim, n_actions)
        else:
            self.layer = nn.Linear(input_dim, n_actions)

    def forward(self, state_vector: torch.Tensor, goal: torch.Tensor = None):
        """
        Performs a forward pass to get a distribution over actions.

        Args:
            state_vector (torch.Tensor): The internal state from the agent's brain.
            goal (torch.Tensor, optional): The goal vector. Defaults to None.

        Returns:
            torch.distributions.Categorical: A distribution over actions.
        """
        if self.goal_dim and goal is not None:
            state_vector = torch.cat([state_vector, goal], dim=-1)

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
