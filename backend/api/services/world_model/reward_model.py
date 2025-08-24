import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    This model predicts the reward for being in a particular latent state (z_t, h_t).
    """
    def __init__(self, latent_dim=32, hidden_dim=200):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )

    def forward(self, z_t, h_t):
        """
        Predict the reward from the latent state.

        Args:
            z_t (torch.Tensor): The stochastic latent state.
            h_t (torch.Tensor): The deterministic hidden state.

        Returns:
            torch.Tensor: The predicted reward.
        """
        # Concatenate the stochastic and deterministic states
        combined_state = torch.cat([z_t, h_t], dim=-1)

        # Predict the reward
        reward_pred = self.model(combined_state)

        return reward_pred
