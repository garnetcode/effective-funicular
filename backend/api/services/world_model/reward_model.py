import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    This model predicts the reward. If a goal is provided, the reward is the
    negative distance to the goal. Otherwise, it predicts the reward from the state.
    """
    def __init__(self, latent_dim=32, hidden_dim=200, goal_dim=None):
        super().__init__()
        self.goal_dim = goal_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )


    def forward(self, z_t, h_t, goal=None):
        """
        Predict the reward.
        """
        if self.goal_dim is not None and goal is not None:
            # The state representation used for goal distance can be either z or h.
            # Using h as it's the deterministic, temporally-aware state.
            return -torch.norm(h_t - goal, dim=-1, keepdim=True)
        else:
            combined_state = torch.cat([z_t, h_t], dim=-1)
            reward_pred = self.model(combined_state)
            return reward_pred
