import torch
import torch.nn as nn

from .world_model.rssm import RSSM
from .world_model.observation_model import ObservationModel
from .world_model.reward_model import RewardModel

class WorldModel(nn.Module):
    """
    The complete Dreamer-style World Model.
    This class encapsulates the RSSM, the Observation Model, and the Reward Model.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=200, goal_dim=None, hyperparams=None):
        super().__init__()
        self.hyperparams = hyperparams or {}
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.rssm = RSSM(obs_dim, action_dim, latent_dim, hidden_dim)
        self.obs_decoder = ObservationModel(obs_dim, latent_dim, hidden_dim)
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + goal_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )

    def get_initial_state(self, batch_size=1, device='cpu'):
        """Returns the initial hidden and latent states."""
        initial_h = torch.zeros(batch_size, self.hidden_dim, device=device)
        initial_z = torch.zeros(batch_size, self.latent_dim, device=device)
        return initial_h, initial_z
