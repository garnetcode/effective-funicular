import torch
import torch.nn as nn

from .predictive_coding import HierarchicalRSSM, PredictiveCodingModule
from .world_model.observation_model import ObservationModel
from .world_model.reward_model import RewardModel

class WorldModel(nn.Module):
    """
    The complete Dreamer-style World Model, adapted for a Hierarchical RSSM.
    This class encapsulates the Hierarchical RSSM (as self.rssm),
    the Observation Model, and the Reward Model.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=200, goal_dim=None, hyperparams=None):
        super().__init__()
        self.hyperparams = hyperparams or {}

        # --- Predictive Coding Hierarchy ---
        # Level 0: Interacts with the raw sensory embeddings
        l0_encoder = nn.Linear(obs_dim, latent_dim)
        l0_decoder = nn.Linear(latent_dim, obs_dim)
        level0 = PredictiveCodingModule(l0_encoder, l0_decoder, latent_dim, hidden_dim, action_dim)

        # Level 1: Interacts with the error from Level 0
        l1_encoder = nn.Linear(latent_dim, latent_dim)
        l1_decoder = nn.Linear(latent_dim, latent_dim)
        level1 = PredictiveCodingModule(l1_encoder, l1_decoder, latent_dim, hidden_dim, action_dim)

        self.rssm = HierarchicalRSSM([level0, level1])
        self.obs_decoder = ObservationModel(obs_dim, latent_dim, hidden_dim)
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + goal_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        )

    def get_initial_state(self, batch_size=1, device='cpu'):
        """Returns the initial hidden and latent states for the hierarchy."""
        num_levels = len(self.rssm.levels)
        initial_h = [torch.zeros(batch_size, level.hidden_dim, device=device) for level in self.rssm.levels]
        initial_z = [torch.zeros(batch_size, level.latent_dim, device=device) for level in self.rssm.levels]
        return initial_h, initial_z
