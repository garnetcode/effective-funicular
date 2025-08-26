import torch
import torch.nn as nn

from .world_model.rssm import RSSM
from .world_model.observation_model import ObservationModel
from .world_model.reward_model import RewardModel

class WorldModel(nn.Module):
    """
    The complete Dreamer-style World Model.
    This class encapsulates the RSSM, the Observation Model, and the Reward Model.
    Its sole purpose is to learn a model of the world.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=200, goal_dim=None, hyperparams=None):
        super().__init__()
        self.hyperparams = hyperparams or {}

        self.rssm = RSSM(obs_dim, action_dim, latent_dim, hidden_dim)
        self.obs_decoder = ObservationModel(obs_dim, latent_dim, hidden_dim)
        self.reward_model = RewardModel(latent_dim, hidden_dim, goal_dim)

    def get_initial_state(self, batch_size=1):
        """Returns the initial hidden and latent states."""
        return (
            torch.zeros(batch_size, self.rssm.hidden_dim),
            torch.zeros(batch_size, self.rssm.latent_dim)
        )

    def forward(self, obs, action, h_prev, z_prev, goal=None):
        """
        A full step of the world model, including encoding and prediction.
        This is used during the "Wake" phase to train the world model itself.

        Returns:
            obs_recon (torch.Tensor): Reconstructed observation.
            reward_pred (torch.Tensor): Predicted reward.
            kl_loss (torch.Tensor): KL divergence loss for regularization.
            h_t (torch.Tensor): New deterministic state.
            z_t (torch.Tensor): New stochastic state.
        """
        # 1. Update state based on observation using the RSSM
        h_t, z_t, kl_loss = self.rssm(obs, action, h_prev, z_prev)

        # 2. Reconstruct observation and predict reward from the new state
        obs_recon = self.obs_decoder(z_t, h_t)
        reward_pred = self.reward_model(z_t, h_t, goal)

        return obs_recon, reward_pred, kl_loss, h_t, z_t
