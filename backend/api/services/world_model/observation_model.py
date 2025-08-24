import torch
import torch.nn as nn

class ObservationModel(nn.Module):
    """
    This model reconstructs the observation (e.g., the game screen) from the
    latent state (z_t, h_t). This forces the model to learn a representation
    that contains all the important visual information.
    """
    def __init__(self, obs_dim, latent_dim=32, hidden_dim=200):
        super().__init__()

        # The decoder reconstructs the observation from the combined latent and hidden states
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, obs_dim)
        )

    def forward(self, z_t, h_t):
        """
        Reconstruct the observation from the latent state.

        Args:
            z_t (torch.Tensor): The stochastic latent state.
            h_t (torch.Tensor): The deterministic hidden state.

        Returns:
            torch.Tensor: The reconstructed observation.
        """
        # Concatenate the stochastic and deterministic states
        combined_state = torch.cat([z_t, h_t], dim=-1)

        # Reconstruct the observation
        obs_recon = self.decoder(combined_state)

        return obs_recon
