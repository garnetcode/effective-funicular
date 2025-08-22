import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    """
    A Recurrent Latent World Model that combines an encoder, a recurrent core,
    and a decoder to predict future states and rewards.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=64, hidden_dim=128):
        super(WorldModel, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 1. Encoder: Compresses observation into a latent state (z)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 2. Recurrent Core (GRU): Models temporal dynamics
        # Input to GRU is the latent state + one-hot encoded action
        self.recurrent_core = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # 3. Decoder: Reconstructs observation and predicts reward from hidden state (h)
        self.decoder_obs = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        self.decoder_reward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action, h_prev):
        """
        Performs one step of the world model loop.

        Args:
            obs (torch.Tensor): The current observation from the environment.
            action (torch.Tensor): The action taken in the previous step.
            h_prev (torch.Tensor): The previous hidden state of the recurrent core.

        Returns:
            tuple: (z, h_next, obs_pred, reward_pred)
                   - z: The new latent state.
                   - h_next: The next hidden state.
                   - obs_pred: The predicted next observation.
                   - reward_pred: The predicted reward.
        """
        # Ensure obs is 2D (batch_size, obs_dim)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if h_prev is not None and h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)

        # 1. Encode the observation to get the latent state
        z = self.encoder(obs)

        # 2. Prepare input for the recurrent core
        # One-hot encode the action, ensuring it's 2D
        action_squeezed = action.long().squeeze()
        if action_squeezed.dim() == 0:
            action_squeezed = action_squeezed.unsqueeze(0)
        action_one_hot = F.one_hot(action_squeezed, num_classes=self.action_dim).float()

        # Concatenate latent state and action
        rnn_input = torch.cat([z, action_one_hot], dim=1)

        # 3. Update the hidden state
        h_next = self.recurrent_core(rnn_input, h_prev)

        # 4. Decode the new hidden state to make predictions
        obs_pred = self.decoder_obs(h_next)
        reward_pred = self.decoder_reward(h_next)

        return z.squeeze(0), h_next.squeeze(0), obs_pred.squeeze(0), reward_pred.squeeze(0)
