import torch
import torch.nn as nn

class TransitionModel(nn.Module):
    """
    This model predicts the next latent state (z_{t+1}, h_{t+1}) given the
    current state (z_t, h_t) and an action (a_t). It acts as the agent's
    internal physics simulator.
    """
    def __init__(self, action_dim, latent_dim=32, hidden_dim=200):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input to the GRU is the stochastic latent state and the action
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Predicts the parameters of the next stochastic latent state distribution
        self.fc_latent_prior = nn.Linear(hidden_dim, 2 * latent_dim) # For mean and std dev

    def forward(self, z_t, a_t, h_t):
        """
        Predict the next state.

        Args:
            z_t (torch.Tensor): The stochastic latent state from the previous step.
            a_t (torch.Tensor): The action taken at the previous step.
            h_t (torch.Tensor): The deterministic hidden state from the previous step.

        Returns:
            tuple: (
                h_next (torch.Tensor): The next deterministic hidden state.
                prior_mean (torch.Tensor): The mean of the predicted latent state distribution.
                prior_std (torch.Tensor): The std dev of the predicted latent state distribution.
            )
        """
        # One-hot encode the action
        a_one_hot = torch.nn.functional.one_hot(a_t.long(), num_classes=self.action_dim).float()

        # Concatenate latent state and action
        rnn_input = torch.cat([z_t, a_one_hot], dim=-1)

        # Update the hidden state
        h_next = self.gru(rnn_input, h_t)

        # Predict the next latent state (prior)
        latent_prior_params = self.fc_latent_prior(h_next)
        prior_mean, prior_log_std = torch.chunk(latent_prior_params, 2, dim=-1)

        # Apply softplus to ensure std is positive
        prior_std = torch.nn.functional.softplus(prior_log_std) + 1e-4

        return h_next, prior_mean, prior_std
