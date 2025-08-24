import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class RepresentationModel(nn.Module):
    """
    This is the Representation Model (Encoder) of the RSSM. It encodes the observation
    into a stochastic latent state `z` by predicting the parameters of a Gaussian
    distribution. It also has a deterministic path that processes the observation.
    """
    def __init__(self, obs_dim, latent_dim=32, hidden_dim=200):
        super().__init__()
        # Encodes the observation
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        # From the encoded observation, predict the posterior distribution of the latent state
        self.fc_latent_posterior = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, obs, h_prev):
        """
        Encodes the observation to get the posterior distribution of the latent state.
        This is q(z_t | obs_t, h_{t-1}).
        """
        # The encoder's input is the raw observation
        encoded_obs = self.encoder(obs)

        # Predict posterior parameters
        latent_posterior_params = self.fc_latent_posterior(encoded_obs)
        posterior_mean, posterior_log_std = torch.chunk(latent_posterior_params, 2, dim=-1)

        # Ensure std is positive
        posterior_std = torch.nn.functional.softplus(posterior_log_std) + 1e-4

        return posterior_mean, posterior_std


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM). This is the core of the Dreamer agent.
    It combines the Representation Model and the Transition Model to learn a
    latent representation of the world and predict its dynamics.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.representation_model = RepresentationModel(obs_dim, latent_dim, hidden_dim)
        # The transition model is defined in its own file but instantiated here
        # We need to import it. Let's assume it's in the same directory.
        from .transition_model import TransitionModel
        self.transition_model = TransitionModel(action_dim, latent_dim, hidden_dim)

    def forward(self, obs, action, h_prev, z_prev):
        """
        The main forward pass of the RSSM for a single time step.

        Args:
            obs (torch.Tensor): Current observation.
            action (torch.Tensor): Previous action.
            h_prev (torch.Tensor): Previous deterministic hidden state.
            z_prev (torch.Tensor): Previous stochastic latent state.

        Returns:
            A tuple containing the new states and the KL loss component.
        """
        # 1. Get posterior from the current observation
        post_mean, post_std = self.representation_model(obs, h_prev)

        # 2. Get prior from the previous state and action (prediction)
        h_t, prior_mean, prior_std = self.transition_model(z_prev, action, h_prev)

        # 3. Sample from the posterior to get the current stochastic state
        # This is the reparameterization trick
        post_dist = Normal(post_mean, post_std)
        z_t = post_dist.rsample()

        # 4. Calculate the KL divergence loss
        # This pushes the posterior towards the prior, regularizing the latent space.
        prior_dist = Normal(prior_mean, prior_std)
        kl_loss = kl_divergence(post_dist, prior_dist).mean()

        return h_t, z_t, kl_loss
