import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class PredictiveCodingModule(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transition_model = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # This layer now predicts the parameters of the prior distribution p(z_t | h_t)
        self.fc_prior = nn.Linear(hidden_dim, 2 * latent_dim)

        # This layer now predicts the parameters of the posterior distribution q(z_t | x_t, h_t)
        # The encoder output is concatenated with the hidden state to form the input
        self.fc_posterior = nn.Linear(encoder.out_features + hidden_dim, 2 * latent_dim)

    def forward(self, x, action, h_prev, z_prev):
        # One-hot encode the action
        a_one_hot = torch.nn.functional.one_hot(action.long(), num_classes=self.transition_model.input_size - z_prev.shape[-1]).float()
        if a_one_hot.dim() == 3:
            a_one_hot = a_one_hot.squeeze(1)

        # --- Transition Model (Prior) ---
        rnn_input = torch.cat([z_prev, a_one_hot], dim=-1)
        h_next = self.transition_model(rnn_input, h_prev)
        prior_params = self.fc_prior(h_next)
        prior_mean, prior_log_std = torch.chunk(prior_params, 2, dim=-1)
        prior_std = torch.nn.functional.softplus(prior_log_std) + 1e-4
        prior_dist = Normal(prior_mean, prior_std)

        # --- Representation Model (Posterior) ---
        encoded_x = self.encoder(x)
        posterior_input = torch.cat([encoded_x, h_next], dim=-1)
        posterior_params = self.fc_posterior(posterior_input)
        posterior_mean, posterior_log_std = torch.chunk(posterior_params, 2, dim=-1)
        posterior_std = torch.nn.functional.softplus(posterior_log_std) + 1e-4
        posterior_dist = Normal(posterior_mean, posterior_std)

        # Sample from the posterior for the next state (reparameterization trick)
        z_next = posterior_dist.rsample()

        # Generate a reconstruction from the sampled latent state
        reconstruction = self.decoder(z_next)

        # The 'error' for the next layer up is the sampled latent state itself
        error = z_next

        return h_next, z_next, (prior_dist, posterior_dist), error, reconstruction


class HierarchicalRSSM(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = nn.ModuleList(levels)

    def forward(self, x, action, prev_states):
        all_h_next = []
        all_z_next = []
        all_dists = []
        all_errors = []
        all_reconstructions = []

        current_input = x
        for i, level in enumerate(self.levels):
            h_prev, z_prev = prev_states[i]
            h_next, z_next, dists, error, reconstruction = level(current_input, action, h_prev, z_prev)

            all_h_next.append(h_next)
            all_z_next.append(z_next)
            all_dists.append(dists)
            all_errors.append(error)
            all_reconstructions.append(reconstruction)
            current_input = error

        return all_h_next, all_z_next, all_dists, all_errors, all_reconstructions

    def forward_sequence(self, obs_seq, action_seq):
        batch_size, seq_len, _ = obs_seq.shape
        # Initialize states for each level
        h_states = [torch.zeros(batch_size, level.transition_model.hidden_size, device=obs_seq.device) for level in self.levels]
        z_states = [torch.zeros(batch_size, level.fc_prior.out_features // 2, device=obs_seq.device) for level in self.levels]

        # To store sequences of states and other data
        seq_h, seq_z, seq_dists, seq_recons, seq_errors = [[] for _ in range(5)]

        for t in range(seq_len):
            obs_t = obs_seq[:, t]
            action_t = action_seq[:, t]

            prev_states = list(zip(h_states, z_states))
            h_next, z_next, dists, errors, recons = self.forward(obs_t, action_t, prev_states)

            # Store results for this time step
            # We need to handle the list-of-lists structure
            if t == 0:
                for i in range(len(self.levels)):
                    seq_h.append([])
                    seq_z.append([])
                    seq_dists.append([])
                    seq_errors.append([])
                seq_recons.append([]) # Assuming one reconstruction from the bottom layer

            for i in range(len(self.levels)):
                seq_h[i].append(h_next[i])
                seq_z[i].append(z_next[i])
                seq_dists[i].append(dists[i])
                seq_errors[i].append(errors[i])
            seq_recons[0].append(recons[0]) # Only store reconstruction from the bottom layer

            h_states, z_states = h_next, z_next

        # Stack all collected tensors along the sequence dimension
        for i in range(len(self.levels)):
            seq_h[i] = torch.stack(seq_h[i], dim=1)
            seq_z[i] = torch.stack(seq_z[i], dim=1)
            # Dists are tuples of distributions, handle separately
            # seq_errors[i] = torch.stack(seq_errors[i], dim=1)
        seq_recons[0] = torch.stack(seq_recons[0], dim=1)

        return seq_h, seq_z, seq_dists, seq_recons, seq_errors

    def calculate_kl_loss(self, dists_seq, free_bits=1.0):
        total_kl_loss = 0
        # dists_seq is a list (levels) of lists (timesteps) of tuples (prior, posterior)
        for level_dists in dists_seq:
            level_kl = 0
            for prior_dist, post_dist in level_dists:
                # Sum over the latent dimensions, then mean over the batch
                kl = kl_divergence(post_dist, prior_dist).sum(dim=-1).mean()
                level_kl += kl

            # Average over sequence length
            level_kl /= len(level_dists)
            total_kl_loss += level_kl

        # Apply free bits to the total KL loss
        kl_loss = torch.max(torch.tensor(0.0, device=total_kl_loss.device), total_kl_loss - free_bits)
        return kl_loss, free_bits
