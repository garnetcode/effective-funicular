import torch
import torch.nn as nn

class PredictiveCodingModule(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # The transition model is now a GRU, similar to the original RSSM
        self.transition_model = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        # A linear layer to predict the next latent state from the new hidden state
        self.fc_prediction = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, action, h_prev, z_prev):
        # One-hot encode the action
        a_one_hot = torch.nn.functional.one_hot(action.long(), num_classes=self.transition_model.input_size - z_prev.shape[-1]).float()
        if a_one_hot.dim() == 3:
            a_one_hot = a_one_hot.squeeze(1)

        # Update hidden state
        rnn_input = torch.cat([z_prev, a_one_hot], dim=-1)
        h_next = self.transition_model(rnn_input, h_prev)

        # Generate a top-down prediction from the new hidden state
        prediction = self.fc_prediction(h_next)

        # Encode the actual input
        target = self.encoder(x)

        # Calculate the error (the new latent state for the next level up)
        error = target - prediction
        z_next = error # The error is the stochastic part of the state

        # Generate a reconstruction from the prediction
        reconstruction = self.decoder(prediction)

        return h_next, z_next, prediction, error, reconstruction


class HierarchicalRSSM(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = nn.ModuleList(levels)

    def forward(self, x, action, prev_h, prev_z):
        all_h_next = []
        all_z_next = []
        all_errors = []
        all_reconstructions = []

        current_input = x

        for i, level in enumerate(self.levels):
            h_prev_i = prev_h[i]
            z_prev_i = prev_z[i]

            h_next, z_next, prediction, error, reconstruction = level(current_input, action, h_prev_i, z_prev_i)

            all_h_next.append(h_next)
            all_z_next.append(z_next)
            all_errors.append(error)
            all_reconstructions.append(reconstruction)

            current_input = error

        # To match the expected return signature at the call site in chimera_agent.py,
        # we return a dummy KL loss. A proper implementation would calculate the
        # KL divergence between the prediction and target at each level.
        kl_loss = torch.tensor(0.0, device=x.device)

        return all_h_next, all_z_next, all_reconstructions, all_errors, kl_loss
