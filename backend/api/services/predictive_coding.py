import torch
import torch.nn as nn

class PredictiveCodingModule(nn.Module):
    def __init__(self, encoder, decoder, transition_model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transition_model = transition_model

    def forward(self, x, state):
        # Generate a top-down prediction
        prediction = self.transition_model(state)

        # Encode the actual input
        target = self.encoder(x)

        # Calculate the error
        error = target - prediction

        # Generate a reconstruction from the prediction
        reconstruction = self.decoder(prediction)

        return prediction, error, reconstruction


class HierarchicalRSSM(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = nn.ModuleList(levels)

    def forward(self, x, states):
        errors = []
        reconstructions = []
        next_states = []

        for i, (level, state) in enumerate(zip(self.levels, states)):
            # Pass the input to the lowest level, and the error from the level below to all other levels
            input_ = x if i == 0 else errors[-1]
            prediction, error, reconstruction = level(input_, state)
            errors.append(error)
            reconstructions.append(reconstruction)
            next_states.append(prediction)

        return next_states, errors, reconstructions
