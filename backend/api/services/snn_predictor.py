import torch
import torch.nn as nn
import snntorch as snn
from snntorch import leaky

class NodePredictor(nn.Module):
    """
    A GRU-based network for predicting the next STAG node's embedding in a sequence.
    This network takes a sequence of historical node embeddings and outputs a
    predicted embedding for the next node.
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, self.embedding_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_sequence, h_0=None):
        """
        Performs a forward pass through the GRU.

        Args:
            input_sequence (torch.Tensor): A tensor of shape (batch_size, sequence_length, embedding_dim).
            h_0 (torch.Tensor, optional): Initial hidden state. Defaults to None.

        Returns:
            torch.Tensor: The predicted embedding for the next node, shape (batch_size, embedding_dim).
        """
        gru_out, _ = self.gru(input_sequence, h_0)
        # We only need the output of the last time step
        last_hidden_state = gru_out[:, -1, :]
        predicted_embedding = self.fc(last_hidden_state)
        return predicted_embedding

    def train_on_batch(self, input_sequence, target_embeddings):
        """
        Trains the NodePredictor on a batch of sequences.

        Args:
            input_sequence (torch.Tensor): Input sequences of node embeddings.
                                           Shape: (batch_size, sequence_length, embedding_dim)
            target_embeddings (torch.Tensor): Target next node embeddings.
                                              Shape: (batch_size, embedding_dim)
        """
        self.optimizer.zero_grad()
        predicted_embeddings = self.forward(input_sequence)
        loss = self.loss_fn(predicted_embeddings, target_embeddings)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_state(self):
        """Returns the state of the NodePredictor for serialization."""
        return {
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }

    @classmethod
    def from_state(cls, state_dict, device='cpu'):
        """Creates a NodePredictor instance from a state dictionary."""
        embedding_dim = state_dict.get('embedding_dim')
        if embedding_dim is None:
            raise KeyError("State is missing 'embedding_dim'.")

        predictor = cls(
            embedding_dim=embedding_dim,
            hidden_dim=state_dict['hidden_dim'],
            num_layers=state_dict.get('num_layers', 1)
        )
        predictor.load_state_dict(state_dict['state_dict'])
        predictor.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        predictor.to(device)
        return predictor
