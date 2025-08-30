import torch
import torch.nn as nn
import snntorch as snn
from snntorch import leaky

class NodePredictor(nn.Module):
    """
    A GRU-based network for predicting the next STAG node in a sequence.
    This network takes a sequence of historical node embeddings and outputs a
    probability distribution over the next possible nodes.
    """
    def __init__(self, embedding_dim, hidden_dim, max_nodes, num_layers=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, self.max_nodes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_sequence, h_0=None):
        """
        Performs a forward pass through the GRU.

        Args:
            input_sequence (torch.Tensor): A tensor of shape (batch_size, sequence_length, embedding_dim).
            h_0 (torch.Tensor, optional): Initial hidden state. Defaults to None.

        Returns:
            torch.Tensor: Logits for the next node prediction, shape (batch_size, max_nodes).
        """
        gru_out, _ = self.gru(input_sequence, h_0)
        # We only need the output of the last time step
        last_hidden_state = gru_out[:, -1, :]
        logits = self.fc(last_hidden_state)
        return logits

    def train_on_batch(self, input_sequence, target_node_indices):
        """
        Trains the NodePredictor on a batch of sequences.

        Args:
            input_sequence (torch.Tensor): Input sequences of node embeddings.
                                           Shape: (batch_size, sequence_length, embedding_dim)
            target_node_indices (torch.Tensor): Target next node indices.
                                                 Shape: (batch_size,)
        """
        self.optimizer.zero_grad()
        logits = self.forward(input_sequence)
        loss = self.loss_fn(logits, target_node_indices)
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
            'max_nodes': self.max_nodes,
            'num_layers': self.num_layers
        }

    @classmethod
    def from_state(cls, state_dict, device='cpu'):
        """Creates a NodePredictor instance from a state dictionary."""
        embedding_dim = state_dict.get('embedding_dim', state_dict.get('input_dim'))
        if embedding_dim is None:
            raise KeyError("State is missing 'embedding_dim' or 'input_dim'.")

        predictor = cls(
            embedding_dim=embedding_dim,
            hidden_dim=state_dict['hidden_dim'],
            max_nodes=state_dict['max_nodes'],
            num_layers=state_dict.get('num_layers', 1)
        )
        predictor.load_state_dict(state_dict['state_dict'])
        predictor.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        predictor.to(device)
        return predictor
