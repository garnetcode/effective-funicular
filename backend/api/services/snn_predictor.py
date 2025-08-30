import torch
import torch.nn as nn
import snntorch as snn
from snntorch import leaky

class SNNPredictor(nn.Module):
    """
    A Spiking Neural Network for predicting the next state in a sequence.
    This network takes a sequence of historical states and outputs a prediction
    for the subsequent state. It's designed to be integrated into the STAG framework
    to provide a "reflexive" prediction of what might happen next.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_steps):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps # Number of time steps to simulate the SNN for

        # --- Network Architecture ---
        # Layer 1: Input to Hidden
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9) # Leaky Integrate-and-Fire neuron

        # Layer 2: Hidden to Output
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.lif2 = snn.Leaky(beta=0.9)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_sequence):
        """
        Performs a forward pass through the SNN for a given number of time steps.

        Args:
            input_sequence (torch.Tensor): A tensor of shape (batch_size, sequence_length, input_dim)
                                           representing the historical states.

        Returns:
            torch.Tensor: The predicted next state, shape (batch_size, output_dim).
        """
        batch_size, sequence_length, _ = input_sequence.shape

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer's membrane potential
        mem2_recording = []

        # SNN simulation loop
        for step in range(self.num_steps):
            # The input to the SNN at each time step is one vector from the sequence
            # We loop through the input sequence
            current_input = input_sequence[:, step % sequence_length, :]

            cur1 = self.fc1(current_input)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_recording.append(mem2)

        # The prediction is the average membrane potential of the output layer over the simulation
        prediction = torch.stack(mem2_recording, dim=0).mean(dim=0)
        return prediction

    def train_on_batch(self, input_sequence, target_sequence):
        """
        Trains the SNN on a batch of sequences.

        Args:
            input_sequence (torch.Tensor): The input sequences for the model.
                                           Shape: (batch_size, sequence_length, input_dim)
            target_sequence (torch.Tensor): The target next states.
                                            Shape: (batch_size, output_dim)
        """
        self.optimizer.zero_grad()

        # Get the prediction from the model
        prediction = self.forward(input_sequence)

        # Calculate loss
        loss = self.loss_fn(prediction, target_sequence)

        # Backpropagate and update weights
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_state(self):
        """Returns the state of the SNN for serialization."""
        return {
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_steps': self.num_steps
        }

    @classmethod
    def from_state(cls, state_dict, device='cpu'):
        """Creates an SNNPredictor instance from a state dictionary."""
        predictor = cls(
            input_dim=state_dict['input_dim'],
            hidden_dim=state_dict['hidden_dim'],
            output_dim=state_dict['output_dim'],
            num_steps=state_dict['num_steps']
        )
        predictor.load_state_dict(state_dict['state_dict'])
        predictor.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        predictor.to(device)
        return predictor
