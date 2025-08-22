import torch
import torch.nn as nn

class StableHopfieldRNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, alpha=0.1):
        """
        Initializes the Stable Hopfield RNN.
        Args:
            num_nodes (int): The number of RNN cells in the network.
            hidden_dim (int): The dimension of the hidden state for each RNN cell.
            alpha (float): The stability gate parameter. A smaller value
                           promotes more stability.
        """
        super(StableHopfieldRNN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.alpha = alpha # Stability gate

        # Each node is a GRU cell
        self.rnn_cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_nodes)]
        )

        # Hopfield-like weight matrix for connections
        self.W = nn.Parameter(torch.randn(num_nodes, num_nodes))
        # Enforce symmetric weights and no self-connections
        self.W.data = 0.5 * (self.W.data + self.W.data.t())
        self.W.data.fill_diagonal_(0)

    def forward(self, initial_hidden_states, num_updates=20):
        """
        Performs the recall process by iteratively updating the hidden states.
        """
        # Ensure hidden_states is a list of tensors
        if not isinstance(initial_hidden_states, list):
             hidden_states = [h.squeeze(0) for h in torch.chunk(initial_hidden_states, self.num_nodes, dim=0)]
        else:
            hidden_states = initial_hidden_states


        for _ in range(num_updates):
            next_hidden_states = []
            for i in range(self.num_nodes):
                # Calculate the corrective input from the Hopfield network
                hopfield_input = torch.zeros_like(hidden_states[i])
                for j in range(self.num_nodes):
                    if i != j:
                        hopfield_input += self.W[i, j] * hidden_states[j]

                # The GRU proposes a new candidate state
                h_old = hidden_states[i]
                h_candidate = self.rnn_cells[i](hopfield_input.unsqueeze(0), h_old.unsqueeze(0)).squeeze(0)

                # --- Gated Update Rule ---
                # The new state is a stable blend of the old and candidate states.
                h_new = (1 - self.alpha) * h_old + self.alpha * h_candidate
                next_hidden_states.append(h_new)

            hidden_states = next_hidden_states

        # Return the final stable states as a single tensor
        return torch.stack(hidden_states)

    def energy(self, hidden_states):
        """
        Calculates the energy of the network state. Lower energy means a more stable state.
        """
        # Ensure hidden_states is a list of tensors
        if not isinstance(hidden_states, list):
             states_list = [h.squeeze(0) for h in torch.chunk(hidden_states, self.num_nodes, dim=0)]
        else:
            states_list = hidden_states

        total_energy = 0.0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                interaction = torch.dot(states_list[i], states_list[j])
                total_energy -= 0.5 * self.W[i, j] * interaction
        return total_energy

    def train_on_pattern(self, pattern, num_epochs=100, learning_rate=0.01, fixed_point_lambda=1.0):
        """
        Trains the network weights to store a given pattern.
        The loss is a combination of the pattern's energy and a fixed-point loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # The pattern is the target fixed point
        target_states = [h.squeeze(0) for h in torch.chunk(pattern, self.num_nodes, dim=0)]

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # 1. Calculate Energy Loss
            energy_loss = self.energy(target_states)

            # 2. Calculate Fixed-Point Loss
            fixed_point_loss = 0.0
            for i in range(self.num_nodes):
                # Calculate the input to the GRU if the network is in the target state
                hopfield_input = torch.zeros_like(target_states[i])
                for j in range(self.num_nodes):
                    if i != j:
                        hopfield_input += self.W[i, j] * target_states[j]

                # Get the candidate state from the GRU
                h_candidate = self.rnn_cells[i](hopfield_input.unsqueeze(0), target_states[i].unsqueeze(0)).squeeze(0)

                # The loss is the difference between the target state and the candidate state
                fixed_point_loss += torch.mean((target_states[i] - h_candidate) ** 2)

            # 3. Combine losses
            total_loss = energy_loss + fixed_point_lambda * fixed_point_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Re-enforce weight constraints after each step
            self.W.data = 0.5 * (self.W.data + self.W.data.t())
            self.W.data.fill_diagonal_(0)
