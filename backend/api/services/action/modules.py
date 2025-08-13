import numpy as np

class ActionHead:
    """
    A simple feed-forward network to map an internal state representation
    to a vector of action logits.
    """
    def __init__(self, input_dim, n_actions=256):
        """
        Initializes the Action Head.

        Args:
            input_dim (int): The dimension of the input state vector from the brain.
            n_actions (int): The number of possible actions (e.g., 256).
        """
        self.input_dim = input_dim
        self.n_actions = n_actions

        # Initialize weights for a simple linear layer
        self.weights = np.random.randn(input_dim, n_actions) * 0.1
        self.biases = np.random.randn(n_actions) * 0.1

    def forward(self, state_vector: np.array):
        """
        Performs a forward pass to get action logits.

        Args:
            state_vector (np.array): The internal state from the agent's brain.

        Returns:
            np.array: A vector of logits for each possible action.
        """
        if state_vector.shape[0] != self.input_dim:
            raise ValueError(f"Input vector dimension {state_vector.shape[0]} does not match expected {self.input_dim}")

        return np.dot(state_vector, self.weights) + self.biases

    def get_state(self):
        """Returns the serializable state of the ActionHead."""
        return {
            'weights': self.weights,
            'biases': self.biases
        }

    def set_state(self, state):
        """Sets the state of the ActionHead from a dictionary."""
        self.weights = state['weights']
        self.biases = state['biases']
