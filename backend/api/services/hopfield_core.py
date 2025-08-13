# Implements the adaptive associative memory substrate (Hopfield Core).
# This is based on a classical, recurrent Hopfield network with a
# synaptic weight matrix (W) and Hebbian-based learning.
# See Section 2.2 of the Project Chimera specification.

import numpy as np

class HopfieldCore:
    def __init__(self, dimensions, learning_rate=0.1, weight_decay=0.01):
        """
        Initializes the Hopfield Core.

        Args:
            dimensions (int): The dimensionality (N) of the pattern vectors.
            learning_rate (float): The learning rate (η).
            weight_decay (float): The weight decay factor (α).
        """
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # The N x N synaptic weight matrix, initialized to zeros.
        self.weights = np.zeros((dimensions, dimensions))

    def learn(self, pattern):
        """
        Learns a new pattern using a Hebbian-based rule with weight decay.
        ΔW = ηξξT
        W_new = (1 - α)W_old + ΔW

        Args:
            pattern (np.array): The N-dimensional pattern vector (ξ) to learn.
        """
        if pattern.shape[0] != self.dimensions:
            raise ValueError(f"Pattern dimension {pattern.shape[0]} does not match network dimension {self.dimensions}")

        # Ensure pattern is a column vector for outer product
        pattern = pattern.reshape(-1, 1)
        delta_w = self.learning_rate * np.dot(pattern, pattern.T)

        self.weights = (1 - self.weight_decay) * self.weights + delta_w
        # Ensure diagonal elements are zero to prevent self-connection
        np.fill_diagonal(self.weights, 0)

    def recall(self, cue, max_iter=100, convergence_threshold=1e-6):
        """
        Retrieves a stored pattern from a cue vector.
        The network state evolves until it settles into a stable attractor state.

        Args:
            cue (np.array): The initial N-dimensional cue vector.
            max_iter (int): The maximum number of iterations to perform.

        Returns:
            np.array: The stable attractor state (recalled memory).
        """
        state = np.copy(cue)

        for _ in range(max_iter):
            prev_state = np.copy(state)

            # Asynchronous update (one neuron at a time)
            for i in range(self.dimensions):
                raw_input = np.dot(self.weights[i, :], state)
                state[i] = 1 if raw_input > 0 else -1 # Using +1/-1 bipolar states

            # Check for convergence
            if np.all(state == prev_state):
                return state

        # print("Recall did not converge, returning last state.")
        return state

    def get_state(self):
        """Returns the serializable state of the Hopfield core."""
        return {
            'dimensions': self.dimensions,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'weights': self.weights.tolist()
        }

    @classmethod
    def from_state(cls, state_dict):
        """Creates a HopfieldCore instance from a state dictionary."""
        core = cls(
            dimensions=state_dict['dimensions'],
            learning_rate=state_dict['learning_rate'],
            weight_decay=state_dict['weight_decay']
        )
        core.weights = np.array(state_dict['weights'])
        return core
