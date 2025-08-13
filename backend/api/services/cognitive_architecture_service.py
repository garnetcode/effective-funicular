# Implements the CognitiveArchitectureService, the central orchestrator for the system.
# This service manages the in-memory state of all three cognitive layers:
# 1. The Hopfield Core (associative memory)
# 2. The STAG Framework (hierarchical topological memory)
# It handles the primary operations: learning, organizing, and persistence.
# See Section 5.1 of the Project Chimera specification.

import os
import json
import numpy as np
from .hopfield_core import HopfieldCore
from .stag_framework import STAG_Framework

# A placeholder for a real text embedding model (like from the Visual/Text Cortex)
def text_to_embedding(text, dimensions=64):
    """
    A simple, deterministic function to convert text to a fixed-size bipolar vector.
    This is a stand-in for a proper sentence transformer or other embedding model.
    """
    # Create a hash of the text to seed the random number generator for reproducibility
    seed = hash(text)
    rng = np.random.RandomState(seed)
    # Generate a random vector and convert it to a bipolar (-1, 1) vector
    vec = rng.randn(dimensions)
    vec = np.sign(vec)
    # Ensure at least one element is non-zero
    if np.all(vec == 0):
        vec[0] = 1
    return vec

class CognitiveArchitectureService:
    def __init__(self, network_id, dimensions=64, load_from_storage=True, **hyperparams):
        """
        Initializes the entire cognitive architecture.

        Args:
            network_id (str): A unique identifier for this network instance.
            dimensions (int): The dimensionality of the embedding space.
            load_from_storage (bool): If True, tries to load the state from file.
            **hyperparams: A dictionary of hyperparameters for the components.
        """
        self.network_id = network_id
        self.dimensions = dimensions
        self.storage_path = os.path.join('backend', 'storage', f'{self.network_id}.json')

        if load_from_storage and os.path.exists(self.storage_path):
            self.load_state()
        else:
            self.hopfield = HopfieldCore(dimensions, **hyperparams)
            self.stag = STAG_Framework(dimensions, **hyperparams)
            self.patterns = [] # Log of source patterns
            self.save_state()

    def save_state(self):
        """Serializes the entire cognitive architecture state to a JSON file."""
        state = {
            'network_id': self.network_id,
            'dimensions': self.dimensions,
            'hopfield_state': self.hopfield.get_state(),
            'stag_state': self.stag.get_serializable_structure(),
            'patterns': self.patterns
        }
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Loads the architecture's state from a JSON file."""
        with open(self.storage_path, 'r') as f:
            state = json.load(f)

        self.dimensions = state['dimensions']
        hyperparams = state.get('hopfield_state', {}) # Assumes params are stored in hopfield state
        self.hopfield = HopfieldCore.from_state(state['hopfield_state'])
        self.stag = STAG_Framework.from_serializable_structure(state['stag_state'], **hyperparams)
        self.patterns = state.get('patterns', [])

    def learn_pattern(self, text_input):
        """
        High-level method to learn a new pattern from text.
        This corresponds to the 'LEARN Mode' in the specification.
        """
        # 1. Convert text to an embedding vector
        embedding = text_to_embedding(text_input, self.dimensions)

        # 2. Update the Hopfield weight matrix
        self.hopfield.learn(embedding)

        # 3. Log the pattern
        self.patterns.append(text_input)

        # 4. Persist the new state
        self.save_state()

        return {"status": "Pattern learned", "embedding": embedding.tolist()}

    def organize_step(self, cue_text=None):
        """
        Performs one step of the organization process.
        This corresponds to the 'ORGANIZE Mode' in the specification.
        """
        if not self.patterns:
            return {"status": "No patterns to organize."}

        # 1. Select a pattern to use as a cue
        if cue_text is None:
            cue_text = np.random.choice(self.patterns)

        cue_embedding = text_to_embedding(cue_text, self.dimensions)

        # 2. Let the Hopfield network recall the stable attractor
        stable_attractor = self.hopfield.recall(cue_embedding)

        # 3. Pass the stable attractor to the STAG/GNG layer for training
        self.stag.process_input(stable_attractor)

        # 4. Persist the new state
        self.save_state()

        return {"status": "Organization step complete", "stable_attractor": stable_attractor.tolist()}

    def get_graph_structure(self):
        """
        Returns the hierarchical graph structure for frontend visualization.
        """
        return self.stag.get_serializable_structure()
