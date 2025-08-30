import os
import json
import torch
import numpy as np
import datetime

class StateHistoryManager:
    """
    Manages saving and loading a single snapshot of an agent's state.
    This simplified manager handles only one file per agent, representing
    the agent's "skill" for a given environment.
    """
    def __init__(self, agent_id, storage_root='backend/storage', **kwargs):
        self.agent_id = agent_id
        self.storage_dir = os.path.join(storage_root, "skills")
        # The snapshot file is named after the agent, which is tied to the environment.
        self.snapshot_file = os.path.join(self.storage_dir, f"{self.agent_id}_skill.pt")
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_snapshot(self, state_dict, version_info={}):
        """
        Saves a snapshot of the agent's state, overwriting the previous one.
        The version_info parameter is kept for compatibility but is not used.
        """
        try:
            torch.save(state_dict, self.snapshot_file)
            # Add a print statement for user feedback
            print(f"Saved best model for agent '{self.agent_id}' to {self.snapshot_file}")
        except Exception as e:
            print(f"Error saving snapshot for agent '{self.agent_id}': {e}")

    def load_snapshot(self, version='latest'):
        """
        Loads the single snapshot file for the agent if it exists.
        The version parameter is kept for compatibility but is not used.
        """
        if not os.path.exists(self.snapshot_file):
            return None

        try:
            # Use map_location to ensure model can be loaded on CPU if saved on GPU
            return torch.load(self.snapshot_file, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Error loading snapshot from {self.snapshot_file}: {e}")
            return None

    def _read_history(self):
        """
        Dummy method for compatibility with existing calls in ChimeraAgent.__init__.
        The new system does not use a history file, but the agent checks this
        to see if it should load a saved state.
        """
        if os.path.exists(self.snapshot_file):
            # Return a dummy history entry to signal that a saved state exists.
            return [{'version': 0, 'type': 'base'}]
        return []
