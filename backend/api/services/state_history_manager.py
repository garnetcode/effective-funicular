import os
import json
import torch
import numpy as np
import datetime

class StateHistoryManager:
    """
    Manages the versioned history of an agent's state using a Git-like
    base-and-diff snapshot system to save memory.
    """
    def __init__(self, agent_id, storage_root='backend/storage', base_snapshot_interval=10):
        self.agent_id = agent_id
        self.storage_dir = os.path.join(storage_root, agent_id + "_history")
        self.history_file = os.path.join(self.storage_dir, 'history.json')
        self.base_snapshot_interval = base_snapshot_interval
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_version_path(self, version):
        return os.path.join(self.storage_dir, f"v{version}.pt")

    def _read_history(self):
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [] # Return empty list if history is corrupted

    def _write_history(self, history):
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def save_snapshot(self, state_dict, version_info={}):
        """
        Saves a snapshot of the agent's state, deciding whether to save a
        full 'base' snapshot or a memory-efficient 'diff'.
        """
        history = self._read_history()
        version = len(history)

        def compute_diff(new_dict, old_dict):
            """Recursively computes the difference between two state dicts."""
            diff = {}
            for key, value in new_dict.items():
                if isinstance(value, torch.Tensor):
                    diff[key] = value - old_dict.get(key, 0)
                elif isinstance(value, dict):
                    diff[key] = compute_diff(value, old_dict.get(key, {}))
                else:
                    diff[key] = value # Non-tensor data is stored in full in the diff
            return diff

        snapshot_type = 'base'
        data_to_save = state_dict

        if version > 0 and version % self.base_snapshot_interval != 0:
            snapshot_type = 'diff'
            try:
                prev_state_dict = self.load_snapshot(version - 1)
                if prev_state_dict:
                    data_to_save = compute_diff(state_dict, prev_state_dict)
                else: # Fallback if previous state can't be loaded
                    snapshot_type = 'base'
            except Exception as e:
                print(f"Warning: Could not create diff for version {version}. Saving as base. Error: {e}")
                snapshot_type = 'base'

        torch.save(data_to_save, self._get_version_path(version))

        # Update and write the history log
        history.append({
            'version': version,
            'type': snapshot_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'info': version_info
        })
        self._write_history(history)
        return version

    def load_snapshot(self, version='latest'):
        """
        Loads a specific version of the agent's state, reconstructing it
        from base and diff snapshots as needed.
        """
        history = self._read_history()
        if not history:
            return None

        if version == 'latest':
            target_version = len(history) - 1
        else:
            target_version = int(version)

        if target_version < 0 or target_version >= len(history):
            raise ValueError(f"Version {target_version} not found in history.")

        # Find the closest preceding base snapshot
        start_version = -1
        for i in range(target_version, -1, -1):
            if history[i]['type'] == 'base':
                start_version = i
                break

        if start_version == -1:
            # This can happen if history is corrupted and has no base snapshots
            return None

        # Load the base snapshot
        try:
            current_state_dict = torch.load(self._get_version_path(start_version))
        except FileNotFoundError:
            return None # Base snapshot file is missing

        def apply_diff(current_dict, diff_dict):
            """Recursively applies a diff to a state dict."""
            for key, value in diff_dict.items():
                if key not in current_dict:
                    current_dict[key] = value
                elif isinstance(value, torch.Tensor):
                    current_dict[key] += value
                elif isinstance(value, dict):
                    apply_diff(current_dict.get(key, {}), value)
                else: # For non-tensor data, the diff is the full new value
                    current_dict[key] = value

        # Apply subsequent diffs
        for i in range(start_version + 1, target_version + 1):
            try:
                diff_dict = torch.load(self._get_version_path(i))
                apply_diff(current_state_dict, diff_dict)
            except FileNotFoundError:
                print(f"Warning: Diff file for version {i} not found. State may be inaccurate.")
                continue

        return current_state_dict
