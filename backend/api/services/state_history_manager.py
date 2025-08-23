import os
import json
import torch
import numpy as np
import datetime

class StateHistoryManager:
    """
    Manages the versioned history of an agent's state using a Git-like
    base-and-diff snapshot system to save memory. Includes history pruning.
    """
    def __init__(self, agent_id, storage_root='backend/storage', base_snapshot_interval=10, max_snapshots=100):
        self.agent_id = agent_id
        self.storage_dir = os.path.join(storage_root, agent_id + "_history")
        self.history_file = os.path.join(self.storage_dir, 'history.json')
        self.base_snapshot_interval = base_snapshot_interval
        self.max_snapshots = max_snapshots
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
            return []

    def _write_history(self, history):
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def _prune_history(self):
        """Deletes the oldest snapshots if the history exceeds max_snapshots."""
        history = self._read_history()
        if len(history) <= self.max_snapshots:
            return

        # Find the indices of all base snapshots
        base_indices = [i for i, snap in enumerate(history) if snap['type'] == 'base']

        # We need at least two base snapshots to safely prune.
        # The first base snapshot is the anchor for the current history segment.
        if len(base_indices) < 2:
            return

        # We can safely delete everything up to the second base snapshot.
        # This preserves the first base snapshot and all its subsequent diffs.
        prune_until_index = base_indices[1]

        snapshots_to_delete = history[:prune_until_index]
        remaining_history = history[prune_until_index:]

        # Delete the old snapshot files
        for snap in snapshots_to_delete:
            version = snap['version']
            filepath = self._get_version_path(version)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as e:
                    print(f"Error deleting old snapshot file {filepath}: {e}")

        # Update the history file with the pruned list
        self._write_history(remaining_history)
        print(f"Pruned {len(snapshots_to_delete)} old snapshots from history.")

    def save_snapshot(self, state_dict, version_info={}):
        """
        Saves a snapshot of the agent's state, then prunes old history.
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
                    diff[key] = value
            return diff

        snapshot_type = 'base'
        data_to_save = state_dict

        # The logic for diffing should be based on the *last version in the current history*, not absolute version
        last_version_in_history = history[-1]['version'] if history else -1

        if version > 0 and (version % self.base_snapshot_interval != 0):
            snapshot_type = 'diff'
            try:
                # Load relative to the last known version
                prev_state_dict = self.load_snapshot(last_version_in_history)
                if prev_state_dict:
                    data_to_save = compute_diff(state_dict, prev_state_dict)
                else:
                    snapshot_type = 'base'
            except Exception as e:
                print(f"Warning: Could not create diff for version {version}. Saving as base. Error: {e}")
                snapshot_type = 'base'

        # The new version number is always one greater than the last
        new_version = last_version_in_history + 1
        torch.save(data_to_save, self._get_version_path(new_version))

        history.append({
            'version': new_version,
            'type': snapshot_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'info': version_info
        })
        self._write_history(history)

        # Prune history after saving the new snapshot
        self._prune_history()

        return new_version

    def load_snapshot(self, version='latest'):
        """
        Loads a specific version of the agent's state, reconstructing it
        from base and diff snapshots as needed.
        """
        history = self._read_history()
        if not history:
            return None

        if version == 'latest':
            target_entry = history[-1]
        else:
            target_entry = next((snap for snap in history if snap['version'] == int(version)), None)

        if not target_entry:
            raise ValueError(f"Version {version} not found in history.")

        # Find the closest preceding base snapshot in the current history
        base_entry = None
        for snap in reversed(history):
            if snap['version'] <= target_entry['version'] and snap['type'] == 'base':
                base_entry = snap
                break

        if not base_entry:
            return None

        try:
            current_state_dict = torch.load(self._get_version_path(base_entry['version']))
        except FileNotFoundError:
            return None

        def apply_diff(current_dict, diff_dict):
            """Recursively applies a diff to a state dict."""
            for key, value in diff_dict.items():
                if key not in current_dict:
                    current_dict[key] = value
                elif isinstance(value, torch.Tensor):
                    current_dict[key] += value
                elif isinstance(value, dict):
                    apply_diff(current_dict.get(key, {}), value)
                else:
                    current_dict[key] = value

        # Apply subsequent diffs
        start_index = history.index(base_entry)
        target_index = history.index(target_entry)
        for i in range(start_index + 1, target_index + 1):
            try:
                diff_dict = torch.load(self._get_version_path(history[i]['version']))
                apply_diff(current_state_dict, diff_dict)
            except FileNotFoundError:
                print(f"Warning: Diff file for version {history[i]['version']} not found. State may be inaccurate.")
                continue

        return current_state_dict
