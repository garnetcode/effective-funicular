import random
import numpy as np
from collections import namedtuple, deque
import torch

Experience = namedtuple('Experience',
                        ('h', 'z', 'activation_path', 'obs', 'action', 'log_prob', 'reward', 'next_obs', 'done', 'goal'))

class SegmentTree:
    """A Segment Tree data structure for efficient sum-based operations."""
    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(2 * size)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        self.tree[parent] = self.tree[2 * parent + 1] + self.tree[2 * parent + 2]
        if parent > 0:
            self._propagate(parent)

    def update(self, idx, value):
        idx += self.size
        self.tree[idx] = value
        self._propagate(idx)

    def find(self, value):
        idx = 0
        while idx < self.size:
            left = 2 * idx + 1
            right = 2 * idx + 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - self.size

    def total(self):
        return self.tree[0]

class PERSequenceBuffer:
    def __init__(self, capacity, sequence_length=50, alpha=0.6, beta_start=0.4, beta_frames=100000, her_replay_strategy='future', her_replay_k=4):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.her_replay_strategy = her_replay_strategy
        self.her_replay_k = her_replay_k
        self.frame = 0
        self.memory = []
        self.priorities = SegmentTree(capacity)
        self.max_priority = 1.0
        self.position = 0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *args):
        """Saves an experience."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience(*args)
        # The priority of a new experience is set to the max priority.
        # Note: The priority is associated with the *starting index* of a sequence.
        # When a new experience is added, it can be the start of a new sequence.
        self.priorities.update(self.position, self.max_priority ** self.alpha)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, num_models=1):
        """
        Samples a batch of sequences with HER.
        """
        # For now, we don't support HER with ensembles.
        if num_models > 1:
             # For ensembles, we don't use PER for now, as it complicates things.
            # We just sample with replacement.
            batches = []
            for _ in range(num_models):
                indices = np.random.choice(len(self.memory) - self.sequence_length + 1, size=batch_size, replace=True)
                batch = [self.memory[i : i + self.sequence_length] for i in indices]
                batches.append(self._format_batch(batch, batch_size))
            return batches

        indices, weights = self._sample_indices(batch_size)
        if indices is None:
            return None, None, None

        batch = [self.memory[i : i + self.sequence_length] for i in indices]

        num_relabeled = int(batch_size / (self.her_replay_k + 1))

        relabeled_indices = np.random.choice(batch_size, num_relabeled, replace=False)

        for i in relabeled_indices:
            sequence = batch[i]
            if self.her_replay_strategy == 'future':
                # Sample a future state from the same sequence as the new goal
                future_idx = np.random.randint(self.sequence_length)
                new_goal = sequence[future_idx].next_obs
            else: # 'final'
                new_goal = sequence[-1].next_obs

            # Relabel the sequence with the new goal and recalculate rewards
            new_sequence = []
            for exp in sequence:
                # The new reward is the negative distance to the new goal
                # We assume the goal is in the observation space for simplicity
                new_reward = -np.linalg.norm(exp.obs - new_goal)
                new_exp = exp._replace(goal=new_goal, reward=new_reward)
                new_sequence.append(new_exp)
            batch[i] = new_sequence

        return self._format_batch(batch, batch_size), indices, weights

    def _sample_indices(self, batch_size):
        """Samples a single batch of indices using PER."""
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        valid_indices = [i for i in range(len(self.memory) - self.sequence_length + 1)]
        if not valid_indices:
            return None, None

        total_priority = self.priorities.total()
        if total_priority == 0:
            indices = np.random.choice(valid_indices, size=batch_size)
        else:
            indices = []
            segment = total_priority / batch_size
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                value = random.uniform(a, b)
                idx = self.priorities.find(value)
                indices.append(idx)

        priorities = np.array([self.priorities.tree[i + self.priorities.size] for i in indices])
        sampling_probs = priorities / total_priority

        weights = (len(valid_indices) * sampling_probs) ** -beta
        weights /= weights.max()

        return indices, np.array(weights, dtype=np.float32)

    def _format_batch(self, batch, batch_size):
        """Formats a batch of sequences into a dictionary of numpy arrays."""
        batch_dict = {}
        # Extract all data first
        data_by_key = {key: [getattr(exp, key) for seq in batch for exp in seq] for key in Experience._fields}

        for key, data in data_by_key.items():
            if key in ['obs', 'next_obs', 'goal']:
                # These are expected to be numpy arrays and can be stacked.
                # Goal is included here because it's a numpy array by design now.
                batch_dict[key] = np.stack(data)
            elif key in ['h', 'z', 'log_prob']:
                # These are tensors.
                processed_data = []
                for d in data:
                    t = d if isinstance(d, torch.Tensor) else torch.tensor(d)
                    if t.dim() == 0:
                        t = t.reshape(1)  # Reshape scalar tensors
                    processed_data.append(t)
                batch_dict[key] = torch.stack(processed_data).detach().cpu().numpy()
            elif key == 'activation_path':
                # This is a list of lists/dicts and is not used in a numeric context.
                # Store it as a raw list.
                batch_dict[key] = data
            else:
                # This handles 'action', 'reward', 'done'.
                # These should be simple numeric types.
                try:
                    batch_dict[key] = np.array(data)
                except ValueError as e:
                    print(f"ERROR formatting key '{key}': {e}")
                    # Provide more context on the data that failed
                    print(f"Data sample for '{key}': {[type(d) for d in data[:5]]}")
                    raise

        for key, value in batch_dict.items():
            # Skip reshaping for keys that are not numpy arrays or are empty.
            if isinstance(value, np.ndarray) and value.size > 0:
                batch_dict[key] = value.reshape(batch_size, self.sequence_length, *value.shape[1:])

        return batch_dict

    def update_priorities(self, batch_indices, batch_priorities):
        """Update priorities of sampled sequences."""
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = np.abs(priority) + 1e-5
            self.priorities.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)
