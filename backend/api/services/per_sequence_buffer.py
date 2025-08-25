import random
import numpy as np
from collections import namedtuple, deque
import torch

Experience = namedtuple('Experience',
                        ('h', 'z', 'activation_path', 'obs', 'action', 'log_prob', 'reward', 'next_obs', 'done'))

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
    def __init__(self, capacity, sequence_length=50, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
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
        Samples a batch of sequences. If num_models > 1, it returns a list of
        batches, one for each model, sampled with replacement.
        """
        if num_models == 1:
            return self._sample_batch(batch_size)
        else:
            # For ensembles, we don't use PER for now, as it complicates things.
            # We just sample with replacement.
            batches = []
            for _ in range(num_models):
                indices = np.random.choice(len(self.memory) - self.sequence_length + 1, size=batch_size, replace=True)
                batch = [self.memory[i : i + self.sequence_length] for i in indices]
                batches.append(self._format_batch(batch, batch_size))
            return batches

    def _format_batch(self, batch, batch_size):
        """Formats a batch of sequences into a dictionary of numpy arrays."""
        batch_dict = {}
        for key in Experience._fields:
            if key in ['obs', 'next_obs']:
                batch_dict[key] = np.stack([getattr(exp, key) for seq in batch for exp in seq])
            else:
                data = [getattr(exp, key) for seq in batch for exp in seq]
                if data and isinstance(data[0], torch.Tensor):
                    batch_dict[key] = torch.stack(data).detach().cpu().numpy()
                else:
                    batch_dict[key] = np.array(data)

        for key, value in batch_dict.items():
            if value.size > 0:
                batch_dict[key] = value.reshape(batch_size, self.sequence_length, *value.shape[1:])

        return batch_dict


    def _sample_batch(self, batch_size):
        """Samples a single batch of sequences using PER."""
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        valid_indices = [i for i in range(len(self.memory) - self.sequence_length + 1)]
        if not valid_indices:
            return None, None, None

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

        batch = [self.memory[i : i + self.sequence_length] for i in indices]

        batch_dict = self._format_batch(batch, batch_size)

        return batch_dict, indices, np.array(weights, dtype=np.float32)


    def update_priorities(self, batch_indices, batch_priorities):
        """Update priorities of sampled sequences."""
        for idx, priority in zip(batch_indices, batch_priorities):
            priority = np.abs(priority) + 1e-5
            self.priorities.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)
