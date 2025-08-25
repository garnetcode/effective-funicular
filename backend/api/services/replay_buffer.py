import random
from collections import namedtuple, deque
import numpy as np
import torch

Experience = namedtuple('Experience',
                        ('h', 'z', 'activation_path', 'obs', 'action', 'log_prob', 'reward', 'next_obs', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, sequence_length=50):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.memory = []
        self.episode_ends = []
        self.position = 0

    def push(self, *args):
        """Saves an experience."""
        e = Experience(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = e
        self.position = (self.position + 1) % self.capacity
        if e.done:
            self.episode_ends.append(self.position)

    def sample(self, batch_size, num_models=1):
        """
        Samples a batch of sequences. If num_models > 1, it returns a list of
        batches, one for each model, sampled with replacement.
        """
        if num_models == 1:
            return self._sample_batch(batch_size)
        else:
            return [self._sample_batch(batch_size, with_replacement=True) for _ in range(num_models)]

    def _sample_batch(self, batch_size, with_replacement=False):
        """Helper method to sample a single batch of sequences."""
        batch = []
        for _ in range(batch_size):
            while True:
                if with_replacement:
                    start_index = random.randint(0, len(self.memory) - self.sequence_length)
                else:
                    # This logic for avoiding episode boundaries is simple and might be slow.
                    # A more robust implementation would pre-calculate valid start indices.
                    start_index = random.randint(0, len(self.memory) - self.sequence_length)

                crosses_boundary = False
                for end_pos in self.episode_ends:
                    if start_index < end_pos < start_index + self.sequence_length:
                        crosses_boundary = True
                        break
                if not crosses_boundary:
                    break

            sequence = self.memory[start_index : start_index + self.sequence_length]
            batch.append(sequence)

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


    def __len__(self):
        return len(self.memory)
